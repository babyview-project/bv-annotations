"""AUDIO-level language ID for a BabyView release (Gemini on Vertex, audio input).

WHY THIS EXISTS
The transcript-level language annotation is blind to ASR auto-translation: Whisper renders
non-English caregiver speech as fluent English text, so transcript labels report ~90% English
even for families that (per Mike's spot checks) are not speaking English at all. Only the
AUDIO can answer what language a child actually heard. This annotator samples windows from
each video's mp3 and asks Gemini to identify the languages SPOKEN, bypassing transcripts.

DESIGN
- 3 windows per video at 20/50/80% of duration, 60s each (shorter videos: what fits).
- One call per window; agreement ACROSS windows within a video is the reliability signal
  (same philosophy as the two-pass transcript design, with time replacing chunk offset).
- Output one row per window: video_id, w (window idx), start_s, langs (json {lang: prop}),
  english_prop, speech (bool). Aggregate per-video/child downstream.
- Resumable via jsonl checkpoint; failures retried on relaunch.

usage:
  python annotate_audio_language.py --release 2026.1 \
      --mp3-root /ccn2b/dataset/babyview/2026.1/mp3 \
      --index /ccn2b/dataset/babyview/2026.1/outputs/release_index.tsv \
      --out /ccn2b/dataset/babyview/2026.1/outputs/annotations/language/audio
"""
import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

SYSTEM = (
    "You identify the languages SPOKEN in short audio clips from home recordings of families "
    "with young children. Judge only from the audio. Ignore singing wordlessly, babbling, and "
    "TV/background media unless it is the only speech present."
)
PROMPT = (
    "Listen to this clip. Return JSON only:\n"
    '{"speech": <true if any intelligible speech>, '
    '"langs": {"<ISO 639-1>": <proportion of the SPOKEN content in that language, summing to 1>}, '
    '"child_directed": <true if most speech is to/with a child>}\n'
    "Example: {\"speech\": true, \"langs\": {\"ko\": 0.8, \"en\": 0.2}, \"child_directed\": true}"
)

ap = argparse.ArgumentParser()
ap.add_argument("--release", default="2026.1")
ap.add_argument("--mp3-root", default="/ccn2b/dataset/babyview/2026.1/mp3")
ap.add_argument("--index", default="/ccn2b/dataset/babyview/2026.1/outputs/release_index.tsv")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gemini-2.5-flash")
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--win", type=int, default=60)
ap.add_argument("--limit", type=int, default=0, help="first N videos only (pilot)")
ap.add_argument("--only-children", default="", help="comma-separated subject_ids (pilot)")
a = ap.parse_args()

OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
ckpt = OUT / f"audio_lang_{a.release}.jsonl"
usage_p = OUT / f"usage_audio_{a.release}.json"

idx = pd.read_csv(a.index, sep="\t")
if a.only_children:
    keep = set(a.only_children.split(","))
    idx = idx[idx.subject_id.isin(keep)]
if a.limit:
    idx = idx.head(a.limit)

done = set()
if ckpt.exists():
    for ln in open(ckpt):
        try:
            r = json.loads(ln)
            if r.get("langs") is not None or r.get("speech") is False:
                done.add((r["video_id"], r["w"]))
        except Exception:
            pass

def duration(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(p)], capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0

jobs = []
for r in idx.itertuples():
    p = Path(a.mp3_root) / r.subject_id / f"{r.video_id}.mp3"
    if not p.exists():
        continue
    jobs.append((r.video_id, r.subject_id, p))
print(f"{len(jobs):,} videos with audio; sampling 3x{a.win}s windows", flush=True)

client = genai.Client()          # Vertex; config from env (GOOGLE_GENAI_USE_VERTEXAI etc.)
cfg = types.GenerateContentConfig(
    system_instruction=SYSTEM, temperature=0.0,
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=0))
lock = threading.Lock()
usage = {"in": 0, "out": 0, "calls": 0}
n_done = [0]

def one(job):
    vid, child, p = job
    dur = duration(p)
    if dur < 5:
        return
    win = min(a.win, int(dur))
    starts = [max(0, dur * f - win / 2) for f in (0.2, 0.5, 0.8)]
    if dur <= win * 1.2:
        starts = [0]
    for w, st in enumerate(starts):
        if (vid, w) in done:
            continue
        try:
            cut = subprocess.run(["ffmpeg", "-nostdin", "-v", "quiet", "-ss", f"{st:.1f}",
                                  "-t", str(win), "-i", str(p), "-acodec", "libmp3lame",
                                  "-ac", "1", "-b:a", "48k", "-f", "mp3", "pipe:1"],
                                 capture_output=True, timeout=120).stdout
            if len(cut) < 2000:
                rec = dict(video_id=vid, child=child, w=w, start_s=round(st, 1),
                           speech=False, langs=None, note="empty-cut")
            else:
                for attempt in range(5):
                    try:
                        r = client.models.generate_content(
                            model=a.model,
                            contents=[types.Part.from_bytes(data=cut, mime_type="audio/mpeg"),
                                      PROMPT],
                            config=cfg)
                        d = json.loads(r.text)
                        langs = d.get("langs") or {}
                        tot = sum(langs.values()) or 1.0
                        langs = {k: round(v / tot, 3) for k, v in langs.items()}
                        rec = dict(video_id=vid, child=child, w=w, start_s=round(st, 1),
                                   speech=bool(d.get("speech")), langs=langs,
                                   child_directed=d.get("child_directed"),
                                   english_prop=langs.get("en", 0.0))
                        with lock:
                            u = r.usage_metadata
                            usage["in"] += u.prompt_token_count or 0
                            usage["out"] += (u.candidates_token_count or 0)
                            usage["calls"] += 1
                        break
                    except Exception as e:
                        if attempt == 4:
                            rec = dict(video_id=vid, child=child, w=w, start_s=round(st, 1),
                                       speech=None, langs=None, note=str(e)[:120])
                        else:
                            time.sleep(2 ** attempt)
            with lock:
                ckpt.open("a").write(json.dumps(rec) + "\n")
                n_done[0] += 1
                if n_done[0] % 500 == 0:
                    print(f"  {n_done[0]:,} windows done", flush=True)
                    usage_p.write_text(json.dumps(usage))
        except Exception as e:
            with lock:
                ckpt.open("a").write(json.dumps(dict(video_id=vid, child=child, w=w,
                                                     speech=None, langs=None,
                                                     note="ffmpeg:" + str(e)[:80])) + "\n")

with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(one, jobs))
usage_p.write_text(json.dumps(usage))

# collapse to parquet
rows = [json.loads(ln) for ln in open(ckpt)]
df = pd.DataFrame(rows).drop_duplicates(["video_id", "w"], keep="last")
df["langs"] = df.langs.map(json.dumps)
df.to_parquet(OUT / f"audio_lang_{a.release}.parquet", index=False)
ok = df[df.speech.notna()]
print(f"AUDIO_LANG_DONE {a.release}: {len(df):,} windows, {df.video_id.nunique():,} videos, "
      f"{100 * ok.speech.mean():.0f}% with speech | usage {usage}", flush=True)
