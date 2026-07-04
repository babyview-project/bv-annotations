"""Score (utterance, frame) pairs for referential alignment with Gemini on Vertex AI.

For each pair we ask Gemini: does this utterance refer to a concrete object that is
*visible* in this frame, and if so which? The returned `alignment` (0-1) is a drop-in,
stronger replacement for CLIP's clip_score_max as a training filter / soft label; the
returned `referent` is a candidate clean label (a soft version of the @sec-topline
label-injection oracle).

Auth: uses Vertex AI via the google-genai SDK, which reads these env vars (set in the
ccn2 .env / service-account setup) — nothing is hard-coded here:
    GOOGLE_GENAI_USE_VERTEXAI=true
    GOOGLE_CLOUD_PROJECT=hs-hs-langcog-gemini
    GOOGLE_CLOUD_LOCATION=us-central1
    GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/vlm-headcam-sa.json

Human subjects: frames contain faces. This sends them to Vertex (IRB-approved, Vertex
data-governance terms — not used to train Google models). Do NOT repoint this at the
AI Studio (generativelanguage) endpoint.

Robustness: concurrent, resumable (checkpoints to a JSONL sidecar; rerun skips done
keys), retries with backoff on rate limits. Run a small --limit first to sanity-check
the prompt, then scale.

Usage:
    python src/gemini_align.py --manifest manifests/aligned_S00360001.parquet \
        --out scored/gemini_aligned_S00360001.parquet --limit 300      # sanity pass
    python src/gemini_align.py --manifest manifests/unfiltered_S00360001.parquet \
        --out scored/gemini_unfiltered.parquet --workers 32            # full run
"""
import argparse
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from google import genai
from google.genai import types
from pydantic import BaseModel

from common import frame_path

SYSTEM = (
    "You annotate egocentric video from a head-mounted camera worn by a young child. "
    "Each image is one video frame; the text is a caregiver's or child's utterance "
    "transcribed from that same moment. Speech is often not about anything on screen. "
    "Judge how strongly the utterance refers to a concrete, physical object that is "
    "actually VISIBLE in this frame."
)

# Kept terse on purpose: short output = low output-token cost over ~1M calls.
PROMPT = (
    "Return JSON with:\n"
    '- "alignment": integer 0-100 for how strongly the utterance refers to a concrete '
    "object VISIBLE in the frame. 100 = clearly names a prominent, plainly-visible, "
    "central object; ~50 = the named object is present but small, partial, or one of "
    "many; 0 = no visible referent (small talk, the object is absent or unidentifiable, "
    "or the utterance is not about a concrete object). Use the full range, not just the "
    "ends.\n"
    '- "referent": the single visible object referred to, as a lowercase common noun '
    '(e.g. "cup", "dog"); "" if alignment is low or none applies.\n'
    "Utterance: "
)


class Align(BaseModel):
    alignment: int
    referent: str


def load_jpeg(video_id, frame_idx, max_px):
    """Load the 1fps frame, downsize longest edge to max_px, return JPEG bytes."""
    im = Image.open(frame_path(video_id, frame_idx)).convert("RGB")
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def score_one(client, model, row, max_px, thinking):
    """Score a single pair. Returns a result dict (never raises)."""
    base = {"video_id": row.video_id, "frame_idx": int(row.frame_idx), "text": row.text}
    try:
        jpg = load_jpeg(row.video_id, row.frame_idx, max_px)
    except Exception as e:
        return {**base, "alignment": None, "referent": None, "error": f"img:{e}"}

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=Align,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=(-1 if thinking else 0)),
    )
    contents = [types.Part.from_bytes(data=jpg, mime_type="image/jpeg"),
                PROMPT + str(row.text)]

    for attempt in range(6):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=cfg)
            a = r.parsed if r.parsed is not None else Align(**json.loads(r.text))
            al = max(0, min(100, int(round(a.alignment))))
            return {**base, "alignment": al, "referent": (a.referent or "").strip().lower(),
                    "error": None}
        except Exception as e:
            msg = str(e)
            transient = any(c in msg for c in ("429", "503", "500", "RESOURCE_EXHAUSTED",
                                               "UNAVAILABLE", "DEADLINE", "timeout"))
            if attempt == 5 or not transient:
                return {**base, "alignment": None, "referent": None, "error": msg[:200]}
            time.sleep(min(60, 2 ** attempt) + random.random())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output parquet (a .jsonl checkpoint sits beside it)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-px", type=int, default=512, help="downsize longest frame edge")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (0 = all)")
    ap.add_argument("--sample", type=int, default=0, help="random N instead of first N")
    ap.add_argument("--thinking", action="store_true", help="enable model thinking (slower, dearer)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    keep = [c for c in ("video_id", "frame_idx", "text", "clip_score_max", "child_id") if c in man.columns]
    man = man[keep].dropna(subset=["video_id", "frame_idx", "text"]).reset_index(drop=True)
    if args.sample:
        man = man.sample(n=min(args.sample, len(man)), random_state=args.seed).reset_index(drop=True)
    elif args.limit:
        man = man.head(args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix(".jsonl")

    # key on the (frame, utterance) triple: a frame can pair with >1 utterance.
    # Recompute keys from record fields so older checkpoints (any key format) are reused.
    def keys_of(df):
        return (df.video_id.astype(str) + "|" + df.frame_idx.astype(int).astype(str)
                + "|" + df.text.astype(str))
    done = set()
    if ckpt.exists():
        recs = [json.loads(l) for l in open(ckpt)]
        if recs:
            done = set(keys_of(pd.DataFrame(recs)))
    todo = man[~keys_of(man).isin(done)]
    print(f"{len(man)} pairs | {len(done)} already scored | {len(todo)} to do", flush=True)

    client = genai.Client()  # Vertex config comes from env
    lock = threading.Lock()
    t0 = time.time()
    n = 0
    with open(ckpt, "a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, client, args.model, r, args.max_px, args.thinking)
                for r in todo.itertuples(index=False)]
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                n += 1
                if n % 200 == 0:
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-n)/max(rate,1e-9)/60:.0f}m",
                          flush=True)

    # consolidate checkpoint -> parquet, merged with passthrough columns (clip_score_max)
    recs = [json.loads(l) for l in open(ckpt)]
    scored = pd.DataFrame(recs).drop_duplicates(["video_id", "frame_idx", "text"], keep="last")
    scored = scored.drop(columns=[c for c in ("key",) if c in scored.columns])
    merged = man.merge(scored, on=["video_id", "frame_idx", "text"], how="left")
    merged.to_parquet(out, index=False)
    ok = merged["alignment"].notna().sum()
    err = merged["error"].notna().sum() if "error" in merged else 0
    print(f"DONE {out}  scored={ok}  errors={err}", flush=True)
    if ok:
        q = merged["alignment"].describe(percentiles=[.1, .25, .5, .75, .9])
        print(q.to_string(), flush=True)


if __name__ == "__main__":
    main()
