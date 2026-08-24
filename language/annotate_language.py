"""Per-utterance language annotation for a BabyView release (Gemini on Vertex).

WHY THIS EXISTS
The Airtable `percent_english` is a household survey: measured against the recordings it
correlates only r=0.47, and children recorded as "0% English" are 96-98% English on tape. Any
analysis that needs to know what language a child actually heard needs per-utterance labels.

WHY CHUNKED, NOT PER-UTTERANCE
43% of utterances are 1-2 words ("Mama", "Yeah", "No"), which no classifier can label in
isolation -- fastText lid.176 gets only 58% precision on its own non-English calls here, with
the classic short-string false positives (it/fr/es/de/tr). Sending a numbered BLOCK of
consecutive utterances gives the model the surrounding conversation, and asks for an extractive
label per line rather than an arithmetic summary (LLMs are unreliable at counting over long
inputs, especially with thinking off).

CONFIDENCE BY CONSTRUCTION
Every utterance is labelled twice, in two passes whose chunk boundaries are OFFSET by half a
block, so the two passes see different context windows. Agreement between them is the confidence
signal; disagreements are the set worth escalating or hand-checking.

OUTPUT  <out>/lang_<release>.parquet
  video_id, utterance_id, text, nwords, lang, mixed, undecidable, pass, chunk_id
one row per (utterance, pass); collapse with agree_passes.py.

usage:
  python annotate_language.py --transcript <parsed.csv> --release 2025.2 --out lang/ --pass 0
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from google import genai
from google.genai import types

SYSTEM = (
    "You label the language of utterances transcribed from home audio recordings of families "
    "with young children. The text is ASR output: fragmentary, sometimes misspelled, sometimes "
    "code-switched, and often only one or two words. You are given a numbered block of "
    "CONSECUTIVE utterances from one recording so that you can use the surrounding conversation "
    "to judge the short ones."
)
PROMPT = (
    "For EVERY numbered line, return its language. Use the surrounding lines as context: a bare "
    '"mama" or "no" in an otherwise Portuguese conversation is Portuguese.\n'
    "Return JSON: {\"labels\": [{\"i\": <line number>, \"lang\": \"<ISO 639-1, e.g. en es pt ja ko zh>\", "
    "\"mixed\": <true if the line itself contains more than one language>, "
    "\"undecidable\": <true if genuinely impossible even with context>}]}\n"
    "Return one entry per line, in order, and do not skip lines.\n\n"
)


def build_chunks(df, size, offset):
    """Consecutive utterances within a video, blocks of `size`, started `offset` in."""
    out = []
    for vid, g in df.groupby("video_id", sort=False):
        g = g.sort_values("utterance_id")
        rows = list(g.itertuples())
        start = 0
        if offset and len(rows) > offset:
            out.append((f"{vid}#pre", rows[:offset]))
            start = offset
        for i in range(start, len(rows), size):
            out.append((f"{vid}#{i}", rows[i:i + size]))
    return [(cid, r) for cid, r in out if r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", default="lang")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--pass", dest="pass_", type=int, default=0, help="0 or 1; 1 offsets chunks")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="chunks, for a smoke test")
    ap.add_argument("--child", default="", help="restrict to one subject id (validation)")
    a = ap.parse_args()

    cols = ["video_id", "utterance_id", "utterance"]
    df = pd.read_csv(a.transcript, usecols=cols).drop_duplicates(["video_id", "utterance_id"])
    df = df.rename(columns={"utterance": "text"})
    df["text"] = df.text.astype(str).str.strip()
    df = df[df.text.str.len() > 0].reset_index(drop=True)
    df["nwords"] = df.text.str.split().str.len()
    if a.child:
        df = df[df.video_id.str.startswith(a.child)].reset_index(drop=True)
        print(f"  restricted to {a.child}: {len(df):,} utterances", flush=True)
    chunks = build_chunks(df, a.chunk, a.chunk // 2 if a.pass_ else 0)
    if a.limit:
        chunks = chunks[:a.limit]
    print(f"{len(df):,} utterances -> {len(chunks):,} chunks (pass {a.pass_})", flush=True)

    os.makedirs(a.out, exist_ok=True)
    ckpt = f"{a.out}/lang_{a.release}_p{a.pass_}.jsonl"
    done = set()
    if os.path.exists(ckpt):
        for line in open(ckpt):
            try: done.add(json.loads(line)["chunk_id"])
            except Exception: pass
    todo = [c for c in chunks if c[0] not in done]
    print(f"  {len(done):,} chunks already done, {len(todo):,} to do", flush=True)

    client = genai.Client()          # Vertex; config from env (GOOGLE_GENAI_USE_VERTEXAI etc.)
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, temperature=0.0, response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=-1 if a.thinking else 0))
    lock = threading.Lock()
    fh = open(ckpt, "a")
    usage = {"in": 0, "out": 0, "calls": 0}

    def do(item):
        cid, rows = item
        body = "\n".join(f"{i+1}. {r.text}" for i, r in enumerate(rows))
        for attempt in range(5):
            try:
                r = client.models.generate_content(model=a.model, contents=PROMPT + body, config=cfg)
                u = getattr(r, "usage_metadata", None)
                if u is not None:
                    with lock:
                        usage["in"] += getattr(u, "prompt_token_count", 0) or 0
                        usage["out"] += getattr(u, "candidates_token_count", 0) or 0
                        usage["calls"] += 1
                labs = json.loads(r.text).get("labels", [])
                by_i = {int(x["i"]): x for x in labs if isinstance(x, dict) and "i" in x}
                recs = []
                for i, row in enumerate(rows):
                    L = by_i.get(i + 1, {})
                    recs.append(dict(chunk_id=cid, video_id=row.video_id,
                                     utterance_id=int(row.utterance_id), text=row.text,
                                     nwords=int(row.nwords), lang=L.get("lang"),
                                     mixed=bool(L.get("mixed", False)),
                                     undecidable=bool(L.get("undecidable", False)),
                                     returned=bool(L)))
                with lock:
                    for rec in recs: fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                return len(recs)
            except Exception as e:
                if attempt == 4:
                    with lock:
                        fh.write(json.dumps(dict(chunk_id=cid, error=str(e)[:150])) + "\n"); fh.flush()
                    return 0
                time.sleep(2 ** attempt)

    t0, n = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, got in enumerate(ex.map(do, todo)):
            n += got
            if k % 200 == 0 and k:
                rate = n / max(time.time() - t0, 1)
                print(f"  {k:,}/{len(todo):,} chunks | {n:,} utterances | {rate:.0f}/s", flush=True)
    fh.close()
    if usage["calls"]:
        per = usage["in"] / usage["calls"], usage["out"] / usage["calls"]
        print(f"\nTOKENS: {usage['calls']:,} calls | in {usage['in']:,} ({per[0]:.0f}/call) "
              f"| out {usage['out']:,} ({per[1]:.0f}/call)", flush=True)
        json.dump({**usage, "chunk": a.chunk}, open(f"{a.out}/usage_{a.release}_p{a.pass_}.json", "w"))

    recs = [json.loads(l) for l in open(ckpt)]
    out = pd.DataFrame([r for r in recs if "error" not in r])
    out["pass"] = a.pass_
    p = f"{a.out}/lang_{a.release}_p{a.pass_}.parquet"
    out.to_parquet(p, index=False)
    miss = (~out.returned).sum() if "returned" in out else 0
    print(f"wrote {p}: {len(out):,} utterances, {miss:,} lines the model failed to return")
    if len(out):
        print(out.lang.value_counts(normalize=True).head(6).mul(100).round(2).to_string())


if __name__ == "__main__":
    main()
