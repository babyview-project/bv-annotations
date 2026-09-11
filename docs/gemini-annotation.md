# Annotating BabyView with Gemini (via Vertex AI) — a lab guide

How we use Gemini as an *annotator* over BabyView frames, transcripts and audio; what the
rules are (this is human-subjects data); what it costs; and how to add a new annotation
layer so other people can use it. Written 2026-09-11 from the three layers we have run at
release scale (referent alignment on 1.84M utterance–frame pairs, two-pass transcript
language ID on 1.84M utterances, audio language ID on 48k one-minute windows).

## 1. The rule: Vertex AI, never AI Studio

BabyView frames show children and families; transcripts and audio are their speech. Sending
that to a model is IRB-approved **only** through Google Cloud **Vertex AI**, whose terms do
not train on inputs and allow zero/short retention. The consumer endpoint
(`generativelanguage.googleapis.com`, "AI Studio", the free API keys) is **not** allowed for
any BabyView content, including "just a few frames to test a prompt". Use a synthetic or
public image for prompt fiddling if you don't want to set up Vertex first.

Corollaries:
- Frames/audio/transcripts never go into a notebook you'll share, a Google Form, a Slack
  message, GitHub, Hugging Face, or a laptop copy. Annotation runs on ccn2 and the outputs are
  aggregate labels (numbers, words) keyed by video/frame ids.
- Don't put keys in code. Config is environment-only (below).

## 2. Setup (once per person, ~15 minutes)

GCP project: `hs-hs-langcog-gemini` (lab-owned; Mike administers). Vertex authenticates with a
**service account**, not an API key. Full commands: [`alignment/SETUP.md`](../alignment/SETUP.md).
The short version:

1. Ask Mike to create you a service-account key scoped to `roles/aiplatform.user` on that
   project (or to add you as a project user so you can create your own).
2. On ccn2: `mkdir -p ~/.secrets && chmod 700 ~/.secrets`; put the key there
   (`chmod 600 ~/.secrets/*.json`).
3. Create `~/.secrets/<project>.env`:
   ```bash
   export GOOGLE_GENAI_USE_VERTEXAI=true
   export GOOGLE_CLOUD_PROJECT=hs-hs-langcog-gemini
   export GOOGLE_CLOUD_LOCATION=us-central1
   export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.secrets/<your-key>.json
   ```
   and `set -a; source ~/.secrets/<project>.env; set +a` before running anything.
4. Python: `pip install google-genai pillow pandas pyarrow` into any env (the `google-genai`
   SDK reads exactly those four variables; nothing is hard-coded).
5. **Revoke the key when your big run is done** (`gcloud iam service-accounts keys delete`).
   A long-lived broad key on a shared filesystem is the main risk here.

## 3. Designing an annotation

What has worked, and what didn't:

- **Pick the unit and the key first.** Our units: an utterance–frame pair keyed by
  `(video_id, frame_idx, text)`, an utterance keyed by `(video_id, utterance_id)`, a 60 s audio
  window keyed by `(video_id, w)`. `video_id` is the release name
  (`S00220001_2024-02-05_1_rec…`); `frame_idx` = second at 1 fps. **Never join on utterance
  text** — different pipelines segment utterances differently and a text join silently drops
  half the rows while looking like a content decision (this bit us once).
- **Ask for JSON with a fixed schema**, one call per unit, `temperature=0`. Use the SDK's
  `response_schema` (a pydantic model) so malformed outputs fail loudly instead of parsing as
  garbage. Example (referent alignment):
  ```
  Return JSON with:
  - "alignment": integer 0-100 for how strongly the utterance refers to a concrete object
    VISIBLE in the frame. 100 = clearly names a prominent, plainly-visible, central object;
    ~50 = the named object is present but small, partial, or one of many; 0 = no visible
    referent (small talk, the object is absent or unidentifiable, or the utterance is not
    about a concrete object). Use the full range, not just the ends.
  - "referent": the single visible object referred to, as a lowercase common noun
    (e.g. "cup", "dog"); "" if alignment is low or none applies.
  Utterance: <text>
  ```
- **Ordinal 0–100 with anchors, not a 0–1 float.** A float collapsed to {0, 1}; the 0–100
  scale with three described anchors gave a usable distribution (90% zeros, then mass at
  50/70/80/90/100). Expect anchoring at round numbers; treat the score as ordinal.
- **Model:** `gemini-2.5-flash`. Flash-Lite piles everything at one value; Pro is ~10× the
  cost for no visible gain on these tasks. Pin the model name in the output metadata.
- **Give context, but the right context.** For frames: downsize the longest edge (we used
  `max_px` in `gemini_align.py`) — full-res buys nothing. For transcripts: chunk utterances
  with surrounding lines so the model can disambiguate, and vary the chunk offsets between
  passes so chunk boundaries don't align with the labels you compare.
- **Build reliability into the design.** Two independent passes (transcript language:
  99.4% agreement, disagreements → NULL rather than a guess) or several samples per unit
  (audio: 3 windows per video at 20/50/80%, 94% of videos have all windows on the same side
  of 50% English). One call gives you a number; two give you an error bar.
- **Validate against something.** Human ratings on a stratified sample (see
  `vlm-headcam/human_check/`), agreement with an independent signal (CLIP similarity for
  alignment, the household language survey for language ID: r = 0.90 for audio vs 0.53 for
  transcripts), and a hand look at disagreements. The audio layer exists because a hand
  check of a "94% English" family found them speaking Japanese: Whisper had auto-translated
  the transcript. **A transcript is not the speech.**

## 4. Running at scale

- **Resumable checkpointing is not optional.** Every run appends one JSON line per unit to a
  `.jsonl`; a relaunch reads it, skips done units, retries failures, and consolidates to
  parquet at the end. Runs of 1.8M calls take a day and *will* be interrupted.
- **Concurrency:** 32 threads per process was fine on Vertex for Flash; back off on 429s with
  jitter. Log every error with the unit key; at the end, a `failures` count you can report.
- **Cost (Flash, on-demand; Batch prediction is roughly half):**

  | layer | units | cost |
  |---|---|---|
  | referent alignment (frame + short text) | 1.84M pairs | ~$255 |
  | transcript language ID, two passes | 1.84M utterances in 92k chunked calls | ~$250 (56M in / 93M out tokens) |
  | audio language ID | 47.8k × 60 s windows | ~$27 (80M in / 1M out) |

  Flash pricing is ~$0.30 per million input tokens and ~$2.50 per million output tokens, so
  **output tokens dominate**: a verbose per-utterance JSON schema cost 4–5× what a compact one
  would (see `language/README.md`). Write the token accounting to a `usage_*.json` next to
  the output; it's what you'll cite.
- **Detach the job from your ssh session** (`setsid nohup … &`, a log file, a DONE marker
  line) and check progress by counting lines in the jsonl. Multi-command ssh strings get
  killed by dropped connections; put the launcher in a script file on the node.
- **Rate limits and quotas** are per project: tell people in the lab before you launch a
  million-call job.

## 5. Where outputs go, and how to document them

Annotations that others might use go into the release tree, not your scratch:
`/ccn2b/dataset/babyview/<release>/outputs/annotations/<layer>/` with
- the consolidated parquet (one row per unit, keys first),
- the raw jsonl checkpoint(s),
- `usage_*.json`,
- a `README.md` stating: what the label means, the exact model + endpoint + date, the prompt
  (verbatim or a pointer to the script and commit), the join key, coverage against the release
  (e.g. "1,838,061 / 1,838,134 scored"), reliability numbers, known biases, and the consumer
  code. Existing examples: `annotations/referent/README.md`, `annotations/language/README.md`,
  `annotations/language/audio/README.md` (all in the 2026.1 tree; sources in
  `vlm-headcam/release_docs/` and here under `language/`).

The tree is mirrored to Oak (`/oak/stanford/groups/mcfrank/babyview-2026.1-mirror/`), so once
your layer is in the release tree and listed in `MANIFEST.tsv`, it's archived.

## 6. Checklist for a new layer

1. Unit + key defined; the manifest of units built from release files (not from someone's
   intermediate CSV).
2. Prompt with a JSON schema, anchored scale, temperature 0; tried on 200 units; distribution
   looked at (not just the mean).
3. Reliability design: two passes or multiple samples per unit.
4. Validation plan: what independent signal or human sample will you compare to?
5. Vertex env sourced; key scoped; run detached and resumable; usage logged.
6. Output in the release tree with a README; key added to `MANIFEST.tsv`; key revoked.

## 7. Code you can copy

- `alignment/gemini_align.py` — frame + text → JSON (referent alignment); threaded, resumable.
- `language/annotate_language.py` + `agree_passes.py` — chunked transcript → language labels,
  two passes with offset chunks, agreement logic.
- `language/annotate_audio_language.py` — mp3 → 60 s windows (ffmpeg) → language proportions.
- `vlm-headcam/human_check/` — stratified sampling, frame pull with opaque ids, the rating app,
  and the agreement analysis for validating an annotation against people.
