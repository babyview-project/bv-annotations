# alignment — referential-alignment scoring (Gemini / Vertex)

> **Status: 🚧 in progress.** The pipeline runs and is validated at small scale; the full-scale
> scoring + downstream retraining comparison is being developed in a `vlm-headcam` session fork.
> Treat the interface below as stable, the operating points (threshold, scale) as not-yet-final.

For each `(utterance, frame)` pair, ask Gemini how strongly the utterance refers to a concrete
object **visible** in the frame, and which object. This is a stronger, referential replacement
for CLIP's whole-frame similarity as a training filter / soft label.

Output per pair: `alignment` (ordinal 0–100) + `referent` (a lowercase common noun, or "").

## Why an API model (not a local one)

The bottleneck in the downstream science is *referential* signal — which utterance goes with
which frame — not encoder capacity. A strong multimodal model used as a **scorer** (not a
learner) is the direct attack. Gemini on **Vertex** (not AI Studio) because the data are
human-subjects: Vertex's terms don't train on inputs and allow zero/short retention (IRB-
approved for this use). See [SETUP.md](SETUP.md).

## Run it

```bash
# env: reuse any env with google-genai + pillow + pandas (e.g. the rtmlib pose_env has them);
# Vertex config comes from ~/.secrets/vlm-headcam.env (source it first).
set -a; source ~/.secrets/vlm-headcam.env; set +a
python gemini_align.py \
  --manifest <pairs>.parquet \        # cols: video_id, frame_idx, text (+ clip_score_max passthrough)
  --out scored/<name>.parquet \
  --model gemini-2.5-flash \          # Flash: best calibrated cheap model; Flash-Lite piles at one value
  --workers 32                        # concurrent; resumable via a .jsonl checkpoint
```

- **Resumable:** appends to a `.jsonl` sidecar; rerun skips done pairs, consolidates to parquet.
- **Cost (Vertex, Batch is ½):** ~$5 for a 50k validation pass, ~$120 for the full ~1.14M-pair
  unfiltered stream on Flash. Cost is not the constraint; IRB/data-handling is.
- `plot_gemini_val.py` — rating histograms + correlation with CLIP for a validation sample.

## Findings so far (small-scale validation)

- 0–100 ordinal scale needed (a 0–1 float collapsed to {0,1}); **Flash** is well-calibrated,
  **Flash-Lite** piles at ~70. ~88% of unfiltered pairs score 0 (correctly — most speech isn't
  about a visible object). Weak-positive correlation with CLIP (ρ≈0.22) = lots of *independent*
  signal; disagreements favor Gemini (catches book/toy referents CLIP misses, rejects CLIP
  keyword false-positives). Whether that beats CLIP as a training filter is the open retrain test.

## Human subjects

Sends frames (with faces) to Vertex. This is IRB-approved for annotation and must use **Vertex**
(not the `generativelanguage`/AI-Studio endpoint). Never hard-code keys; config is env-only.
