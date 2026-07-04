# pose QA / model choice

Review tooling for sanity-checking pose output, and the write-up of *why* the production
pipeline uses mmpose `rtmw-x` rather than a faster ONNX model.

## The calibration finding (why `rtmw-x`, not distilled `rtmw-dw-x-l`)

We first built a faster stack — YOLO detector + **`rtmw-dw-x-l`** (the distilled RTMW model,
run via [`rtmlib`](https://github.com/Tau-J/rtmlib) as ONNX). It was fast (~40 fps/GPU) but had
a too-high false-positive rate: full skeletons drawn on non-people (a hand at the frame edge, a
doll, a photo on the wall).

The cause is **not the detector** — YOLO12x and YOLO11m fire the *same* false-positive boxes,
and the detector's threshold couldn't fix it (person-like clutter scores high). It's the **pose
model's confidence calibration** on those boxes:

| On the same false-positive box | `rtmw-x` (mmpose, [0,1]) | `rtmw-dw-x-l` (distilled, [0,3]) |
|---|---|---|
| body-median keypoint score | **0.1–0.3** → vanishes at thr 0.3 | 1.3–1.8 → draws at thr 1.0 (overlaps real people) |

The non-distilled `rtmw-x` gives hallucinated skeletons near-zero confidence, so they're
filterable/invisible; the distilled model is overconfident on them and you *can't* threshold
them out (FP scores overlap real detections). Distillation traded calibration for speed. So the
production pipeline uses `rtmw-x` via mmpose — which happens to be exactly khaiaw's original
choice. (An `rtmw-x` ONNX would let us keep mmpose's calibration at rtmlib's speed, but none is
published; exporting one is a possible future optimization.)

## Tooling

- **`pose_compare.py`** — old-vs-new panels drawn with one clean drawer (limbs as lines,
  face/hands as dots — *not* mmpose's dense face-contour, which is unreadable on noisy points).
  `--diverse` samples across children; reports NEW recall on old-person frames and FP rate on
  old-empty frames. Faces present → review on-node or in a gitignored local dir.
- **`pose_infer.py`** — the rtmlib (ONNX) inference path. Not production (calibration above),
  but useful for fast experiments and A/B panels. Notable fixes baked in: bypass rtmlib's
  whole-frame fallback (it fits a skeleton to *every* empty frame otherwise), zero out
  keypoints predicted outside the image, swap the sealed YOLOX-humanart detector for a
  tunable YOLO.

## Quick sanity checks

- Coverage (fraction of frames with ≥1 person) should be ~40–55% on typical BabyView video, not
  ~100% (100% means an empty-frame fallback bug).
- On a known clean person, body-keypoint scores should be well above 0.3; on empty frames, no
  detections (or near-zero scores).
