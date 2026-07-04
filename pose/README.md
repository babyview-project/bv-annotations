# pose — whole-body pose annotation

Per frame, per detected person: **133-keypoint COCO-WholeBody** pose (17 body + 6 feet +
68 face + 42 hands) with per-keypoint confidence. This is the annotation neuroscience/social-
cue work cares about (hands for pointing/holding, dense face for gaze).

Pipeline: **YOLO12x** person detector → **mmpose RTMW-x cocktail14 (256×192)** top-down pose.
Adapted from Khai Aw's original `babyview-pose` pipeline (`/ccn2/u/khaiaw/Code/babyview-pose`);
`run_model.py` here is that script with three production patches (below).

## Output format

One pickle per frame, mirroring the input layout:
`/ccn2/dataset/babyview/<RELEASE>/outputs/pose_1fps/<video_id>/<frame:05d>.pkl`

```python
{
  "person_detection_dict": {"person_bboxes": np.ndarray[N,4] xyxy,
                            "person_confs":  np.ndarray[N]},      # YOLO12x
  "pose_dict": { person_id: {"keypoints": [133,2], "keypoint_scores": [133]}, ... }
}
```
Read it with `pose_lib.py` (`load_pose(path)` → `{bboxes, confs, persons:[{kp,score}]}`,
handles the CUDA-tensor pickles). Keypoint **scores are [0,1]** and well-calibrated — false
positives get near-zero scores (see [qa/README.md](qa/README.md) for why this matters and why
we did *not* use the faster distilled `rtmw-dw-x-l` model).

Frames with no detected person get a pkl with empty `pose_dict` (so the run is resumable and
"processed but empty" is distinguishable from "not yet done").

## Run it on a release

1. Build the env once (finicky — mmcv must match torch): see [SETUP.md](SETUP.md).
2. Point the (patched) script at the release's frames:

```bash
cd /data2/mcfrank/babyview-pose          # working copy (models symlinked from khaiaw's downloads/)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /data2/mcfrank/mmpose_env/bin/python run_model.py \
    --input_dir  /ccn2a/dataset/babyview/<RELEASE>/extracted_frames_1fps/ \
    --output_dir /ccn2/dataset/babyview/<RELEASE>/outputs/pose_1fps \
    --num_processes 32          # 4 Ray workers/GPU × 8 GPUs
```

- **Resumable:** skips frames whose pkl already exists. If it dies (bad node, preemption),
  just rerun the same command.
- **Throughput:** ~35 fps/GPU → ~5 h for 2025.2's 5.42M frames on 8 A40. Scale `--num_processes`
  to `4 × #GPUs`.
- Detached long run: `setsid bash -c "… > /data2/mcfrank/pose_full.log 2>&1" < /dev/null &`,
  then watch with `find <output> -name '*.pkl' | wc -l` and `tail pose_full.log`.

## Patches vs khaiaw's original `run_model.py`

Kept minimal; each traces to a real failure on 2025.2 data:

1. **Drop zero-area detections** — YOLO occasionally emits a degenerate box; the pose
   `warpAffine` crashes on a 0-size crop. Filter `w>1 & h>1` before pose.
2. **`try/except` around `inference_topdown`** — a corrupt/edge frame no longer kills the
   whole Ray worker (and its chunk); it's logged (`POSE FAIL …`) and saved as no-detection.
3. **Visualization disabled** (`if i % 1000` → `if False`) — the every-1000th-frame overlay
   crashed in `revert_heatmap`/`warpAffine`, cost time, and wrote **face jpgs** into the
   output. We don't want rendered frames in the annotation dataset.

## Post-run

`create_csv_from_pkl.py` (in khaiaw's repo) flattens the pkls to a CSV if a table is wanted.

## Files

- `run_model.py` — patched inference (YOLO12x + mmpose RTMW-x, Ray-parallel).
- `pose_lib.py` — canonical reader + COCO-WholeBody index constants + a PIL overlay helper.
- `requirements.txt` / `SETUP.md` — the env.
- `qa/` — old-vs-new / model-comparison review tooling and the calibration write-up.
