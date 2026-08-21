# Handoff: whole-body pose over BabyView 2026.1

*Written 2026-08-21 from the 2025.2 run and the state of ccn2 that day. Companion to
`pose/README.md` (pipeline), `pose/SETUP.md` (env), `docs/adding-a-release.md` (generic runbook).*

## TL;DR

- **What we produce:** per frame, per detected person, 133-kpt COCO-WholeBody pose
  (YOLO12x person boxes → mmpose **RTMW-x cocktail14 256×192**), one `.pkl` per frame mirroring the
  frame tree, plus a flat 29-col CSV (one row per detected person). Keypoint scores are [0,1] and
  **calibrated** — false-positive skeletons get ~0 scores, which is the whole reason for this model
  choice (see `pose/qa/README.md`).
- **2025.2 is done** (2026-07-04): 5,419,919 / 5,419,920 frames, 0 worker errors, validated against
  the old 4M-frame run. Outputs: `/ccn2/dataset/babyview/2025.2/outputs/pose_1fps/` (pkls) and
  `…/outputs/pose_1fps_bbox_limbs.csv` (1.4 GB, 6.5M rows).
- **2026.1 is not runnable yet.** On ccn2 it has only `outputs/merged_transcripts_parsed.csv` (8.3M
  token rows) and one child's mp3s — **no `extracted_frames_1fps/`**. Alvin (`tanawm`) is cycling
  videos GCP → ccn → frames/mp3/clips → Oak. Pose needs the 1-fps frames, so it runs per chunk as
  frames land (or once the whole tree exists).
- **The pose env is gone.** `/data2/mcfrank/mmpose_env` no longer exists on node14 — rebuild it
  first (`pose/SETUP.md`, ~30 min, the mmcv/torch pin is the only hard part).
- **Repo drift:** the node copy `~/bv-annotations` is 4 commits behind local (`f8e5186` vs
  `e0ce9e8`); the CSV builder (`pose/create_csv.py`) exists only locally + as
  `/data2/mcfrank/babyview-pose/create_csv_2025_2.py`. No GitHub remote yet.

## The pipeline, concretely

```
/ccn2a/dataset/babyview/<REL>/extracted_frames_1fps/<video_id>/<NNNNN>.jpg    (1 fps, ~61 KB/frame)
        │  run_model.py  (Ray, 4 workers/GPU; YOLO12x classes=[0] → inference_topdown RTMW-x)
        ▼
/ccn2/dataset/babyview/<REL>/outputs/pose_1fps/<video_id>/<NNNNN>.pkl         (~2.7 KB/frame)
        │  create_csv.py  (CPU-unpickle, per-worker part files, stream concat)
        ▼
/ccn2/dataset/babyview/<REL>/outputs/pose_1fps_bbox_limbs.csv                  (1 row / person)
```

- **Code:** working copy `/data2/mcfrank/babyview-pose/run_model.py` = khaiaw's
  `/ccn2/u/khaiaw/Code/babyview-pose/pose-detection/run_model.py` + 3 patches (skip zero-area boxes;
  try/except around `inference_topdown` → `POSE FAIL` log + empty pose; viz disabled — it crashed on
  heatmaps and wrote face jpgs). Canonical copy: `pose/run_model.py` in this repo.
- **Weights:** symlinked into the working copy from khaiaw's `downloads/` (`rtmw-x_8xb704-270e_cocktail14-256x192.py`
  + `…-13a2546d_20231208.pth`) and `yolo12x.pt`. Those files still exist (checked 2026-08-21).
- **Reader:** `pose/pose_lib.py` → `load_pose(path)` → `{bboxes, confs, persons:[{kp[133,2], score[133]}]}`;
  handles the CUDA-tensor pickles on CPU.
- **Resumable:** skips existing pkls; empty-`pose_dict` pkl = "processed, nobody there", distinct
  from "not yet done". Rerunning the same command after a crash is the recovery.

## Why RTMW-x via mmpose (don't "optimize" this away)

We built a faster ONNX stack (rtmlib + distilled `rtmw-dw-x-l`, ~40 fps/GPU) and rejected it. The
false positives (skeletons on hands/dolls/wall photos) are **not** the detector's fault — YOLO12x and
YOLO11 fire the same boxes — they're the pose model's calibration: distilled `rtmw-dw-x-l` scores FP
bodies 1.3–1.8 (on a [0,3] scale, overlapping real people; unfilterable), non-distilled `rtmw-x`
scores them 0.1–0.3 (vanish at thr 0.3). Panels comparing the two on identical boxes were reviewed
and approved by Mike 2026-07-04. Diagnostics that established this: `pose/qa/pose_compare.py`,
`pose/qa/pose_infer.py`, and the throwaway `vlm-headcam/src/pose_{diag,fp_check,boxscore,yolo_test}.py`.
A future speedup that keeps calibration = export `rtmw-x` itself to ONNX (none published).

## What the 2025.2 run taught us (operational)

| Thing | Number / fact |
|---|---|
| input | 5,419,920 frames, 8,566 video dirs, ~330 GB of jpgs |
| throughput | ~35 fps/GPU → 8 A40 ≈ 5 h wall (dedicated GPUs) |
| output | ~15 GB of pkls; CSV 1.4 GB / 6.5M person-rows |
| coverage | ~66% of frames have ≥1 person (old 4M run: 69%) |
| failures | 1 corrupt frame; 0 Ray worker deaths after the try/except patch |
| Ray gotcha | Ray object spill fills `/tmp`; pin `TMPDIR=/data2/mcfrank/tmp` and `_temp_dir` on /data2 — and /data2 was 95% full at the time (raylet warned). Check `df` first. |
| CSV gotcha | returning 5.4M-row frames through Ray's object store spills to disk → `create_csv.py` writes per-worker part files and stream-concats (that's the version in this repo; the node's `create_csv_2025_2.py` ends with a harmless `NameError` after writing — fixed in `e0ce9e8`) |
| logs | the run log (`pose_full.log`) was **not** preserved; what remains on node: `/data2/mcfrank/tb.log` (throughput bench, 2026-07-04 09:34), `/data2/mcfrank/pose_csv.log` (CSV build), and dirs `pose_bench_out/ pose_calib_out/ pose_smoke_out/ pose_tb_out/ pose_2025_2/` (bench/calibration/smoke artifacts). **Keep the 2026.1 log** (`tee` it to `/data2/mcfrank/pose_2026_1.log` and copy to Oak after). |

QA used after the run: `pose/qa/pose_sanity.py` (coverage / persons-per-frame / score calibration
histograms) and `pose_compare.py --diverse` overlays (faces — review on-node or in a gitignored dir).

## 2026.1 plan

### 0. Before anything
```bash
# sync this repo to the node (it is 4 commits behind) — from the Mac:
rsync -av --exclude .git ~/Projects/bv-annotations/ ccn2-14:~/bv-annotations/
# rebuild the env (gone):  follow pose/SETUP.md step by step; verify `from mmcv.ops import nms` loads
# confirm weights symlinks still resolve:
ls -L /data2/mcfrank/babyview-pose/downloads/rtmw-x_8xb704-270e_cocktail14-256x192.py /data2/mcfrank/babyview-pose/yolo12x.pt
```

### 1. Wait for / confirm frames
2026.1 video ids follow the 2025.2 convention (`S00220001_2024-02-05_1_rec…`, some `_rotated`), so
the pipeline's path logic needs no change. Check:
```bash
FR=/ccn2a/dataset/babyview/2026.1/extracted_frames_1fps
ls $FR | wc -l; find $FR -maxdepth 2 -name '*.jpg' | head -3
```
Expect ≥2× 2025.2: ~11M frames, ~700 GB of jpgs. `/ccn2a` had 2.7 TB free on 2026-08-21 — enough,
but coordinate with Alvin: his plan is to cycle chunks through ccn and out to Oak, so **frames may
exist only transiently**. Run pose on each chunk while it is on disk; the pkls are small (~30 GB
total) and stay on `/ccn2`.

### 2. Run (per chunk or whole tree — same command, it's resumable)
```bash
mkdir -p /ccn2/dataset/babyview/2026.1/outputs/pose_1fps
cd /data2/mcfrank/babyview-pose
export TMPDIR=/data2/mcfrank/tmp
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /data2/mcfrank/mmpose_env/bin/python run_model.py \
   --input_dir  /ccn2a/dataset/babyview/2026.1/extracted_frames_1fps/ \
   --output_dir /ccn2/dataset/babyview/2026.1/outputs/pose_1fps \
   --num_processes 32  > /data2/mcfrank/pose_2026_1.log 2>&1" < /dev/null &
```
- `--num_processes = 4 × #GPUs`. The node is shared (GPUs 3–7 were busy on 2026-08-21); with 3 free
  GPUs use `CUDA_VISIBLE_DEVICES=0,1,2 --num_processes 12` and expect ~30 h for 11M frames, or ask
  to pause other jobs.
- Watch: `find /ccn2/dataset/babyview/2026.1/outputs/pose_1fps -name '*.pkl' | wc -l`,
  `grep -c 'POSE FAIL' /data2/mcfrank/pose_2026_1.log`.

### 3. After: CSV + QA + register
```bash
TMPDIR=/data2/mcfrank/tmp /data2/mcfrank/mmpose_env/bin/python ~/bv-annotations/pose/create_csv.py \
   /ccn2/dataset/babyview/2026.1/outputs/pose_1fps /ccn2/dataset/babyview/2026.1/outputs/pose_1fps_bbox_limbs.csv
/data2/mcfrank/mmpose_env/bin/python ~/bv-annotations/pose/qa/pose_sanity.py /ccn2/dataset/babyview/2026.1/outputs/pose_1fps
```
Sanity targets from 2025.2: pkl count == frame count; coverage ~60–70%; body-keypoint score
histogram bimodal with FP mass near 0. Then archive log + CSV to Oak
(`oak-dtn:/oak/stanford/groups/mcfrank/…`, see `vlm-headcam` train-on-ccn2 skill) and record the
release + this repo's SHA next to the outputs.

## Open items / decisions for Mike
- **Chunked vs whole-tree run** depends on Alvin's cycling schedule — ask him whether frames will
  persist on `/ccn2a` or get evicted to Oak per chunk.
- **Incremental ids:** if 2026.1 is a superset of 2025.2 (same video ids), the resumable skip lets us
  seed `2026.1/outputs/pose_1fps` by **hard-linking/copying the 2025.2 pkls** for unchanged videos and
  only computing the new ones — worth checking id overlap before burning ~50 GPU-hours.
- **Push `bv-annotations` to GitHub** (name pending since July) so the node and laptop stop drifting.

## UPDATE 2026-08-21 (afternoon): the run is live

Supersedes the "2026.1 is not runnable yet" TL;DR above. What changed:

- **Inputs:** khaiaw's complete GCS mirror (Jul 22–24) is at `/ccn2b/dataset/babyview/gcloud/pull/` (20,301 mp4s,
  11.5 TB, verified vs bucket + Airtable). No 1-fps frames existed for 2026.1, so **we extract them ourselves** with
  `frames/extract_frames_1fps.py` — khaiaw's 2025.2 recipe, verified pixel-identical on a re-extracted 2025.2 video.
- **Reuse:** 2026.1 ⊇ 2025.2 exactly (all 8,566 ids, durations match pkl counts ±1 s) → 2025.2 pkls are copied into
  the 2026.1 tree (`pose/copy_2025_2_pose.sh`); only the 11,735 new videos (≈6.4M frames) are extracted + posed.
- **Env:** rebuilt per SETUP.md with two extra pins learned the hard way — `setuptools<81` (mmengine imports
  `pkg_resources`) and `ultralytics==8.4.87` (the 2025.2 freeze; latest drifted) — and numpy 1.26.4 installed **last
  with `--no-deps`** (any later `pip install` drags numpy 2 back in → `xtcocotools` ABI crash). Verified: same
  #persons and 0.000 px keypoint diff vs the released 2025.2 pkls on 20 frames.
- **Where:** Cliona's `/ccn2b/dataset/babyview/2026.1/` is group-only, so everything is staged at
  `/ccn2b/dataset/babyview/_mcfrank_2026.1_staging/{extracted_frames_1fps,outputs/pose_1fps}` and `mv`'d in once the
  dir is opened. Run state/logs: `/data2/mcfrank/pose_2026_1/` (markers, waves, manifest, meta/crosswalk).
- **Chain (all detached on node14):** extraction (24 niced CPU procs, ~15 h) → `pose/pose_waves.sh` (8 GPUs, waves
  over newly extracted videos, per-video done markers, 3 retries) → `pose/finalize.sh` (frame dims, CSV, sanity).
  Watch: `tail /data2/mcfrank/pose_2026_1/{extract_2026_1,pose_waves,finalize}.log`.
- **Camera versions (Airtable `camera`):** bones = V2, mini = V3 (4K; ~1.5k videos in 4:3 mode → 512×683 frames),
  headlight = separate rig (landscape; exclude). 2025.2 has no mini video. Frame dims now vary per video — always
  normalize by the per-video (w, h) from `frames/frame_dims.py`.
