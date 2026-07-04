# Running all annotations on a new release

When a new BabyView release lands (e.g. `2026.x`), regenerating every annotation is a path swap
plus a couple of sanity checks. Nothing here is release-specific except the `<RELEASE>` string.

## 0. Confirm inputs exist

```bash
RELEASE=2026.1
FR=/ccn2a/dataset/babyview/$RELEASE/extracted_frames_1fps
ls $FR | head; ls $FR | wc -l                      # videos
find $FR -maxdepth 2 -name '*.jpg' | head           # frame naming = <video_id>/<frame:05d>.jpg
```

If the naming scheme differs from `<video_id>/<NNNNN>.jpg`, check each pipeline's input glob /
output-path logic (pose keys output off the last two path components).

## 1. Pose

```bash
mkdir -p /ccn2/dataset/babyview/$RELEASE/outputs/pose_1fps
cd /data2/mcfrank/babyview-pose
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /data2/mcfrank/mmpose_env/bin/python run_model.py \
    --input_dir  /ccn2a/dataset/babyview/$RELEASE/extracted_frames_1fps/ \
    --output_dir /ccn2/dataset/babyview/$RELEASE/outputs/pose_1fps \
    --num_processes 32
```
Rebuild `mmpose_env` first if it's gone (`pose/SETUP.md`). ~35 fps/GPU. See `pose/README.md`.
Sanity: coverage ~40–55%; spot-check a few overlays with `pose/qa/pose_compare.py`.

## 2. Alignment (when the pairs manifest exists for the release)

Needs a `(video_id, frame_idx, text)` pairs manifest (built from the release's transcripts +
frame sampling — that step lives with the science, currently `vlm-headcam`). Then
`alignment/README.md`. Vertex auth per `alignment/SETUP.md`.

## 3. Register the outputs

Add the new `<RELEASE>/outputs/<annotation>/` path to whatever inventory the lab keeps, and note
the release + date + git SHA of this repo used, so a given annotation set is traceable to the
exact pipeline that produced it.

## Checklist

- [ ] inputs present, naming matches
- [ ] env rebuilt / verified (`mmcv._ext` loads)
- [ ] idle GPUs (`gpustat`), enough output disk (`df -h`)
- [ ] pose run complete (`find … | wc -l` == frame count), coverage sane, overlays spot-checked
- [ ] alignment run (if manifest ready)
- [ ] outputs registered with release + repo SHA
