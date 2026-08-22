# frames — 1-fps frame extraction for a release

`extract_frames_1fps.py` reproduces khaiaw's 2025.2 recipe **exactly** (verified pixel-identical on a
re-extracted 2025.2 video, 2026-08-21): `ffmpeg -vf "fps=1:round=near,scale=<short side→512>" -qscale:v 1
-pix_fmt yuvj444p -video_track_timescale 1000 -frame_pts 1 -vsync vfr <video_id>/%05d.jpg` (index = second).
Original: `/ccn2/u/khaiaw/Code/babyview-pose/babyview-dataset-user/extract_frames_from_videos.py`.

Adds: explicit video list (so you can prioritize), per-video marker files for resumability, a manifest
(video, status, n_frames, w, h, secs), `-y -nostdin`, thread cap, `nice`, and three-instance parallelism over one
list (`--reverse`, `--shuffle SEED`; marker + "dir younger than 30 min" rules keep instances disjoint).
Throughput (ccn2, 4 threads/proc): 1080p ≈ 13-15× realtime, 4K ≈ 5-6× realtime per CPU process.

**NVDEC (`--hwaccel_gpus 0,1,…`)** — `-hwaccel cuda -hwaccel_output_format cuda` +
`fps=1:round=near,hwdownload,format=nv12,scale=…` — takes decode off the CPUs (4K ≈ 1.75× faster per stream;
~2 streams/GPU before `cuvidCreateDecoder` starts failing on the A40's session limit). **Use it ONLY when the
output height is even.**

> **Byte-identity depends on output parity** (measured 2026-08-21, per-frame over a stratified sample):
> even output (512×910) → **10/10 videos byte-identical**; **odd output (512×683, the mini 4:3 mode) → 5/5 videos
> DIFFER** (e.g. 225 of 525 frames), because `hwdownload,format=nv12` → swscale rounds chroma differently for an
> odd height. `frames/verify_nvdec.sh` is the check; `frames/redo_nvdec_odd.sh` re-extracts affected videos on CPU.
> The extractor does not yet gate on this automatically — pass `--hwaccel_gpus` only for known-even sources, or
> verify afterwards. Also NOT identical: cuvid's own `-resize`, and `format=yuv420p` before `scale`.

Frame sizes: portrait 16:9 video → 512×910; the mini camera's 4:3 mode (2880×3840) → 512×683;
headlight (landscape 1920×1080) → 910×512. `frame_dims.py` caches (video, w, h, n_frames) per dir —
needed to normalize coordinates downstream.

```bash
python extract_frames_1fps.py --list videos.txt --out_root <REL>/extracted_frames_1fps \
    --marker_dir /data2/mcfrank/<run>/markers --manifest <run>/frames_manifest.tsv --num_processes 24 --threads 4
```
