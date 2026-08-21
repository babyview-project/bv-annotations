# frames — 1-fps frame extraction for a release

`extract_frames_1fps.py` reproduces khaiaw's 2025.2 recipe **exactly** (verified pixel-identical on a
re-extracted 2025.2 video, 2026-08-21): `ffmpeg -vf "fps=1:round=near,scale=<short side→512>" -qscale:v 1
-pix_fmt yuvj444p -video_track_timescale 1000 -frame_pts 1 -vsync vfr <video_id>/%05d.jpg` (index = second).
Original: `/ccn2/u/khaiaw/Code/babyview-pose/babyview-dataset-user/extract_frames_from_videos.py`.

Adds: explicit video list (so you can prioritize), per-video marker files for resumability, a manifest
(video, status, n_frames, w, h, secs), `-y -nostdin`, thread cap, `nice`. CPU-only: NVDEC (`h264_cuvid`)
is available on ccn2 but its resize path is not bit-identical to swscale, so we keep CPU decode.
Throughput (ccn2, 4 threads/proc): 1080p ≈ 13-15× realtime, 4K ≈ 5-6× realtime per process.

Frame sizes: portrait 16:9 video → 512×910; the mini camera's 4:3 mode (2880×3840) → 512×683;
headlight (landscape 1920×1080) → 910×512. `frame_dims.py` caches (video, w, h, n_frames) per dir —
needed to normalize coordinates downstream.

```bash
python extract_frames_1fps.py --list videos.txt --out_root <REL>/extracted_frames_1fps \
    --marker_dir /data2/mcfrank/<run>/markers --manifest <run>/frames_manifest.tsv --num_processes 24 --threads 4
```
