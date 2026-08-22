#!/bin/bash
# After finalize (CSV) lands: goal-2 analyses over the full 2026.1 pose CSV + IMU, joined on the Jul-24 crosswalk.
W=/data2/mcfrank/pose_2026_1; ST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging; PY=/data2/mcfrank/mmpose_env/bin/python
until grep -q FINALIZE_DONE $W/finalize.log 2>/dev/null; do sleep 600; done
cd $W
$PY pose_hand_camera.py --csv $ST/outputs/pose_1fps_bbox_limbs.csv --crosswalk meta/crosswalk_airtable_2026-07-24.tsv \
   --dims meta/frame_dims_2025_2.tsv --dims meta/frame_dims_2026_1_new.tsv --out analysis_2026_1 --release_ids meta/release_2026_1_ids.txt \
   --exclude_cameras headlight --fov bones=110 --fov mini=130 --release_ids meta/release_2026_1_ids.txt > analysis_2026_1.log 2>&1
echo "pose analysis rc=$? $(date)" >> final_analysis.log
$PY imu_camera_pitch.py analyze --imu imu/imu_video_pitch.tsv --crosswalk meta/crosswalk_airtable_2026-07-24.tsv \
   --out imu/analysis_2026_1 --pose_video_summary analysis_2026_1/video_summary.csv --exclude_cameras headlight --fov bones=110 --fov mini=130 --release_ids meta/release_2026_1_ids.txt > imu/analyze_2026_1.log 2>&1
echo "imu analysis rc=$? $(date)" >> final_analysis.log
echo "FINAL_ANALYSIS_DONE $(date)" >> final_analysis.log
