#!/bin/bash
# After pose waves + 2025.2 copy finish: flat CSV over the whole 2026.1 pose tree, sanity QA, frame dims for new videos.
W=/data2/mcfrank/pose_2026_1; ST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging
PY=/data2/mcfrank/mmpose_env/bin/python
until grep -q POSE_ALL_DONE $W/pose_waves.log 2>/dev/null; do sleep 600; done
until grep -q COPY_DONE $W/copy_2025_2_pose.log 2>/dev/null; do sleep 300; done
echo "START finalize $(date)" >> $W/finalize.log
nice -n 5 /data2/mcfrank/ladder/condaenv/bin/python $W/frame_dims.py $ST/extracted_frames_1fps $W/meta/frame_dims_2026_1_new.tsv > $W/frame_dims_new.log 2>&1
echo "dims done $(date)" >> $W/finalize.log
cd $W && TMPDIR=/data2/mcfrank/tmp $PY ~/bv-annotations/pose/create_csv.py $ST/outputs/pose_1fps $ST/outputs/pose_1fps_bbox_limbs.csv > $W/create_csv.log 2>&1
echo "csv done rc=$? $(date)" >> $W/finalize.log
$PY ~/bv-annotations/pose/qa/pose_sanity.py $ST/outputs/pose_1fps > $W/pose_sanity.log 2>&1
echo "FINALIZE_DONE $(date)" >> $W/finalize.log
