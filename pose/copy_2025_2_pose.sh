#!/bin/bash
# Seed 2026.1 pose dir with the 2025.2 pkls (same videos, same frames -- verified: durations match to the second).
SRC=/ccn2/dataset/babyview/2025.2/outputs/pose_1fps
DST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging/outputs/pose_1fps
mkdir -p $DST
cd $SRC
ls | nice -n 10 xargs -P 8 -I{} cp -r {} $DST/ 2> /data2/mcfrank/pose_2026_1/copy_2025_2_pose.err
echo COPY_DONE $(date)
