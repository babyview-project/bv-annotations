#!/bin/bash
# NVDEC decode is byte-identical to the CPU recipe ONLY for even output dimensions. With an odd target
# height (the mini 4:3 mode -> 512x683, plus one 512x911) the hwdownload/nv12 -> swscale chroma path
# rounds differently, so those frames are NOT reproducible from the documented CPU recipe.
# Verified 2026-08-21: 10/10 even-height samples byte-identical, 5/5 odd-height samples differ.
# Fix: re-extract exactly those videos on CPU and let the wave driver re-pose them.
W=/data2/mcfrank/pose_2026_1
ST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging
L=$W/meta/nvdec_redo_ids.txt
# never touch frame dirs while a wave is building a list or running (one missing jpg kills the Ray job)
while pgrep -u mcfrank -f "run_mode[l].py" > /dev/null; do sleep 120; done
pkill -u mcfrank -f "[p]ose_waves.sh"; sleep 3
n=0
while read v; do
  rm -rf $ST/extracted_frames_1fps/$v $ST/outputs/pose_1fps/$v
  rm -f $W/markers/$v $W/pose_done/$v $W/pose_tries/$v
  grep -m1 "/$v\.mp4$" $W/extract_list_2026_1.txt >> $W/redo_videos.txt && n=$((n+1))
done < $L
echo "REDO: cleared $n videos ($(wc -l < $W/redo_videos.txt) source paths)"
/data2/mcfrank/ladder/condaenv/bin/python $W/extract_frames_1fps_gpu.py \
   --list $W/redo_videos.txt --out_root $ST/extracted_frames_1fps --marker_dir $W/markers \
   --manifest $W/frames_manifest_redo.tsv --num_processes 24 --threads 4 --nice 5 > $W/extract_redo.log 2>&1
echo "REDO_EXTRACT_DONE $(date): $(awk -F'\t' '$2=="ok"' $W/frames_manifest_redo.tsv | wc -l) ok, $(awk -F'\t' '$2!="ok"' $W/frames_manifest_redo.tsv | wc -l) failed"
# confirm the redone frames now match the canonical recipe (they were just made by it) and restart pose
setsid nohup bash $W/pose_waves.sh >> $W/pose_waves.log 2>&1 < /dev/null &
echo "REDO_DONE $(date); pose driver restarted"
