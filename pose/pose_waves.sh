#!/bin/bash
# Run the (unchanged) 2025.2 pose pipeline over newly extracted 2026.1 videos in waves, until
# extraction reports ALL_DONE and nothing is pending. Per video: pose-done marker iff #pkl == #jpg.
W=/data2/mcfrank/pose_2026_1
FR=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging/extracted_frames_1fps
OUT=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging/outputs/pose_1fps
PY=/data2/mcfrank/mmpose_env/bin/python
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NPROC=${NPROC:-32}; MINBATCH=${MINBATCH:-300}; MAXTRIES=3
export TMPDIR=/data2/mcfrank/tmp
mkdir -p $W/pose_done $W/waves $W/pose_tries $OUT
wave=0
while true; do
  comm -23 <(ls $W/markers | sort) <(ls $W/pose_done | sort) > $W/waves/pending.txt
  n=$(wc -l < $W/waves/pending.txt)
  xdone=$(grep -c ALL_DONE $W/extract_2026_1.log)
  if [ "$n" -eq 0 ]; then
    if [ "$xdone" -ge 1 ]; then echo "POSE_ALL_DONE $(date)"; break; fi
    sleep 600; continue
  fi
  if [ "$n" -lt "$MINBATCH" ] && [ "$xdone" -eq 0 ]; then sleep 600; continue; fi
  wave=$((wave+1)); L=$W/waves/wave_${wave}.txt; : > $L
  cp $W/waves/pending.txt $W/waves/wave_${wave}_videos.txt
  while read v; do ls $FR/$v/*.jpg >> $L; done < $W/waves/wave_${wave}_videos.txt
  echo "WAVE $wave start: $n videos, $(wc -l < $L) frames, $(date)"
  (cd /data2/mcfrank/babyview-pose && CUDA_VISIBLE_DEVICES=$GPUS $PY run_model.py \
      --input_frames_txt_file $L --output_dir $OUT --num_processes $NPROC > $W/waves/wave_${wave}.log 2>&1)
  while read v; do
    nj=$(ls $FR/$v | wc -l); np=$(ls $OUT/$v 2>/dev/null | wc -l)
    t=$(( $(cat $W/pose_tries/$v 2>/dev/null || echo 0) + 1 )); echo $t > $W/pose_tries/$v
    if [ "$nj" -eq "$np" ]; then echo "$np" > $W/pose_done/$v
    elif [ "$t" -ge "$MAXTRIES" ]; then echo "INCOMPLETE jpg=$nj pkl=$np" > $W/pose_done/$v; echo "GAVE UP $v jpg=$nj pkl=$np"
    else echo "INCOMPLETE (retry) $v jpg=$nj pkl=$np"; fi
  done < $W/waves/wave_${wave}_videos.txt
  echo "WAVE $wave done $(date); POSE FAIL lines: $(grep -c 'POSE FAIL' $W/waves/wave_${wave}.log); tracebacks: $(grep -c Traceback $W/waves/wave_${wave}.log)"
done
