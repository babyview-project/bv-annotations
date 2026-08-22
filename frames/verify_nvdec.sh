#!/bin/bash
# 3,230 of the 2026.1 frame dirs were decoded on NVDEC rather than CPU. Byte-identity was verified on
# 2 videos before the run; this checks a stratified random sample end-to-end (every frame, not a subset)
# by re-extracting each with the canonical CPU recipe and comparing.
W=/data2/mcfrank/pose_2026_1
ST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging/extracted_frames_1fps
PULL=/ccn2b/dataset/babyview/gcloud/pull
T=/data2/mcfrank/tmp/nvdec_verify; rm -rf $T; mkdir -p $T
SC="scale='if(gte(iw,ih),-1,512):if(gte(iw,ih),512,-1)'"
# sample: 18 portrait 16:9 + 6 of the 4:3 mini mode, restricted to in-release videos we still have frames for
awk -F'\t' '$2=="ok" && $4==512 && $5==910 {print $1}' $W/frames_manifest_gpu.tsv | shuf --random-source=<(yes) | head -18  > $T/sample.txt
awk -F'\t' '$2=="ok" && $4==512 && $5==683 {print $1}' $W/frames_manifest_gpu.tsv | shuf --random-source=<(yes) | head -6 >> $T/sample.txt
echo "sampling $(wc -l < $T/sample.txt) NVDEC-extracted videos"
check() {
  v=$1; T=/data2/mcfrank/tmp/nvdec_verify
  src=$(grep -m1 "/$v\.mp4$" /data2/mcfrank/pose_2026_1/extract_list_2026_1.txt)
  [ -z "$src" ] && { echo "NOSRC $v"; return; }
  o=$T/$v; mkdir -p $o
  ffmpeg -y -nostdin -hide_banner -loglevel error -threads 4 -i "$src" \
     -vf "fps=1:round=near,scale='if(gte(iw,ih),-1,512):if(gte(iw,ih),512,-1)'" -qscale:v 1 -pix_fmt yuvj444p \
     -video_track_timescale 1000 -frame_pts 1 -vsync vfr $o/%05d.jpg 2>/dev/null
  # the video may be in the release staging tree or in the sidelined (non-2026.1) tree
  G=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging/extracted_frames_1fps/$v
  [ -d "$G" ] || G=/ccn2b/dataset/babyview/_mcfrank_not_in_2026.1/extracted_frames_1fps/$v
  ncpu=$(ls $o | wc -l); ngpu=$(ls $G 2>/dev/null | wc -l)
  diff=0; for f in $o/*.jpg; do cmp -s "$f" "$G/$(basename $f)" || diff=$((diff+1)); done
  [ "$ncpu" = "$ngpu" ] && [ "$diff" = "0" ] && echo "IDENTICAL $v n=$ncpu" || echo "DIFFER $v cpu=$ncpu gpu=$ngpu bytediff=$diff"
  rm -rf $o
}
export -f check
xargs -P 8 -I{} bash -c 'check {}' < $T/sample.txt
echo "NVDEC_VERIFY_DONE $(date)"
