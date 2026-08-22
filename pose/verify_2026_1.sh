#!/bin/bash
# QA gate before landing: repair any incomplete frame copies, then verify the staged 2026.1 tree
# (membership, frames per video, pkl-per-frame coverage). Writes a report; does not move anything.
W=/data2/mcfrank/pose_2026_1
ST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging
SRC=/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps
R=$W/verify_report.txt
while pgrep -u mcfrank -f "copy_2025_2_frame[s].sh" > /dev/null; do sleep 120; done
# repair pass: anything without a completion marker (races, partial copies) is redone
bash $W/copy_2025_2_frames.sh >> $W/frame_copy.log 2>&1
while pgrep -u mcfrank -f "run_mode[l].py" > /dev/null; do sleep 300; done
{
echo "=== 2026.1 staging verification $(date)"
echo "release list:            $(wc -l < $W/meta/release_2026_1_ids.txt)"
echo "frame dirs staged:       $(ls $ST/extracted_frames_1fps | wc -l)"
echo "pose dirs staged:        $(ls $ST/outputs/pose_1fps | wc -l)"
echo "stray (not in release):  $(comm -23 <(ls $ST/extracted_frames_1fps | sort) <(sort $W/meta/release_2026_1_ids.txt) | wc -l)"
echo "missing frames dirs:     $(comm -13 <(ls $ST/extracted_frames_1fps | sort) <(sort $W/meta/release_2026_1_ids.txt) | wc -l)"
echo "missing pose dirs:       $(comm -13 <(ls $ST/outputs/pose_1fps | sort) <(sort $W/meta/release_2026_1_ids.txt) | wc -l)"
echo "leftover .part dirs:     $(ls $ST/extracted_frames_1fps | grep -c '\.part$')"
echo "--- per-video frame vs pkl counts (all 16,306):"
} > $R
python3 - >> $R 2>&1 <<'PY'
import os
ST='/ccn2b/dataset/babyview/_mcfrank_2026.1_staging'
W='/data2/mcfrank/pose_2026_1'
ids=[l.strip() for l in open(f'{W}/meta/release_2026_1_ids.txt') if l.strip()]
bad_missing=[]; bad_short=[]; ok=0; tot_f=tot_p=0
for v in ids:
    fd=f'{ST}/extracted_frames_1fps/{v}'; pd=f'{ST}/outputs/pose_1fps/{v}'
    nf=len(os.listdir(fd)) if os.path.isdir(fd) else 0
    npk=len(os.listdir(pd)) if os.path.isdir(pd) else 0
    tot_f+=nf; tot_p+=npk
    if nf==0: bad_missing.append(v)
    elif npk!=nf: bad_short.append((v,nf,npk))
    else: ok+=1
print(f'complete (pkl==jpg): {ok}/{len(ids)}   total frames {tot_f:,}  total pkls {tot_p:,}')
print(f'no frames at all:    {len(bad_missing)}  (12 sub-second clips expected)')
for v in bad_missing[:15]: print('   NOFRAMES', v)
print(f'pkl != jpg:          {len(bad_short)}')
for v,nf,npk in bad_short[:25]: print(f'   SHORT {v} jpg={nf} pkl={npk}')
open(f'{W}/verify_incomplete_ids.txt','w').write('\n'.join(v for v,_,_ in bad_short)+'\n')
PY
echo "VERIFY_DONE $(date)" >> $R
cat $R
