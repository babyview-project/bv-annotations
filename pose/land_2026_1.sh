#!/bin/bash
# Land the staged 2026.1 frames + pose into the release dir (same filesystem -> instant renames), then verify.
# Run ONLY after FINAL_ANALYSIS_DONE (no writers left). Usage: bash land_2026_1.sh [--go]
set -u
ST=/ccn2b/dataset/babyview/_mcfrank_2026.1_staging; REL=/ccn2b/dataset/babyview/2026.1; W=/data2/mcfrank/pose_2026_1
[ "${1:-}" = "--go" ] || { echo "dry run — add --go"; }
grep -q ANALYSES_DONE $W/chain.log 2>/dev/null || { echo "chain not finished (no ANALYSES_DONE in chain.log)"; exit 1; }
grep -q "blocked" $W/chain.log && { echo "chain finished BLOCKED - do not land"; exit 1; }
nbad=$(wc -l < $W/bad_pkls_runtime.txt 2>/dev/null || echo 0)
[ "$nbad" -eq 0 ] || { echo "REFUSING: $nbad unreadable pkls recorded during the CSV build"; exit 1; }
grep -q "pkl != jpg:          0" $W/verify_report.txt || { echo "REFUSING: verify_report shows pkl/jpg mismatches"; exit 1; }
pgrep -u mcfrank -f "extract_frames_1fp[s].py|run_mode[l].py|create_cs[v].py|copy_one" >/dev/null && { echo "writers still running"; exit 1; }
for p in $REL/extracted_frames_1fps $REL/outputs/pose_1fps $REL/outputs/pose_1fps_bbox_limbs.csv; do [ -e $p ] && { echo "target exists: $p"; exit 1; }; done
nf=$(ls $ST/extracted_frames_1fps | wc -l); np=$(ls $ST/outputs/pose_1fps | wc -l); echo "staged: $nf frame dirs, $np pose dirs, csv $(du -h $ST/outputs/pose_1fps_bbox_limbs.csv | cut -f1)"
# the GCS pull is a superset of the release: refuse to land anything that is not exactly the 2026.1 set
REL_N=$(wc -l < $W/meta/release_2026_1_ids.txt)
stray=$(comm -23 <(ls $ST/extracted_frames_1fps | sort) <(sort $W/meta/release_2026_1_ids.txt) | wc -l)
short=$(comm -13 <(ls $ST/extracted_frames_1fps | sort) <(sort $W/meta/release_2026_1_ids.txt) | wc -l)
echo "release list: $REL_N | staged-but-not-in-release: $stray | in-release-but-missing: $short (12 sub-second clips expected)"
[ "$stray" -eq 0 ] || { echo "REFUSING: $stray non-2026.1 dirs still staged (run sideline_excluded.sh)"; exit 1; }
[ "${1:-}" = "--go" ] || exit 0
mkdir -p $REL/outputs
mv $ST/extracted_frames_1fps $REL/extracted_frames_1fps && mv $ST/outputs/pose_1fps $REL/outputs/pose_1fps && mv $ST/outputs/pose_1fps_bbox_limbs.csv $REL/outputs/pose_1fps_bbox_limbs.csv || { echo "MV FAILED"; exit 1; }
echo "moved: $(ls $REL/extracted_frames_1fps | wc -l) frame dirs, $(ls $REL/outputs/pose_1fps | wc -l) pose dirs"
# leave a back-pointer so the run's own logs/markers (which reference the staging paths) stay interpretable
ln -s $REL/extracted_frames_1fps $ST/extracted_frames_1fps; ln -s $REL/outputs/pose_1fps $ST/outputs/pose_1fps
cp $W/frames_manifest*.tsv $W/meta/frame_dims_2026_1_new.tsv $W/meta/release_2026_1_ids.txt $W/meta/excluded_from_2026_1_ids.txt $W/meta/videos_airtable_2026-07-24.csv $REL/outputs/ 2>/dev/null
cat > $REL/outputs/README_pose_1fps.md <<EOT
# 2026.1 frames + whole-body pose — provenance ($(date +%F))

- extracted_frames_1fps/: 1 fps jpgs, short side 512 (portrait 16:9 -> 512x910, mini 4:3 mode -> 512x683, headlight landscape -> 910x512),
  same recipe as 2025.2 (bv-annotations frames/extract_frames_1fps.py; pixel-identical to khaiaw's extract_frames_from_videos.py).
  Source videos: /ccn2b/dataset/babyview/gcloud/pull (khaiaw's full GCS mirror, 2026-07-24). Frame index = second.
  MEMBERSHIP: exactly the videos tagged '2026.1' in videos_airtable_2026-07-24.csv (16,306 of the 20,301 in the pull).
  The other 3,995 (3,793 untagged/post-release uploads, 201 preschool, 1 absent from the export) were also processed
  but live in /ccn2b/dataset/babyview/_mcfrank_not_in_2026.1/ -- do not mix them in. Id lists in outputs/.
  8,566 videos are the 2025.2 set (identical files); 11,735 are new. 12 sub-second clips (<0.5 s; 11 x S00270001, 1 x S00560001)
  yield no 1-fps frame and have no dir here (listed as fail_rc0 in frames_manifest*.tsv). Per-video dims: frame_dims_2026_1_new.tsv + 2025.2 = 512x910 / 910x512.
- outputs/pose_1fps/: YOLO12x person boxes -> mmpose RTMW-x cocktail14 256x192 (133-kpt COCO-WholeBody), one pkl per frame
  (bv-annotations pose/run_model.py, env per pose/SETUP.md: torch 2.1.0+cu121, mmcv 2.1.0, mmpose 1.3.2, ultralytics 8.4.87, numpy 1.26.4).
  2025.2 videos' pkls were copied from /ccn2/dataset/babyview/2025.2/outputs/pose_1fps (verified same frames); new videos computed 2026-08-21/22.
- outputs/pose_1fps_bbox_limbs.csv: flat table, one row per detected person (bv-annotations pose/create_csv.py; ids strip '_processed').
- Reader: bv-annotations pose/pose_lib.py. Run logs: /data2/mcfrank/pose_2026_1/ (node14) and Oak archive.
EOT
echo LANDED
