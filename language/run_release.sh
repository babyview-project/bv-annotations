#!/bin/bash
# Both passes + collapse for one release. Resumable; safe to re-launch.
#   bash run_release.sh 2025.2 /ccn2a/dataset/babyview/2025.2/outputs/merged_transcripts_parsed.csv
set -u
REL=$1; T=$2; OUT=${3:-/ccn2/dataset/babyview/annotations/language}
cd "$(dirname "$0")"
set -a; . "$HOME/.secrets/vlm-headcam.env"; set +a
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.secrets/vlm-headcam-sa.json}"
P=/data2/mcfrank/ladder/condaenv/bin/python
mkdir -p "$OUT"
for PASS in 0 1; do
  echo "=== $REL pass $PASS ==="
  $P annotate_language.py --transcript "$T" --release "$REL" --out "$OUT" \
     --pass $PASS --workers 32 || { echo "PASS $PASS FAILED"; exit 1; }
done
$P agree_passes.py --release "$REL" --out "$OUT"
echo "LANG_${REL}_DONE"
