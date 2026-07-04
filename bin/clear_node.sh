#!/usr/bin/env bash
# clear_node.sh — free this node by killing every GPU compute process (all GPUs).
# Needs sudo for other users' processes. Usage: ./clear_node.sh [--dry-run]
set -u
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# PIDs of all GPU compute apps across every GPU (from the same table gpustat shows)
mapfile -t PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -un)

if [ ${#PIDS[@]} -eq 0 ]; then
  echo "No GPU compute processes on $(hostname)."
  exit 0
fi

echo "GPU compute processes on $(hostname):"
printf "  %-8s %-12s %-10s %s\n" PID USER GPU_MEM CMD
for p in "${PIDS[@]}"; do
  [ -z "$p" ] && continue
  u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F', *' -v P="$p" '$1==P{print $2" MiB"; exit}')
  cmd=$(ps -o args= -p "$p" 2>/dev/null | cut -c1-60)
  printf "  %-8s %-12s %-10s %s\n" "$p" "${u:-?}" "${mem:-?}" "${cmd:-?}"
done

if [ "$DRY" -eq 1 ]; then
  echo "(dry-run) would run: sudo kill -9 ${PIDS[*]}"
  exit 0
fi

echo "Killing ${#PIDS[@]} process(es) with: sudo kill -9 ${PIDS[*]}"
sudo kill -9 "${PIDS[@]}"
sleep 2
echo "GPU state now:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
