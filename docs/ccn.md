# Running on CCN (ccn2)

## Node & GPUs

- Work on **`ccn2-14`** (`ssh ccn2-14`). 8× NVIDIA A40, 46 GB each. **Shared** — always
  `gpustat` / `nvidia-smi` before launching; pick idle GPUs with `CUDA_VISIBLE_DEVICES`.
- SSH auth is occasionally flaky (`Permission denied` on a good key) — just retry.
- `bin/clear_node.sh` frees **all** GPU compute processes on the node (needs sudo; kills other
  users' jobs too, so it's for when the node is yours). `clear_node.sh --dry-run` lists first.
  Note others may relaunch and re-grab GPUs right after.

## Filesystem map

| Path | What | Notes |
|------|------|-------|
| `/ccn2a/dataset/babyview/<RELEASE>/extracted_frames_1fps/` | input frames (1 fps) | shared, read-only to us |
| `/ccn2/dataset/babyview/<RELEASE>/outputs/<annotation>/` | annotation outputs | shared; ~99% full — check space |
| `/ccn2/u/khaiaw/Code/babyview-pose/` | original pose repo + model weights (`downloads/`, `yolo12x.pt`) | read-only; we symlink the weights |
| `/data2/mcfrank/` | node-local **NVMe** scratch: Python envs, working copies, logs | fast; also ~99% full — mind `TMPDIR` |
| `~/.secrets/` | Vertex service account + env file | chmod 600, never in git |

**Disk is tight** on both `/ccn2/dataset/babyview` and `/data2` (~99%). Point `pip`/build tmp at
`/data2/mcfrank/tmp` (not `/tmp`, which is small). Annotation outputs are small (pose over 2025.2
≈ a few GB); envs (torch/mmcv) are the big consumers.

## Environments (node-local, reproducible from specs)

| Env | For | Spec |
|-----|-----|------|
| `/data2/mcfrank/mmpose_env` | pose (mmpose RTMW-x) | `pose/requirements.txt` + `pose/SETUP.md` |
| `/data2/mcfrank/pose_env` | rtmlib pose QA + Gemini alignment | rtmlib + onnxruntime-gpu 1.20.1 + google-genai |

These live on node-local NVMe (so only on ccn2-14). They're disposable — rebuild from the specs.
onnxruntime-gpu needs cuDNN: `pose_env` borrows it from a torch install via
`LD_LIBRARY_PATH=$(ls -d …/site-packages/nvidia/*/lib | tr '\n' ':')`.

## Long runs

`setsid` to survive your SSH session; log to a file; poll with `find <out> -name '*.pkl' | wc -l`.
All pipelines here are **resumable** (skip already-written outputs), so preemption/crash → rerun
the same command. Ray-based runs put their temp under `/data2/mcfrank/ray_tmp`.

## Human subjects

Frames contain faces. Overlays/panels for visual QA are generated on the node and reviewed there
or pulled to a **gitignored** local dir (`vlm-headcam/skeletons/`). Only keypoints, scores, and
aggregate numbers leave the node; never frame imagery, and never to a public URL.
