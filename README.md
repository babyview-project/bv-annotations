# bv-annotations

Reproducible **annotation pipelines for BabyView releases** — the code and CCN-specific
runbooks that turn a raw release (extracted frames + transcripts) into derived annotations
(pose, referential alignment, …). Internally facing: heavy on "how to actually run this on
ccn2," light on science.

This repo *generates* annotations. The science that *consumes* them (contrastive training,
evals, the write-up) lives in [`vlm-headcam`](https://github.com/mcfrank/vlm-headcam).

## Annotations

| Annotation | What it produces | Status | Docs |
|------------|------------------|--------|------|
| **pose** | 133-kpt COCO-WholeBody pose (body+face+hands) per frame, per detected person | ✅ production (2025.2 done/running) | [pose/README.md](pose/README.md) |
| **alignment** | Per-(utterance, frame) referential-alignment score + candidate referent noun (Gemini/Vertex) | 🚧 in progress | [alignment/README.md](alignment/README.md) |

## Release conventions

A "release" is a dated snapshot of processed BabyView video (e.g. `2025.2`, later `2026.x`).
Everything is keyed off the release name so re-running on new data is a path swap:

- **Frames (input):** `/ccn2a/dataset/babyview/<RELEASE>/extracted_frames_1fps/<video_id>/<frame:05d>.jpg`
  (1 fps). 2025.2 = 5,419,920 frames across 8,566 videos, ~83 children.
- **Annotations (output):** `/ccn2/dataset/babyview/<RELEASE>/outputs/<annotation>/…`, mirroring
  the `<video_id>/<frame>` layout so no filename crosswalk is ever needed.
- `<video_id>` encodes child + date + session, e.g. `S00220001_2024-02-05_1_rec…`.

To run everything on a new release, see [docs/adding-a-release.md](docs/adding-a-release.md).

## Running on CCN

Compute norms, the filesystem map, environments, and GPU etiquette (including `clear_node.sh`)
are in [docs/ccn.md](docs/ccn.md). Short version:

- Node: `ccn2-14`, 8× NVIDIA A40 (46 GB), **shared** — check `gpustat`/`nvidia-smi` first.
- Per-annotation Python envs live on `/data2/mcfrank/` (node-local NVMe). They are
  reproducible from each annotation's `requirements.txt` + `SETUP.md`.
- Secrets (Vertex service account) live in `~/.secrets/` — never in this repo.

## Human-subjects note

BabyView frames contain faces. Keypoints/scores and aggregate numbers are shareable; **raw
frames and skeleton-overlay images are not** — generate overlays on the node, review them
there or pull to a gitignored local dir, and never commit frame imagery.
