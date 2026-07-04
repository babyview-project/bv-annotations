"""Whole-body (133-kpt COCO-WholeBody) pose over 2025.2 frames via rtmlib (RTMDet + RTMW,
ONNX/CUDA) — no mmcv/mmdet needed. Processes frames in their native 2025.2 naming, so no
gcp-hash crosswalk. Per video, writes one compact .npz (ragged persons flattened).

Needs the CUDA-12 onnxruntime to find cuDNN 9 — run with:
  NV=/data2/mcfrank/ladder/condaenv/lib/python3.11/site-packages/nvidia
  export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=<gpu> pose_env/bin/python src/pose_infer.py ...

Modes:
  --benchmark VIDEO_ID   time throughput on one video + save skeleton overlays for review
  --shard FILE           process the video_ids listed in FILE (one per line), write .npz each
"""
import argparse
import time
from pathlib import Path
import numpy as np
import cv2

FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")


def make_model(mode="performance", detector="yolo", yolo_conf=0.7, yolo_model="yolo11m.pt"):
    """rtmlib RTMW for pose. Detector is swappable: the bundled YOLOX-humanart has its
    threshold baked into the ONNX (untunable, over-fires on person-like home clutter);
    a stock YOLO11 person detector exposes a tunable confidence, so a high threshold cuts
    the moderate-confidence false-positive boxes RTMW would otherwise fill with skeletons."""
    from rtmlib import Wholebody
    m = Wholebody(mode=mode, backend="onnxruntime", device="cuda")
    if detector == "yolo":
        from ultralytics import YOLO
        y = YOLO(yolo_model)
        def _det(img):
            r = y(img, classes=[0], conf=yolo_conf, verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy()
        m._det = _det
    else:
        m._det = m.det_model
    return m


def infer(m, img):
    """Detect persons, then pose only on real boxes. Bypasses rtmlib's whole-frame
    fallback (which fits a skeleton to every empty frame). Zeroes the confidence of
    keypoints predicted OUTSIDE the image — RTMW extrapolates occluded joints off-screen
    (e.g. a full body below a pair of on-screen hands), and a point outside the frame is
    not a real observation. Returns ([P,133,2],[P,133])."""
    bboxes = m._det(img)
    if len(bboxes) == 0:
        return np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
    kps, scs = m.pose_model(img, bboxes=list(bboxes))
    kps = np.asarray(kps, np.float32); scs = np.asarray(scs, np.float32).copy()
    H, W = img.shape[:2]
    oob = (kps[..., 0] < 0) | (kps[..., 0] >= W) | (kps[..., 1] < 0) | (kps[..., 1] >= H)
    scs[oob] = 0.0
    return kps, scs


def anatomy_ok(kp):
    """Frac of person-rows with plausible layout (nose above shoulder midline)."""
    if len(kp) == 0:
        return 0.0
    import numpy as np
    return float(np.mean(kp[:, 0, 1] < kp[:, [5, 6], 1].mean(1)))


def frames_of(video_id):
    d = FRAMES / video_id
    return sorted(d.glob("*.jpg"))


def run_video(model, video_id, limit=0):
    """Return per-person flattened arrays + frame bookkeeping for one video."""
    fps_paths = frames_of(video_id)
    if limit:
        fps_paths = fps_paths[:limit]
    fidx, npers = [], []
    kp_all, sc_all, box_all, pf_all = [], [], [], []
    for p in fps_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        fi = int(p.stem)
        kps, scs = infer(model, img)                       # [P,133,2], [P,133]
        fidx.append(fi); npers.append(len(kps))
        for j in range(len(kps)):
            kp_all.append(kps[j]); sc_all.append(scs[j]); pf_all.append(fi)
    out = dict(
        frame_idx=np.asarray(fidx, np.int32),
        n_person=np.asarray(npers, np.int16),
        kp=(np.asarray(kp_all, np.float16) if kp_all else np.zeros((0, 133, 2), np.float16)),
        sc=(np.asarray(sc_all, np.float16) if sc_all else np.zeros((0, 133), np.float16)),
        person_frame=np.asarray(pf_all, np.int32),
    )
    return out, len(fidx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="performance", choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--benchmark", metavar="VIDEO_ID")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--shard")
    ap.add_argument("--outdir", default="/data2/mcfrank/pose_2025_2")
    ap.add_argument("--overlays", default="/data2/mcfrank/vlm-headcam/pose_bench_overlays")
    ap.add_argument("--det-thr", type=float, default=0.8, help="YOLOX person-detection score threshold")
    ap.add_argument("--kpt-thr", type=float, default=0.6, help="keypoint draw threshold (RTMW scale ~[0,3])")
    args = ap.parse_args()

    model = make_model(args.mode, det_thr=args.det_thr)

    if args.benchmark:
        vid = args.benchmark
        # warmup
        w = frames_of(vid)[:5]
        for p in w:
            im = cv2.imread(str(p))
            if im is not None:
                model(im)
        t0 = time.time()
        out, nf = run_video(model, vid, limit=args.limit)
        dt = time.time() - t0
        cov = float(np.mean(out["n_person"] > 0)) if nf else 0.0
        ppf = float(out["n_person"].mean()) if nf else 0.0
        print(f"BENCHMARK {vid}  (det_thr={args.det_thr}, kpt_thr={args.kpt_thr})")
        print(f"  frames={nf}  time={dt:.1f}s  throughput={nf/dt:.1f} fps")
        print(f"  coverage(>=1 person)={cov*100:.1f}%  mean persons/frame={ppf:.2f}  max={out['n_person'].max()}")
        print(f"  anatomy-sane (nose>shoulders): {anatomy_ok(out['kp'].astype('float32'))*100:.1f}%")
        # overlays for human review (faces present — reviewer looks, not us)
        from rtmlib import draw_skeleton
        od = Path(args.overlays); od.mkdir(parents=True, exist_ok=True)
        paths = frames_of(vid)
        # spread samples across the whole video (not the first N detections)
        stride = max(1, len(paths) // 200)
        saved = 0
        for k in range(0, len(paths), stride):
            if saved >= 12:
                break
            im = cv2.imread(str(paths[k]))
            if im is None:
                continue
            kps, scs = infer(model, im)
            if len(kps):
                vis = draw_skeleton(im.copy(), kps, scs, kpt_thr=args.kpt_thr)
                cv2.imwrite(str(od / f"{vid}_{paths[k].stem}.jpg"), vis)
                saved += 1
        print(f"  wrote {saved} overlays (spread across video) -> {od}")
        return

    if args.shard:
        vids = [l.strip() for l in open(args.shard) if l.strip()]
        outd = Path(args.outdir); outd.mkdir(parents=True, exist_ok=True)
        t0, done = time.time(), 0
        for vid in vids:
            fp = outd / f"{vid}.npz"
            if fp.exists():
                continue
            out, nf = run_video(model, vid)
            np.savez_compressed(fp, **out)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(vids)} videos  {(time.time()-t0)/done:.1f}s/vid", flush=True)
        print(f"DONE shard: {done} videos in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
