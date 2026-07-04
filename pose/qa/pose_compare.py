"""Old (2025.03 mmpose) vs new (RTMW/rtmlib) pose on identical frames, drawn with ONE
clean drawer (limbs as lines; face + hands as dots — no dense face-contour scribble).
Side-by-side panels for human review. Runs on the old sampled_frames so the pixels match.
"""
import argparse, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
import pose_lib
from pose_infer import make_model, infer

BODY_EDGES = pose_lib.BODY_EDGES
FACE, LHAND, RHAND = slice(23, 91), slice(91, 112), slice(112, 133)


def draw(img, persons, thr):
    """persons: list of (kp[133,2], score[133]). Body limbs as lines; joints/face/hands as dots."""
    for kp, sc in persons:
        for a, b in BODY_EDGES:
            if sc[a] > thr and sc[b] > thr:
                cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0, 255, 0), 2)
        for i in range(17):                       # body joints
            if sc[i] > thr:
                cv2.circle(img, tuple(kp[i].astype(int)), 3, (0, 255, 255), -1)
        for i in range(23, 91):                   # face
            if sc[i] > thr:
                cv2.circle(img, tuple(kp[i].astype(int)), 1, (255, 200, 0), -1)
        for i in range(91, 133):                  # hands
            if sc[i] > thr:
                cv2.circle(img, tuple(kp[i].astype(int)), 2, (255, 0, 255), -1)
    return img


def label(img, txt):
    cv2.rectangle(img, (0, 0), (len(txt) * 12 + 12, 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def old_persons_of(gcp, sec, thr=0.3):
    try:
        op = pose_lib.load_pose(str(Path(pose_lib.POSE_ROOT) / gcp / f"{sec:05d}.pkl"))
    except Exception:
        return None
    ppl = [(P["kp"], P["score"]) for P in op["persons"]]
    has = any((P[1][:17] > thr).sum() >= 8 for P in ppl)   # a real person: >=8 confident body kpts
    return ppl, has


def panel_for(model, gcp, sec, old_ppl, old_thr, new_thr, out):
    img = cv2.imread(str(Path(pose_lib.FRAME_ROOT) / gcp / f"{sec:05d}.jpg"))
    if img is None:
        return False
    old_img = label(draw(img.copy(), old_ppl, old_thr), f"OLD mmpose  n={len(old_ppl)}")
    kps, scs = infer(model, img)
    new_ppl = list(zip(np.asarray(kps, np.float32), np.asarray(scs, np.float32)))
    new_img = label(draw(img.copy(), new_ppl, new_thr), f"NEW RTMW  n={len(new_ppl)}")
    cv2.imwrite(str(Path(out) / f"{gcp}_{sec:05d}.jpg"), np.hstack([old_img, new_img]))
    return len(new_ppl)


def diverse_videos(seed=0):
    """gcp videos interleaved across children so an N-prefix samples all children."""
    import os, random
    from collections import defaultdict
    rng = random.Random(seed)
    bychild = defaultdict(list)
    for g in os.listdir(pose_lib.POSE_ROOT):
        bychild[g.split("_")[0]].append(g)
    for c in bychild:
        rng.shuffle(bychild[c])
    children = sorted(bychild)
    order, i = [], 0
    while any(i < len(bychild[c]) for c in children):
        for c in children:
            if i < len(bychild[c]):
                order.append(bychild[c][i])
        i += 1
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diverse", action="store_true", help="sample across children/videos")
    ap.add_argument("--gcp", default="00220001_2024-02-05_1_57f12f7764_processed")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--want-person", type=int, default=80)
    ap.add_argument("--want-empty", type=int, default=20)
    ap.add_argument("--old-thr", type=float, default=0.3)
    ap.add_argument("--new-thr", type=float, default=1.0)
    ap.add_argument("--out", default="/data2/mcfrank/vlm-headcam/pose_compare")
    args = ap.parse_args()

    model = make_model("performance")
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    import random

    if not args.diverse:
        pose_dir = Path(pose_lib.POSE_ROOT) / args.gcp
        secs = sorted(int(p.stem) for p in pose_dir.glob("*.pkl"))
        for sec in secs[:: max(1, len(secs) // args.n)][: args.n]:
            r = old_persons_of(args.gcp, sec)
            panel_for(model, args.gcp, sec, r[0] if r else [], args.old_thr, args.new_thr, od)
        print(f"wrote panels -> {od}"); return

    # diverse: one frame per video, across children; mix person-present + old-empty
    np_person = np_empty = 0
    person_hit = empty_hit = 0            # frames where NEW fired >=1 detection
    rng = random.Random(1)
    for gcp in diverse_videos():
        if np_person >= args.want_person and np_empty >= args.want_empty:
            break
        pose_dir = Path(pose_lib.POSE_ROOT) / gcp
        secs = [int(p.stem) for p in pose_dir.glob("*.pkl")]
        if not secs:
            continue
        rng.shuffle(secs)
        chosen = None
        for sec in secs[:6]:                       # a few tries per video
            r = old_persons_of(gcp, sec)
            if r is None:
                continue
            ppl, has = r
            if has and np_person < args.want_person:
                chosen = (sec, ppl, True); np_person += 1; break
            if not has and np_empty < args.want_empty:
                chosen = (sec, ppl, False); np_empty += 1; break
        if chosen:
            newn = panel_for(model, gcp, chosen[0], chosen[1], args.old_thr, args.new_thr, od)
            if newn >= 0:
                if chosen[2]:
                    person_hit += newn > 0
                else:
                    empty_hit += newn > 0
    print(f"wrote {np_person} person + {np_empty} empty panels -> {od}")
    print(f"NEW recall on old-person frames: {person_hit}/{np_person} = {person_hit/max(np_person,1)*100:.0f}%")
    print(f"NEW fires on old-empty frames:   {empty_hit}/{np_empty} = {empty_hit/max(np_empty,1)*100:.0f}%  (FP proxy)")


if __name__ == "__main__":
    main()
