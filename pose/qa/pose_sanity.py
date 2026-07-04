"""Post-run sanity check on a pose output dir: coverage, persons/frame, score calibration.
Samples pkls across videos (reading all N million is slow and unnecessary).

  python pose_sanity.py /ccn2/dataset/babyview/<RELEASE>/outputs/pose_1fps
"""
import os, sys, random, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # find pose_lib
import pose_lib

OUT = sys.argv[1] if len(sys.argv) > 1 else "/ccn2/dataset/babyview/2025.2/outputs/pose_1fps"
N_VID, N_PK = 250, 40
rng = random.Random(0)

vids = [v for v in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, v))]
rng.shuffle(vids)
nf = cov = fails = 0
npersons, body_meds = [], []
for v in vids[:N_VID]:
    d = os.path.join(OUT, v)
    pks = [p for p in os.listdir(d) if p.endswith(".pkl")]
    rng.shuffle(pks)
    for pk in pks[:N_PK]:
        try:
            o = pose_lib.load_pose(os.path.join(d, pk))
        except Exception:
            fails += 1
            continue
        nf += 1
        k = len(o["persons"])
        npersons.append(k)
        if k:
            cov += 1
            for P in o["persons"]:
                body_meds.append(float(np.median(P["score"][:17])))

print(f"sampled {nf} frames from {min(len(vids), N_VID)} videos (of {len(vids)} total)")
print(f"coverage (>=1 person): {cov / max(nf,1) * 100:.1f}%   [expect ~40-55% for BabyView]")
print(f"persons/frame: mean {np.mean(npersons):.2f}  max {max(npersons) if npersons else 0}")
print("persons/frame hist:", dict(sorted(collections.Counter(npersons).items())))
if body_meds:
    q = np.percentile(body_meds, [10, 50, 90])
    print(f"per-detection body-score median (calibrated [0,1]): 10/50/90 = "
          f"{q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}   [real people should sit well above 0.3]")
print(f"pkl load errors: {fails}")
