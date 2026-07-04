"""Validation plots for the Gemini alignment scores: rating histograms per model and
correlation with CLIP's whole-frame score. Aggregate/derived numbers only (no frames)."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data2/mcfrank/vlm-headcam/book_figs/gemini_val.png"
FLASH, LITE = sys.argv[1], sys.argv[2]

d1 = pd.read_parquet(FLASH); d2 = pd.read_parquet(LITE)
d1 = d1[d1.alignment.notna()]; d2 = d2[d2.alignment.notna()]
m = d1.merge(d2, on=["video_id", "frame_idx"], suffixes=("_f", "_l"))
print(f"flash n={len(d1)}  lite n={len(d2)}  paired n={len(m)}")


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return pearsonr(x, y)[0], spearmanr(x, y)[0]

for name, d in [("FLASH", d1), ("LITE", d2)]:
    pr, sr = corr(d.alignment, d.clip_score_max)
    print(f"{name}: mean={d.alignment.mean():.1f} sd={d.alignment.std():.1f} "
          f"frac0={np.mean(d.alignment==0):.2f} frac100={np.mean(d.alignment==100):.2f} "
          f"| vs CLIP  pearson={pr:.3f} spearman={sr:.3f}")
pr, sr = corr(m.alignment_f, m.alignment_l)
print(f"FLASH vs LITE: pearson={pr:.3f} spearman={sr:.3f}")

fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), dpi=130)
bins = np.linspace(0, 100, 21)
ax[0, 0].hist(d1.alignment, bins=bins, color="#185fa5", alpha=0.6, label="Flash")
ax[0, 0].hist(d2.alignment, bins=bins, color="#b0655a", alpha=0.6, label="Flash-Lite")
ax[0, 0].set_title("Rating histograms (0–100)"); ax[0, 0].set_xlabel("Gemini alignment")
ax[0, 0].set_ylabel("pairs"); ax[0, 0].legend()

pr, sr = corr(m.alignment_f, m.alignment_l)
ax[0, 1].scatter(m.alignment_f, m.alignment_l, s=10, alpha=0.4, color="#2c2c2a")
ax[0, 1].set_title(f"Flash vs Flash-Lite  (r={pr:.2f}, ρ={sr:.2f})")
ax[0, 1].set_xlabel("Flash"); ax[0, 1].set_ylabel("Flash-Lite")

for a, d, name, col in [(ax[1, 0], d1, "Flash", "#185fa5"), (ax[1, 1], d2, "Flash-Lite", "#b0655a")]:
    pr, sr = corr(d.alignment, d.clip_score_max)
    a.scatter(d.clip_score_max, d.alignment, s=10, alpha=0.4, color=col)
    a.set_title(f"{name} vs CLIP  (r={pr:.2f}, ρ={sr:.2f})")
    a.set_xlabel("CLIP clip_score_max"); a.set_ylabel(f"{name} alignment")

fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
