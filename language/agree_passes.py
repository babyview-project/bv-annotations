"""Collapse the two annotation passes into the published table.

`lang` is deliberately NULL where the passes disagree: a disagreement is information (it marks
the ambiguous, usually 1-2 word, usually context-assimilated cases), not noise to average away.
Downstream code should filter on `agree`, or treat null `lang` as "unknown" rather than guessing.

usage: python agree_passes.py --release 2025.2 --out lang/
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--release", required=True)
ap.add_argument("--out", default="lang")
a = ap.parse_args()

K = ["video_id", "utterance_id"]
p0 = pd.read_parquet(f"{a.out}/lang_{a.release}_p0.parquet")
p1 = pd.read_parquet(f"{a.out}/lang_{a.release}_p1.parquet")
for p in (p0, p1):
    p.drop_duplicates(K, inplace=True)

m = p0[K + ["text", "nwords", "lang", "mixed", "undecidable", "chunk_id"]].merge(
    p1[K + ["lang", "mixed", "undecidable", "chunk_id"]], on=K, how="outer",
    suffixes=("_p0", "_p1"))
m["agree"] = m.lang_p0.eq(m.lang_p1) & m.lang_p0.notna()
m["lang"] = m.lang_p0.where(m.agree)
m["is_english"] = m.lang.eq("en").where(m.lang.notna())
m["mixed"] = m.mixed_p0.fillna(False) | m.mixed_p1.fillna(False)
m["undecidable"] = m.undecidable_p0.fillna(False) | m.undecidable_p1.fillna(False)
cols = K + ["text", "nwords", "lang", "lang_p0", "lang_p1", "agree", "is_english",
            "mixed", "undecidable", "chunk_id_p0", "chunk_id_p1"]
out = m[cols].sort_values(K).reset_index(drop=True)
path = f"{a.out}/lang_{a.release}.parquet"
out.to_parquet(path, index=False)

n = len(out)
print(f"wrote {path}: {n:,} utterances")
print(f"  both passes present : {100*(m.lang_p0.notna() & m.lang_p1.notna()).mean():.2f}%")
print(f"  passes agree        : {100*out.agree.mean():.2f}%")
print(f"  English (agreed)    : {100*out.is_english.fillna(False).mean():.2f}% of all; "
      f"{100*out.loc[out.agree,'is_english'].mean():.2f}% of agreed")
print(f"  mixed / undecidable : {100*out.mixed.mean():.2f}% / {100*out.undecidable.mean():.2f}%")
print("\n  top languages (agreed only):")
print(out.loc[out.agree, "lang"].value_counts(normalize=True).head(8).mul(100).round(2).to_string())
