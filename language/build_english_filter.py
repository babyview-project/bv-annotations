"""Two-level English filter for a BabyView release, from the per-utterance language annotation.

RULE (state this verbatim in Methods):
  1. VIDEO level  — drop a recording whose English share is below `--video-min` (default 0.50).
     Rationale: in a majority-non-English recording, even the English utterances sit in a
     non-English communicative context, and both the ASR and the language labels are least
     reliable there. Videos with fewer than `--min-utts` labelled utterances are exempt from the
     rule (their share is too noisy to act on) and are KEPT.
  2. UTTERANCE level — within surviving videos, drop utterances labelled non-English.

Why not the child level: the Airtable household survey correlates only r=0.47 with what is on
tape, and filtering by child cost 4.3 pts relative to dropping the same number of RANDOM children
— it deleted mostly-English data from bilingual households. Video-level keeps every child.

POLICY on the annotation's own uncertainty (all switchable):
  - `undecidable` utterances are KEPT and are EXCLUDED from the video-level denominator. They are
    overwhelmingly 1-2 words ("Yeah", "Mama") that carry no language commitment; counting them as
    non-English would make short-utterance-heavy videos look foreign, and dropping them would bias
    the corpus toward long utterances.
  - utterances where the two passes DISAGREE are kept by default (`--drop-disagree` to change),
    and also excluded from the denominator.

usage: python build_english_filter.py --lang lang/lang_2025.2.parquet \
         --pairs manifests/grid_baseline_train.parquet --out manifests/en_video50.parquet
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True, help="lang_<release>.parquet from agree_passes.py")
ap.add_argument("--pairs", required=True, help="training pair manifest to filter")
ap.add_argument("--out", required=True)
ap.add_argument("--video-min", type=float, default=0.50)
ap.add_argument("--min-utts", type=int, default=20, help="below this a video is exempt and kept")
ap.add_argument("--drop-disagree", action="store_true")
ap.add_argument("--report", default="", help="write the per-video decision table here")
a = ap.parse_args()

L = pd.read_parquet(a.lang)
L["child"] = L.video_id.str.split("_").str[0]
# decidable = both passes agreed AND neither called it undecidable
dec = L.agree & ~L.undecidable
L["en"] = L.is_english.fillna(False)

# ---- level 1: video English share over DECIDABLE utterances only ----------------
v = (L.assign(dec=dec, en_dec=L.en & dec)
       .groupby("video_id")
       .agg(n=("utterance_id", "size"), n_dec=("dec", "sum"), n_en=("en_dec", "sum")).reset_index())
v["en_share"] = v.n_en / v.n_dec.where(v.n_dec > 0)
v["exempt"] = v.n_dec < a.min_utts
v["keep_video"] = v.exempt | (v.en_share >= a.video_min)
print(f"videos: {len(v):,} | exempt (<{a.min_utts} decidable) {v.exempt.sum():,} | "
      f"dropped {(~v.keep_video).sum():,} | kept {v.keep_video.sum():,}")

# ---- level 2: within kept videos, drop utterances labelled non-English ---------
keepv = set(v.loc[v.keep_video, "video_id"])
u = L[L.video_id.isin(keepv)].copy()
drop_utt = (u.lang.notna() & ~u.en)                       # confidently non-English
if a.drop_disagree:
    drop_utt |= ~u.agree
u["keep_utt"] = ~drop_utt
print(f"utterances in kept videos: {len(u):,} | dropped non-English {int(drop_utt.sum()):,} "
      f"({100*drop_utt.mean():.2f}%)")

# ---- apply to the pair manifest ------------------------------------------------
P = pd.read_parquet(a.pairs)
ok = u.loc[u.keep_utt, ["video_id", "text"]].drop_duplicates()
before = len(P)
out = P.merge(ok, on=["video_id", "text"], how="inner")
kid_b = P.video_id.str.split("_").str[0].nunique()
kid_a = out.video_id.str.split("_").str[0].nunique()
out.to_parquet(a.out, index=False)
print(f"\npairs {before:,} -> {len(out):,} ({100*len(out)/before:.1f}%)   "
      f"children {kid_b} -> {kid_a}   videos {P.video_id.nunique():,} -> {out.video_id.nunique():,}")
print(f"wrote {a.out}")
if a.report:
    v.to_csv(a.report, index=False); print(f"wrote {a.report}")
