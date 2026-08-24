"""Supplemental table: English exposure per child, by household survey vs measured on tape.

Columns per child: the Airtable survey figure, the measured utterance-level share (decidable
utterances only), how many of the child's videos pass the >=50% video rule, and what fraction of
their pairs survive the filter. The survey-vs-measured gap is the point of the table.
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True)
ap.add_argument("--demog", default="/data2/mcfrank/vlm-headcam/data/BV-Main Demographics-Grid view.csv")
ap.add_argument("--video-min", type=float, default=0.50)
ap.add_argument("--min-utts", type=int, default=20)
ap.add_argument("--out", default="lang/supp_english_by_child.csv")
a = ap.parse_args()

L = pd.read_parquet(a.lang)
L["child"] = L.video_id.str.split("_").str[0]
dec = L.agree & ~L.undecidable
L = L.assign(dec=dec, en_dec=L.is_english.fillna(False) & dec)

v = (L.groupby(["child", "video_id"])
       .agg(n_dec=("dec", "sum"), n_en=("en_dec", "sum")).reset_index())
v["share"] = v.n_en / v.n_dec.where(v.n_dec > 0)
v["keep"] = (v.n_dec < a.min_utts) | (v.share >= a.video_min)

c = (L.groupby("child").agg(utts=("utterance_id", "size"), dec=("dec", "sum"),
                            en=("en_dec", "sum"),
                            undec=("undecidable", "mean"), disag=("agree", lambda s: 1 - s.mean()))
       .reset_index())
c["measured_pct_en"] = (100 * c.en / c.dec.where(c.dec > 0)).round(1)
vv = v.groupby("child").agg(videos=("video_id", "size"), videos_kept=("keep", "sum")).reset_index()
c = c.merge(vv, on="child")

d = pd.read_csv(a.demog, low_memory=False)
d["survey_pct_en"] = pd.to_numeric(d.percent_english.astype(str).str.rstrip("%"), errors="coerce")
c = c.merge(d[["subject_id", "survey_pct_en", "languages", "num_lang"]],
            left_on="child", right_on="subject_id", how="left").drop(columns="subject_id")
c["gap"] = (c.measured_pct_en - c.survey_pct_en).round(1)
c["pct_undecidable"] = (100 * c.undec).round(1)
c["pct_pass_disagree"] = (100 * c.disag).round(1)
c = c[["child", "languages", "num_lang", "survey_pct_en", "measured_pct_en", "gap",
       "utts", "dec", "pct_undecidable", "pct_pass_disagree", "videos", "videos_kept"]]
c = c.sort_values("measured_pct_en")
c.to_csv(a.out, index=False)
print(c.to_string(index=False))
r = c.dropna(subset=["survey_pct_en", "measured_pct_en"])
print(f"\ncorr(survey, measured) = {r.survey_pct_en.corr(r.measured_pct_en):.2f}   "
      f"median |gap| = {r.gap.abs().median():.1f} pts")
print(f"children where the survey is off by >20 pts: {(r.gap.abs() > 20).sum()} of {len(r)}")
print(f"wrote {a.out}")
