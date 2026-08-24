# language — per-utterance language annotation

What language was each utterance in? Produced with Gemini-2.5-Flash on Vertex, labelling
**blocks of consecutive utterances** so short ones get disambiguated by their conversation.

## Why this annotation exists

The Airtable `percent_english` is a **household survey**, not a description of the recordings.
Measured against the transcripts it correlates only **r = 0.47**, and children recorded as
"0% English" are 96–98% English on tape. Any analysis that conditions on language — filtering a
training corpus, comparing bilingual and monolingual input, estimating how much English a child
actually heard — needs per-utterance labels, not a household average.

An off-the-shelf classifier is not sufficient either. fastText `lid.176` on this corpus:

| | |
|---|---|
| when it says **English** | right **99%** (574/580) |
| when it says **non-English** | right **58%** (323/560) |

Its false positives are the classic short-string artifacts (`it`, `fr`, `es`, `de`, `tr`, `eo`,
`la`) — unsurprising when **43% of utterances are 1–2 words**. Filtering on it would wrongly drop
~21% of decidable utterances.

## Method

- **Chunked, not per-utterance.** Numbered blocks of 40 consecutive utterances from one recording.
  Context is the whole point: a bare "mama" in an otherwise Portuguese conversation is Portuguese.
- **Extractive, not arithmetic.** The model returns a code per numbered line. It is never asked to
  count or to produce a percentage — LLMs are unreliable at aggregation over long inputs, and this
  runs with thinking off.
- **Two passes, offset boundaries.** Pass 1's chunks start half a block later than pass 0's, so
  every utterance is labelled twice from *different* context windows. Agreement between passes is
  the confidence signal, and it is published rather than hidden.

## Schema — `lang_<release>.parquet`, one row per utterance

| column | type | meaning |
|---|---|---|
| `video_id` | str | joins to the release's transcript and frame tree |
| `utterance_id` | int | joins to `merged_transcripts_parsed.csv` |
| `text` | str | the utterance as transcribed |
| `nwords` | int | whitespace token count — **read this before trusting a label** |
| `lang` | str | ISO 639-1 where both passes agree, else `null` |
| `lang_p0`, `lang_p1` | str | the two independent labels |
| `agree` | bool | `lang_p0 == lang_p1` |
| `is_english` | bool | `lang == "en"`; null when the passes disagree |
| `mixed` | bool | either pass flagged code-switching *within* the utterance |
| `undecidable` | bool | either pass judged it impossible even with context |
| `chunk_id_p0/p1` | str | provenance: which block supplied the context |

## Measured reliability (31,665 utterances, three multilingual children, both passes)

| | agreement |
|---|---|
| exact language | **95.2%** |
| English / non-English | **95.6%** |
| 1–2 word utterances | 94.4% |
| 3–7 words | ~97% |
| flagged undecidable by either pass | 3.3% |
| flagged mixed | 0.3% |

## Known biases — read before using

1. **Context assimilation.** By design, ambiguous short lines are pulled toward the chunk's
   dominant language. That is right for "mama" and wrong sometimes: in a Korean-dominant
   recording, plainly English lines ("That way, the screen will be good.") were occasionally
   labelled `ko`. Most pass-to-pass disagreements are of this kind and have a **median length of
   2 words**.
2. **ASR is upstream.** These labels describe the *transcript*, not the *audio*. Whisper
   large-v3 occasionally renders non-English speech as English-looking text (and emits a literal
   `foreign` token); no text-based method can recover what the transcript does not contain.
3. **`undecidable` should usually be kept, not dropped.** It concentrates in 1–2 word utterances
   ("Yeah", "Mama", "No") that carry no language commitment; dropping them biases a corpus toward
   long utterances.
4. **Disagreements are not noise to be averaged away.** Use `agree` as a filter or a weight;
   `lang` is deliberately null where the passes differ.

## Deriving an English-filtered corpus

`build_english_filter.py` implements the two-level rule the vlm-headcam paper uses. State it in
Methods verbatim:

> A recording was excluded if fewer than 50% of its language-decidable utterances were English;
> recordings with fewer than 20 decidable utterances were exempt from this rule. Within retained
> recordings, individual non-English utterances were excluded.

Deliberately **not** a child-level filter. The household survey correlates r = 0.47 with the
recordings, and filtering by child cost 4.3 points relative to dropping the same number of
*random* children — it deleted mostly-English data from bilingual households while keeping the
genuinely non-English content of others. Video-level retains every child.

Uncertainty policy, all switchable:
- `undecidable` utterances are KEPT and excluded from the video-level denominator (they are
  overwhelmingly 1–2 words carrying no language commitment; counting them as non-English would
  make short-utterance-heavy recordings look foreign)
- pass-disagreement utterances are kept by default (`--drop-disagree`) and likewise excluded from
  the denominator

`supplemental_table.py` emits the per-child survey-vs-measured table, including the size of the
gap, the undecidable and disagreement rates, and how many of each child's videos survive the rule.

## Running it

```bash
set -a; . ~/.secrets/vlm-headcam.env; set +a     # Vertex service account
for P in 0 1; do
  python annotate_language.py \
    --transcript /ccn2a/dataset/babyview/2025.2/outputs/merged_transcripts_parsed.csv \
    --release 2025.2 --out lang/ --pass $P --workers 32
done
python agree_passes.py --release 2025.2 --out lang/
```
Resumable: chunks already in the `.jsonl` checkpoint are skipped, so re-running after a failure
is cheap. `--child <subject_id>` restricts to one child for validation.

## Cost (measured, not estimated)

15.3 input + 25.2 output tokens per utterance-label, from 1,935 real calls.

| | calls | input | output |
|---|---|---|---|
| 2025.2, 2 passes | 51k | 31.4M | 51.7M |
| 2026.1, 2 passes | 92k | 56.2M | 92.8M |
| **both** | **143k** | **87.6M** | **144.5M** |

Output dominates. A compact `i:code` response schema would cut it ~4–5× at the cost of the
`mixed`/`undecidable` fields.
