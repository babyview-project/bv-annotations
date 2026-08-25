# annotations/language/audio/ — AUDIO-level language identification

**This is the authoritative language measurement for the release.** The transcript-level
annotation (../lang_2026.1.parquet) is contaminated by ASR auto-translation: Whisper renders
non-English caregiver speech as fluent English text, so transcript-based %English reads ~90%
even for households speaking almost no English. Measured against this audio layer:
corr(audio, household survey) = 0.90; corr(audio, transcript-based) = 0.53. The survey was
approximately right; the transcript measurement was the artifact.

## Files
- `audio_lang_2026.1.parquet` — one row per sampled window (47,826 windows, 16,274 videos,
  0 errors). Columns: video_id, child, w (window 0-2), start_s, speech (bool),
  langs (JSON {iso639-1: proportion}), english_prop, child_directed.
- `audio_lang_2026.1.jsonl` — append-log checkpoint (same content).
- `usage_audio_2026.1.json` — 47,825 calls, 79.6M in / 1.1M out tokens (~$27).

## Method
3 windows per video at 20/50/80% of duration, 60 s each (short videos: one window), mono
48kbps mp3 cut with ffmpeg, one gemini-2.5-flash call per window on Vertex AI (project
hs-hs-langcog-gemini — no Google training on our data), temperature 0, JSON output.
Reliability: 94.2% of multi-window videos have all windows on the same side of 50% English.
98% of windows contain speech. Annotator: bv-annotations/language/annotate_audio_language.py.

## Join
video_id is the release name; per-video aggregate = mean english_prop over speech windows.
Consumers: the English training filter (video-level threshold on this aggregate) and the
supplement's survey/transcript/audio comparison table.

## Known limits
- 3x60s samples a ~10 min video (~30%); per-video estimates are means of 1-3 windows.
- Proportions are model judgments, not word counts; interjections/names may be ambiguous.
- Windows without speech are excluded from aggregates.
