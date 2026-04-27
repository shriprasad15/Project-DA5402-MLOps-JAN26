# Unified Label Schema & Per-Source Mappings

All ingested sources are normalized into a single DataFrame schema before being
written to `data/processed/{train,val,test}.parquet`.

## Unified schema

| Column                | Type    | Range / Domain                                                              |
| --------------------- | ------- | --------------------------------------------------------------------------- |
| `text`                | string  | Cleaned utterance                                                           |
| `passive_aggression`  | float   | `[0.0, 1.0]`                                                                |
| `sarcasm`             | float   | `[0.0, 1.0]`                                                                |
| `tone`                | string  | `{neutral, friendly, assertive, aggressive, passive_aggressive}` (see `contracts.tone_enum.Tone`) |
| `source`              | string  | Originating dataset identifier                                              |
| `weak_label`          | bool    | `True` when label is heuristic / distant-supervised                         |

## Per-source mapping

| Source               | Native label(s)              | `passive_aggression` | `sarcasm`             | `tone`                                                         | `weak_label` |
| -------------------- | ---------------------------- | -------------------- | --------------------- | -------------------------------------------------------------- | ------------ |
| `sarcasm_headlines`  | `sarcasm ∈ {0, 1}`           | `0.0`                | `float(sarcasm)`      | `passive_aggressive` if `sarcasm > 0.5` else `neutral`         | `True`       |
| `isarcasm`           | `sarcasm ∈ {0, 1}`           | `0.0`                | `float(sarcasm)`      | `passive_aggressive` if `sarcasm > 0.5` else `neutral`         | `True`       |
| `goemotions`         | `labels: list[int]` (simplified) | `0.0`            | `0.0`                 | `aggressive` if anger group hit, `friendly` if joy group hit, else `neutral` | `True`       |
| `enron_subset`       | none (unlabeled)             | `0.0`                | `0.0`                 | `neutral`                                                      | `True`       |
| `synthetic_*`        | pre-scored by generator      | pass-through         | pass-through          | pass-through                                                   | `True`       |

## Rationale notes

- **GoEmotions index groupings.** We collapse the 28-class GoEmotions simplified
  label set onto our 5-tone enum using three index buckets:
  anger = `{2, 3, 16}` (anger, annoyance, disapproval),
  joy = `{17, 18, 0}` (joy, love, admiration),
  neutral = `{27}`. Anything not matching anger or joy falls back to `neutral`.
  Anger outranks joy when both are present to avoid masking aggression with a
  polite co-label.
- **Enron kept neutral / weak.** The Enron subset is an unlabeled email corpus
  used to ground the text distribution of production-like emails. We hold tone
  at `neutral` and rely on a downstream heuristic PA scoring pass (post
  synthesis) rather than forging labels from nothing.
- **Synthetic rows pass through unchanged.** Rows produced by the synthetic
  generator (`synthetic_*`) already carry validated `passive_aggression`,
  `sarcasm`, and `tone` fields, so `to_unified` copies them verbatim.
- **`weak_label = True` for every source in Wave 1.** Once gold-annotated data
  arrives in Wave 2, its adapter will set `weak_label = False`.
