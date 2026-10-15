# Design: new rules for judging the model


**Status:** ☑ RATIFIED — implementation permitted per Constitution IX.1.
**Ratified by:** Shiva Kaushik
**Ratified on:** 2026-10-13

## Context

See `proposal.md` (Why) for the motivation. The code facts that shape the approach:

- **Test seasons come from the clock.** `ChronologicalSplit`
  (`src/f1pred/evaluate/split.py:36`) resolves its seasons through `CorpusWindow`
  (`src/f1pred/corpus.py:27`), a 16-season window ending at the last finished
  season. Its holdout is the newest three seasons. When 2026 finishes, it moves
  from 2023–2025 to 2024–2026.
- **The loader reads everything on disk.** `load_corpus(seasons=None)`
  (`src/f1pred/dataset.py:97`) loads every season file present, and `prepare()`
  calls it that way.
- **Six places build a 2023–2025 split:** `cli.py` (train, evaluate, simulate),
  `runner.run`, `calibration`, `fit_diagnostic` and `oracle_arm` through
  `default_split()`, and `telemetry_arm` with its own window. Each one scores, or
  trains around, the seasons that become the final exam.
- **A one-season split already exists.** `TransferSplit`
  (`src/f1pred/evaluate/regime_transfer.py:132`) holds explicit training seasons
  plus one evaluated season, and the ranker already trains against it.
- **The scoring and bootstrap pieces are reusable as they are.**
  `score_predictor` (harness.py) produces per-race scores. `bootstrap_metrics` and
  `bootstrap_delta` (bootstrap.py) resample whole races, and the delta is paired.
  The practice exam and the keep/drop rule need nothing new from them.
- **The ranker takes its early-stopping and interval-calibration seasons from the
  tail of whatever training seasons it is given** (`ranker.py:677`). So each
  practice season's model tunes itself only on seasons before it, with no extra
  work.
- **The luck limit mixes units twice** (`noise_floor.py:198`, `:257`). It
  subtracts a spread (standard deviation) from an average miss, and it compares
  finishers-only pairs against an error that includes retired drivers. A rough
  re-check on the corpus (1,533 pairs) gives about 2.1 places as an average miss,
  against the 2.8 reported. The model's finishers-only error is not yet measured.

## Goals / Non-Goals

**Goals:**
- Make the season roles structural. Scoring a final-exam or sealed season without
  an opening, or loading a sealed season without opening it, raises an error.
  A convention that relies on people remembering is not enough.
- Reuse the existing scoring, bootstrap and one-season split code. Build no
  parallel machinery.
- Keep every existing results file readable as a historical record.

**Non-Goals:**
- Collecting 2026 data and actually opening the bonus exam. Both happen at the end
  of the rework. This change builds and tests the gate with fixture data.
- Moving the diagnostics (calibration, oracle arm, fit diagnostic, telemetry arm)
  onto the practice exam. They stay on 2023–2025, and re-running one now counts as
  a logged final-exam opening. The probability-correction change moves calibration
  later.
- Retraining the forecasting model (`predict`) on the newest seasons. It keeps
  training on 2010–2022 for now.

## Decisions

### 1. The exam plan is a committed, versioned record, not a rolling window
`config.py` gains an `EXAM_PLAN` (version, season roles, `rework_complete` flag).
Code writes it to `results/exam_plan.json`, and a test checks that file against
the constant.
- *Why not keep the window?* It moves the test set by itself (see Context). The
  window was built (plan D-9) so that nobody forgets to update a hard-coded season
  list. Here the risk runs the other way: the roles must not move unless someone
  deliberately amends them.
- `CorpusWindow` stays for collection, which is what it is still useful for.

### 2. Practice exam: generalise the one-season split, then pool
`TransferSplit` moves into `split.py` as `SeasonFold` (training seasons plus one
scored season). `regime_transfer` imports it from there. The practice exam builds
three folds: (2010–2019 → 2020), (2010–2020 → 2021), (2010–2021 → 2022).

Each fold trains a model and scores it, the grid and the standings with
`score_predictor`. The per-race frames are joined, and the existing bootstrap runs
over the pooled ~61 races. Per-season figures come from the same frames.
- `train()` is typed against `ChronologicalSplit` but only uses `.split()`,
  `.assert_no_leakage()` and `.training_seasons`. A small protocol
  (`TrainingBoundary`) names those three, so both split types are accepted
  explicitly rather than by accident.
- The pooled three seasons meet the Constitution's three-season minimum for a
  test set (Article IV.2). The amendment says so in words, so no one has to
  interpret the rule.
- *Alternative rejected:* one model trained on 2010–2019 scoring all of
  2020–2022. Fewer training runs, but the 2022 season would be predicted with
  two-year-old knowledge. That is not how the model is used, and it understates
  what the model can do.

### 3. The gate sits on the split and on the loader
- **Splits.** Building a `ChronologicalSplit` or `SeasonFold` whose scored seasons
  include final-exam or bonus seasons requires an `ExamOpening`. Only the exam log
  creates one, and it does so by appending the "opened" entry. `default_split()`
  is removed. Its callers either take the practice exam or demand an opening.
  This also protects the diagnostics: re-running the oracle arm without an
  opening fails loudly.
- **Loader.** `load_corpus` and `prepare` drop sealed seasons unless they are
  passed an opening for the bonus exam, or a forecast-inputs purpose (for
  `predict`). That purpose allows reading the season but never training on or
  scoring it. The fold or split checks also stop training on it.
- *Why both?* The split gate stops scoring. The loader gate stops a sealed season
  from reaching training through a path that never builds a split.

### 4. The exam log is an append-only, hash-chained JSON file
`results/exam_log.json` holds a list of entries. Each entry stores a SHA-256
fingerprint of the previous entry. The whole chain is checked before anything is
appended, and a broken chain refuses with the first bad index.
- One opening produces two entries: "opened" (written before scoring) and
  "result" (written before display). A crashed run still leaves "opened" behind.
- *Alternative rejected:* checking the log against git history. It depends on the
  run's environment, and it fails in a fresh clone that has no history.

### 5. The keep/drop rule is a pure function, and its inputs are stored
`acceptance.decide(delta_A, delta_B, candidate_B, grid_B) -> Decision` implements
the tiers in `specs/change-acceptance`. It is pure so that every tier, and every
edge where a range touches zero, can be tested without a model.
- `results/current_best.json` holds the current best's identity and its
  per-race practice-exam scores. Judging a candidate does not retrain the current
  best. `bootstrap_delta` already refuses frames that cover different races.
- Each judgment writes `results/decisions/NNN-<candidate>.json`.
- The stored scores carry a `SCORING_VERSION`. If the scoring code changes, the
  stored current best is refused until it is re-scored. Stale numbers would
  otherwise quietly enter a paired difference.
- **Assumption, for review:** "candidate's B still beats the grid" compares point
  values, not a confidence range. Today's model does not clearly beat the grid on
  B, so a range-based test would make the prefer-A tier unreachable at the start.

### 6. Command surface

| Command | After this change |
|---|---|
| `f1pred evaluate` | Practice exam for today's settings; writes `results/practice_exam_2020_2022.json` |
| `f1pred judge CANDIDATE` | Practice exam plus a decision record; `--initial` records the first current best |
| `f1pred exam final --stage NAME` | Opens the final exam; logged |
| `f1pred exam bonus` | Refuses until 2026 has finished, is collected, and `rework_complete` is set |
| `f1pred simulate` | Unchanged purpose. Scores 2023–2025, so it now requires `--stage` and is logged |
| `f1pred train` | Trains on 2010–2022; `--holdout-seasons` removed |

Candidates are registered by name in code (`evaluate/candidates.py`). This change
registers only `today`, the current ranker and settings. Later changes add theirs.

### 7. Luck limit, like for like
`teammate_pair_floor` changes in two ways:
- It uses the absolute difference, so which teammate is listed first does not
  matter.
- It reports the average miss per driver as mean|difference|/√2. That step
  assumes each teammate's luck has the same symmetric shape. The report also gives
  mean|difference|/2, a lower bound that needs no such assumption.

The 95% range comes from resampling races. `error_relative_to_floor` takes the
model's finishers-only average places off. A new finishers-only error measure is
added to the metric battery as a reported figure, not as a decision input.

## Risks / Trade-offs

- **Random-seed luck can look like an improvement.** The race bootstrap covers
  which races were run, not the training seed. A change could pass on seed luck.
  → Mitigation: every judgment uses the same fixed seed. If a later change shows
  seed sensitivity, a seed-averaging requirement is proposed then, not invented
  now.
- **61 practice races still give wide ranges** (about ±0.03 on A). Real gains
  smaller than that will be dropped. → Accepted. It is the honest cost, and the
  rule's middle tier ("one better, other level") is what lets smaller real gains
  through.
- **Three training runs per judgment.** → About 75 seconds expected, measured and
  recorded against the 300-second budget.
- **The 2022 rule reset sits inside the practice exam.** Changes that help across
  a rule reset will look better than they do in a stable year. → Per-season
  figures are always reported next to the pooled one.
- **Diagnostics now need a logged opening to re-run.** → Intended. Each re-run is
  a look at the final exam.
- **The `--holdout-seasons` removal is a breaking change for scripts.** → Only the
  project's own commands use it. The README and command help are updated in the
  same change.

## Migration Plan

1. Add the exam plan, `SeasonFold`, the gates and the log. Existing tests for
   splits and the CLI are updated to the new defaults.
2. Fix the luck limit and regenerate `results/noise_floor.json`. The old figure is
   kept in the change notes as the "before".
3. Run `f1pred judge today --initial` to record the first current best on the
   practice exam.
4. Leave the existing 2023–2025 results files untouched. Their file names already
   identify them as final-exam-era records.

Rollback is a git revert. Nothing is deleted from `results/`, and no data is
re-collected.

## Open Questions

- What the stage names will be for final-exam openings (for example, one per
  rework change, or one per group). This doesn't affect the mechanism. It is
  decided when the first stage finishes.
