# Proposal: new rules for judging the model

**Status:** ☑ RATIFIED — implementation permitted per Constitution IX.1.
**Ratified by:** Shiva Kaushik
**Ratified on:** 2026-10-13

Change 1 of 6 in the model rework. Every later change is judged by the rules this
one sets, so it goes first.

## Why

The model does not beat the simplest possible forecast. On the 70 test races
(2023–2025) its order match is 0.660 against the starting grid's 0.655. The 95%
range on that gap runs from −0.024 to +0.032, so the difference may be zero. On
average places off it is slightly *worse* than the grid (3.37 against 3.31).
The rework will try many small improvements. Today's testing setup cannot judge
them honestly, for three reasons:

1. **There is one test set, and it gets reused.** Every check scores the same
   2023–2025 races. If we try twenty changes and keep whichever scores best, some
   "improvements" will be luck that only fits those 70 races.
2. **There is no written rule for keeping a change.** A and B (below) are now equal
   priorities, and nothing says what to do when one improves and the other
   worsens. A rule picked after the results are in can be bent to fit them.
3. **The luck limit is compared unfairly.** The report says the model is only 0.58
   places above the best a pre-race forecast could do. That figure subtracts a
   *spread* measure (standard deviation) from an *average miss*, and the two are
   different units. Measured like for like, the real room is probably larger, about
   1 place, and the current figure understates how much is left to win.

Two problems with the season setup itself, found while reading the code:

- **The test seasons move by themselves.** Test seasons come from a rolling
  16-season window. When 2026 finishes, the window shifts to 2011–2026 and the test
  set becomes 2024–2026 with no code change. 2023, today's test season, would
  quietly become training data mid-rework.
- **New data leaks into everything automatically.** The loader reads every season
  file on disk. Once 2026 is downloaded, ordinary training and scoring would
  include it, and the "never seen" 2026 exam would already have been seen.

## What Changes

- **Seasons get fixed roles,** written down once and changed only by amendment:

  | Seasons | Role | Used for |
  |---|---|---|
  | 2010–2019 | Learning only | Training, never scored as a test |
  | 2020–2022 | **Practice exam** | Every keep/drop decision. Each season is predicted by a model trained only on earlier seasons. |
  | 2023–2025 | **Final exam** | Opened only when a stage of work finishes. Every opening is logged. Scores are reported, never used to choose. |
  | 2026 | **Bonus exam (sealed)** | Opened once, at the very end of the rework, after the 2026 season has finished |

- **A written keep/drop rule.** A = order match (Spearman rank correlation, the
  project's primary score); B = average places off (mean absolute error). Each
  candidate change is compared with the current best model on the practice exam:
  1. **Keep (both better):** A and B both clearly improve.
  2. **Keep (one better, other level):** one clearly improves, the other is not
     clearly worse.
  3. **Keep (prefer A):** A clearly improves, B gets worse, but B still beats the
     starting grid's B on the same races.
  4. Otherwise **drop**.

  "Clearly" means the 95% range on the paired difference sits wholly on the good
  side of zero. C (honest win, podium and points chances) is not part of the rule.
- **Sealing.** Sealed and final-exam seasons are kept out of training and scoring
  unless a specific command explicitly opens them. A sealed season on disk does
  nothing until it is opened.
- **Exam log.** A committed, append-only record of every final and bonus exam
  opening: date, code version, stage name, scores.
- **Luck limit fixed.** The limit is reported as an average miss, so it compares
  directly with B. The old subtraction is removed.
- **BREAKING (commands):** `f1pred evaluate` will score the practice exam by
  default, not 2023–2025. Final and bonus exams move behind their own commands,
  which log every opening. `--holdout-seasons` is removed from `evaluate`,
  `train` and `simulate`, and `simulate` (which scores 2023–2025) now needs a stage
  name and is logged, because the season roles are no longer chosen per run.
- **Constitution amendment (to version 1.4.0):** a new section in the chronological
  evaluation article (Article IV) sets out the exam roles, the ban on tuning against
  the final exam, and the rule that a one-season exam (the 2026 bonus) is
  supplementary and never on its own evidence that the model works. Only the
  owner can ratify it.

**Not in this change:** downloading 2026 data (it cannot happen until the season
finishes), and model improvements: collecting qualifying lap times, predicting
places gained from the grid, settings search, a separate retirement model,
probability correction. Each is a later change judged by these rules.

## Capabilities

### New Capabilities
- `evaluation-protocol`: which seasons play which exam role, how each exam is
  scored (practice exam walk-forward, one model per season), sealing, and the exam
  log.
- `change-acceptance`: the keep/drop rule for a candidate model change, judged
  against the current best model on the practice exam.
- `noise-floor`: the luck limit, stated in the same units as average places off.

### Modified Capabilities
None in `openspec/specs/` (it is empty). This change amends requirements in the
hand-written spec (`specs/001-core-predictor/spec.md`): the fixed-width rolling
window (FR-508) no longer picks the test seasons; the holdout scored by `evaluate`
(FR-501, FR-801..804) becomes the practice exam; and the noise-floor comparison
(FR-755) changes units. Those requirement texts return to Draft for re-ratification
of the amended sections (Constitution IX.5).

## Impact

- **Code:** `src/f1pred/evaluate/split.py` (new walk-forward fold and exam plan),
  `src/f1pred/corpus.py` and `src/f1pred/dataset.py` (sealing at load time),
  `src/f1pred/evaluate/harness.py` and `runner.py` (practice exam run, decision
  record), `src/f1pred/evaluate/noise_floor.py` (units fix), `src/f1pred/cli.py`
  (commands).
- **Results:** new `results/exam_plan.json`, `results/exam_log.json`, and a
  baseline practice exam result for the current model, which becomes the first
  "current best". The current-model results file (`model_vs_baselines_2023_2025.json`)
  stays as the record of where the rework started.
- **Governance:** `.specify/memory/constitution.md` gains an amendment entry, and
  the affected requirement sections of spec 001 are marked for re-ratification.
- **Cost:** a practice exam trains three models instead of one, so the run takes
  roughly three times as long. Today's one-model run took 24 seconds (measured
  2026-09-05), so about 75 seconds is expected, inside the 300-second budget
  (NFR-01). The time is measured and recorded, not assumed.
