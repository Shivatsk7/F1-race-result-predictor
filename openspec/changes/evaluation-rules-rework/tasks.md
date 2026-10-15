# Tasks

**Status:** ☑ RATIFIED — implementation permitted per Constitution IX.1.
**Ratified by:** Shiva Kaushik
**Ratified on:** 2026-10-13

## 1. Governance (gate before any code)

- [x] 1.1 Add the proposed Constitution amendment to `.specify/memory/constitution.md`: a new section in Article IV covering the exam roles, tuning only on the practice exam, final-exam openings logged and never used to choose, and a one-season exam being supplementary only. It also states that a pooled three-season practice exam meets the three-season minimum in Article IV.2. Add an Amendments row for v1.4.0 marked "Proposed — not ratified". Verify with `git diff` that no other article text changed.
- [x] 1.2 In `specs/001-core-predictor/spec.md`, mark the amended requirements as returned to Draft for re-ratification, naming this change, and untick their ratification-checklist sections. These are the rolling window (FR-508), the holdout (FR-501), the noise floor (FR-755) and the command list (FR-801); FR-802..804 (error messages, CSV input, number precision) do not change. Verify with `grep -n "evaluation-rules-rework" specs/001-core-predictor/spec.md`, which shows one marker per amended requirement.
- [x] 1.3 STOP for the owner to ratify the amendment and re-ratify the marked sections, with name and date recorded in each header (Constitution IX.4). Verify the headers show it before starting group 2.

## 2. Exam plan

- [x] 2.1 Add `EXAM_PLAN` to `src/f1pred/config.py` (version, season roles, `rework_complete=False`) and a function that returns the role of a season. Verify with new tests: every season 2010–2026 has exactly one role, and roles are unchanged when the date is set to 2027-03-01.
- [x] 2.2 Write `results/exam_plan.json` from the constant. Verify with a test that fails if the file and the constant disagree. *(Done: the file also carries the standard corpus stamp, marked "not corpus" with its reason, because every results file must; the check compares everything except that stamp.)*

## 3. One-season split and the training boundary

- [x] 3.1 Move `TransferSplit` from `src/f1pred/evaluate/regime_transfer.py` to `src/f1pred/evaluate/split.py` as `SeasonFold`, and import it back into `regime_transfer`. Verify that `tests/test_regime_transfer.py` passes without edits to its assertions.
- [x] 3.2 Add a `TrainingBoundary` protocol (`split`, `assert_no_leakage`, `training_seasons`) and type `ranker.train()` against it. Verify with tests that `train()` accepts both split types and refuses a fold whose training rows include the scored season or later. *(Done: `TransferError` is now the fold's own error type, so the transfer tests catch it unchanged. `train()` also checks that the rows it was handed come only from the split's declared training seasons, so a faulty split cannot widen the fit even if its own check is missing.)*

## 4. Exam log and the gates

- [ ] 4.1 Add `src/f1pred/evaluate/exam_log.py`: an append-only `results/exam_log.json` with a fingerprint chain, "opened" and "result" entries, and a chain check before every append. Verify with tests that appending works, that an edited, deleted or reordered entry is refused with the first bad index, and that an "opened" entry stays when the run errors.
- [ ] 4.2 Make `ExamOpening` obtainable only from the exam log. Require one when building any split or fold that scores a final-exam or bonus season, and remove `default_split()`. Verify with tests that building a split that scores 2023 without an opening raises, and that it works with one.
- [ ] 4.3 Seal bonus seasons in `load_corpus` and `prepare` (`src/f1pred/dataset.py`), with explicit purposes for bonus-exam and forecast inputs. Verify with tests that use a fixture 2026 checkpoint in a temporary directory: a default load, training and the practice exam never see 2026; forecast inputs can read it but cannot train on it; a bonus opening can read it.
- [ ] 4.4 Make `calibration`, `fit_diagnostic`, `oracle_arm` and `telemetry_arm` require an opening when they score 2023–2025. Update their tests. Verify that their test files pass and that a call without an opening raises.

## 5. Practice exam

- [ ] 5.1 Add the practice exam run. It builds three folds (2010–2019 → 2020, 2010–2020 → 2021, 2010–2021 → 2022) and scores the model, grid, standings and the random control per fold with `score_predictor`. It pools the per-race scores for the existing race bootstrap and also reports each season, and the results file records the exam name and plan version. Verify with fixture-corpus tests: each fold's training seasons are correct, no 2023+ row enters any training set, and the pooled race count equals the sum of the folds.
- [ ] 5.2 Run the practice exam on the real corpus for today's model. Verify that the race count matches the corpus's 2020–2022 races, that the wall-clock time is recorded and under 300 seconds, and that `results/practice_exam_2020_2022.json` holds the model-vs-grid verdict.

## 6. Keep/drop rule

- [ ] 6.1 Add `acceptance.decide()` as a pure function implementing the four tiers. Verify with table-driven tests covering every scenario in `specs/change-acceptance/spec.md`, plus the edge where a range ends exactly at zero (which counts as level).
- [ ] 6.2 Add `results/current_best.json` (identity, per-race practice scores, `SCORING_VERSION`), decision records under `results/decisions/`, and a candidate registry `src/f1pred/evaluate/candidates.py` holding `today`. Verify with tests: a kept candidate becomes the current best; a dropped one leaves it unchanged; a `SCORING_VERSION` mismatch is refused; a decision record contains no final-exam or bonus-exam figure.
- [ ] 6.3 Run `f1pred judge today --initial` on the real corpus. Verify that `results/current_best.json` exists and that its per-race scores equal those in `results/practice_exam_2020_2022.json`.

## 7. Luck limit

- [ ] 7.1 Fix `teammate_pair_floor` in `src/f1pred/evaluate/noise_floor.py`: use the absolute difference, report the average miss as mean|difference|/√2 with the assumption-free lower bound mean|difference|/2, and add a 95% range from resampling races. Verify with tests: swapping teammates leaves the result unchanged, and a fixture with a known distribution gives the expected value.
- [ ] 7.2 Add a finishers-only average-places-off measure as a reported figure, not a decision input, and make `error_relative_to_floor` use it. Verify with tests that retired drivers are excluded and that the all-drivers figure is reported on its own labelled line.
- [ ] 7.3 Regenerate `results/noise_floor.json` on the real corpus. Verify that the pair count is close to the previous 1,540, and that the headline is in average-miss units and near the rough check of about 2.1 places. Any large gap is investigated, not accepted.

## 8. Commands and documentation

- [ ] 8.1 Update `src/f1pred/cli.py`: `evaluate` runs the practice exam; add `judge CANDIDATE [--initial]`, `exam final --stage NAME` and `exam bonus`; `simulate` requires `--stage`; remove `--holdout-seasons`. Verify with `tests/test_cli.py` covering each command and the missing-stage refusal.
- [ ] 8.2 Implement the `exam bonus` refusals. Verify with fixture tests covering four cases: refuses when the season is unfinished, refuses when it is not collected, refuses when `rework_complete` is false, and labels a second opening as a repeat that is not evidence.
- [ ] 8.3 Update `README.md` in plain language: the four season roles, the keep/drop rule, and the corrected luck limit. Show the model's practice-exam result against the grid, including a loss if there is one (Constitution II.4). Verify that every command shown in the README runs as written.
- [ ] 8.4 Update `results/README.md` to describe `exam_plan.json`, `exam_log.json`, `current_best.json`, `decisions/` and `practice_exam_2020_2022.json`. Verify that every file in `results/` is listed.

## 9. Integration checks

- [ ] 9.1 Run the full test suite and `graphify update .`. Verify that all tests pass and that the output is recorded in the change notes.
- [ ] 9.2 Run `openspec validate evaluation-rules-rework --strict`. Verify that it passes.
- [ ] 9.3 Map every scenario in the three spec files to the test that covers it, as a table in the change notes. Verify that no scenario is left without a test.

## Workflow follow-up

- Archive the change after the owner reviews the finished work, then sync the three new specs into `openspec/specs/`.
- Next change in the rework: collect qualifying lap times, judged by these rules.
