# Spec Delta: evaluation-protocol


**Status:** ☑ RATIFIED — implementation permitted per Constitution IX.1.
**Ratified by:** Shiva Kaushik
**Ratified on:** 2026-10-13

## Purpose

Decides which seasons the model may learn from and which seasons test it, so that
every score reported for the model is measured on races it has not seen and that
no test set is worn out by repeated use.

## ADDED Requirements

### Requirement: Fixed season roles
The system SHALL assign every season one role from a single committed exam plan:
learning-only 2010–2019, practice exam 2020–2022, final exam 2023–2025, bonus exam
2026. The roles SHALL NOT change when the calendar or the data on disk changes;
only a recorded amendment SHALL change them.

#### Scenario: Roles survive a season rollover
- **WHEN** the 2026 season has finished and its data is on disk
- **THEN** the practice exam is still 2020–2022 and the final exam still 2023–2025
- **AND** no command treats 2026 as training data or as part of the final exam

#### Scenario: Every results file names its exam
- **WHEN** any evaluation writes a results file
- **THEN** the file records which exam it scored, the seasons in it, and the
  exam plan's version

### Requirement: Practice exam is scored season by season
The practice exam SHALL score each of its seasons with a model trained only on
seasons before it, and SHALL pool all practice-exam races into one set of
per-race scores for the confidence ranges.

#### Scenario: No season is predicted with knowledge of itself or later seasons
- **WHEN** the practice exam scores the 2021 season
- **THEN** the model used for 2021 was trained only on 2010–2020
- **AND** a model whose training rows include 2021 or later is refused with an
  error naming the offending seasons

#### Scenario: One pooled result
- **WHEN** the practice exam finishes
- **THEN** it reports one score per measure over all practice races (about 61),
  each with a 95% range from resampling whole races
- **AND** it also reports each season separately

#### Scenario: The grid is always scored alongside
- **WHEN** the practice exam scores any model
- **THEN** the starting-grid forecast and the standings forecast are scored on the
  identical races, with the same measures

### Requirement: Final exam is opened only deliberately and every opening is logged
The system SHALL score the final exam only through a dedicated command that
requires the name of the finished stage of work, and SHALL append every opening to
the exam log before showing any score.

#### Scenario: Opening the final exam
- **WHEN** a person opens the final exam naming a finished stage
- **THEN** an "opened" entry with the date, code version, stage name and model
  identity is added to the exam log before any final-exam race is scored
- **AND** a "result" entry with every score and its range is added before any
  score is shown

#### Scenario: A run that fails partway
- **WHEN** a final-exam run stops with an error after it was opened
- **THEN** the "opened" entry stays in the log, so the opening still counts

#### Scenario: Opening without naming a stage
- **WHEN** the final exam is requested without a stage name
- **THEN** the command refuses and no score is computed

#### Scenario: Ordinary evaluation never touches it
- **WHEN** a person runs the ordinary evaluate command
- **THEN** it scores the practice exam, and no final-exam race is scored

### Requirement: Final exam scores never choose between changes
The keep/drop decision for a candidate change SHALL be computed from practice-exam
scores only. A decision record SHALL NOT read or cite a final-exam or bonus-exam
score.

#### Scenario: A decision record is built
- **WHEN** a candidate change is judged
- **THEN** its decision record contains practice-exam figures only

### Requirement: Sealed seasons stay out until opened
Seasons with the bonus-exam role SHALL be excluded from every corpus load, training
run, and evaluation unless the caller explicitly opens them for a stated purpose.
Final-exam seasons SHALL be loadable for training only by final-exam and
bonus-exam runs.

#### Scenario: 2026 data downloaded but sealed
- **WHEN** the 2026 season file exists on disk and a person runs train, evaluate,
  or the practice exam
- **THEN** no 2026 row is read, trained on, or scored

#### Scenario: Practice exam cannot learn from the final exam
- **WHEN** the practice exam trains its models
- **THEN** no row from 2023, 2024 or 2025 is in any of its training sets

#### Scenario: Forecasting a live race
- **WHEN** a person asks for a forecast of a race in a sealed season
- **THEN** earlier races of that season may be read to build the race's inputs
- **AND** the model used was trained without that season, and nothing is scored

### Requirement: Bonus exam opens once, at the end, on a finished season
The bonus exam SHALL open only when its season has finished, its data is collected,
and the exam plan marks the rework complete. The first opening is the result; any
later opening SHALL be logged and labelled a repeat that is not evidence.

#### Scenario: Season not yet finished
- **WHEN** the bonus exam is requested before the 2026 season's last race
- **THEN** the command refuses and names the reason

#### Scenario: A second opening
- **WHEN** the bonus exam is opened after it has already been opened once
- **THEN** the log entry is labelled a repeat and the scores are shown under that
  label

#### Scenario: One season is not enough on its own
- **WHEN** bonus-exam scores are reported
- **THEN** the report states that a single season is supplementary and that it
  cannot by itself show the model works

### Requirement: The exam log is append-only and committed
The exam log SHALL be a committed results file whose entries are never edited or
removed. Each entry SHALL carry a fingerprint of the entry before it, so that an
edited or deleted entry is detectable from the file alone.

#### Scenario: Tampered log
- **WHEN** an existing log entry has been changed or deleted
- **THEN** the next opening refuses to run and reports the first entry whose
  fingerprint no longer matches
