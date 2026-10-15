# Spec Delta: change-acceptance


**Status:** ☑ RATIFIED — implementation permitted per Constitution IX.1.
**Ratified by:** Shiva Kaushik
**Ratified on:** 2026-10-13

## Purpose

Decides, by a rule fixed before any result is seen, whether a candidate change to
the model is kept or dropped, so that small improvements accumulate without
results being read selectively.

## ADDED Requirements

### Requirement: Two measures, judged on the practice exam
A candidate SHALL be judged on two measures: A, order match (Spearman rank
correlation, higher is better), and B, average places off (mean absolute error,
lower is better). It SHALL be compared with the current best model on the
identical practice-exam races. Probability quality (C) SHALL NOT enter the rule.

#### Scenario: Paired comparison
- **WHEN** a candidate is judged
- **THEN** A and B differences are computed race by race between candidate and
  current best, and each difference gets a 95% range from resampling whole races
  with a fixed seed

### Requirement: Meaning of "clearly" better or worse
A measure SHALL count as clearly better only when its whole 95% range lies on the
improving side of zero, and clearly worse only when its whole range lies on the
worsening side. A range that includes zero SHALL count as level.

#### Scenario: Range includes zero
- **WHEN** the B difference has range −0.04 to +0.02 places
- **THEN** B counts as level, neither better nor worse

### Requirement: The keep/drop rule
The system SHALL apply these tiers in order and record the first that matches:
1. keep (both better): A and B clearly better;
2. keep (one better, other level): one clearly better, the other not clearly worse;
3. keep (prefer A): A clearly better, B clearly worse, and the candidate's B still
   beats the grid's B on the same races;
4. otherwise drop.

#### Scenario: Both improve
- **WHEN** A and B are both clearly better
- **THEN** the decision is keep, tier "both better"

#### Scenario: A improves, B level
- **WHEN** A is clearly better and B is level
- **THEN** the decision is keep, tier "one better, other level"

#### Scenario: B improves, A level
- **WHEN** B is clearly better and A is level
- **THEN** the decision is keep, tier "one better, other level"

#### Scenario: A improves at B's cost, B still beats the grid
- **WHEN** A is clearly better, B is clearly worse, and the candidate's average
  places off is lower than the grid's on the practice exam
- **THEN** the decision is keep, tier "prefer A"

#### Scenario: A improves at B's cost, B falls behind the grid
- **WHEN** A is clearly better, B is clearly worse, and the candidate's average
  places off is equal to or higher than the grid's on the practice exam
- **THEN** the decision is drop

#### Scenario: B improves at A's cost
- **WHEN** B is clearly better and A is clearly worse
- **THEN** the decision is drop

#### Scenario: Nothing clear
- **WHEN** both A and B are level
- **THEN** the decision is drop

### Requirement: Grid comparison uses the practice-exam grid
The "beats the grid" test in the prefer-A tier SHALL compare the candidate's
average places off with the starting-grid forecast's average places off on the
same practice-exam races, as point values. It SHALL NOT use a figure from another
exam, such as the final exam's 3.31.

#### Scenario: Grid figure source
- **WHEN** the prefer-A tier is evaluated
- **THEN** the decision record shows the grid's practice-exam average places off
  that was used

### Requirement: Every decision is recorded
Every judgment SHALL write a committed decision record holding: the candidate and
current-best identities, both differences with their ranges, the matched tier, the
decision, and the candidate's A and B against the grid. A kept candidate SHALL
become the current best, and the record SHALL say so.

#### Scenario: Kept candidate becomes the current best
- **WHEN** a candidate is kept
- **THEN** the next judgment compares against that candidate

#### Scenario: Dropped candidate is still recorded
- **WHEN** a candidate is dropped
- **THEN** its decision record is written, and the current best is unchanged

### Requirement: The first current best is today's model
Before any rework change is judged, the system SHALL score today's model on the
practice exam and record it as the first current best.

#### Scenario: Starting point
- **WHEN** the first rework change is judged
- **THEN** it is compared with today's model's practice-exam scores, recorded
  before that change was built
