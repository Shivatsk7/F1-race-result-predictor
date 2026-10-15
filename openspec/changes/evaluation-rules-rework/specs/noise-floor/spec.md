# Spec Delta: noise-floor


**Status:** ☑ RATIFIED — implementation permitted per Constitution IX.1.
**Ratified by:** Shiva Kaushik
**Ratified on:** 2026-10-13

## Purpose

States how much of a driver's finishing position no pre-race forecast can
predict, in the same units as the model's average places off, so the remaining
room for improvement is measured honestly.

## ADDED Requirements

### Requirement: Luck limit in average-miss units
The system SHALL estimate the luck limit as an average number of places, from pairs
of teammates who started within three grid places of each other and both finished.
The estimate SHALL be in the same units as average places off (B).

#### Scenario: Units match
- **WHEN** the luck limit is reported next to the model's average places off
- **THEN** both are average numbers of places, and the report names the method
  used for each

### Requirement: No subtraction across different units
The system SHALL NOT subtract a spread figure (standard deviation) from an
average-miss figure, or report any difference between them as room for
improvement.

#### Scenario: Room for improvement
- **WHEN** the report states how far the model is from the luck limit
- **THEN** both figures are average-miss values from the same exam's measurement
  rules

### Requirement: Same drivers on both sides of the comparison
The luck limit comes only from drivers who finished and were classified. When the
report sets the model against the limit, it SHALL use the model's average places
off over classified finishers on the same exam. The all-drivers figure SHALL be
shown separately, labelled as including retirements.

#### Scenario: Retired drivers excluded from the comparison
- **WHEN** the model's error is set beside the luck limit
- **THEN** the model figure counts only classified finishers
- **AND** the all-drivers average places off is shown on its own line with that
  label

### Requirement: The estimate states its assumption and its uncertainty
The luck-limit report SHALL state the assumption used to turn a teammate
difference into a single driver's luck, and SHALL give a 95% range for the limit
from resampling whole races.

#### Scenario: Report contents
- **WHEN** the luck limit is written to its results file
- **THEN** the file holds the point value, its range, the number of teammate pairs,
  and a plain sentence naming the assumption
