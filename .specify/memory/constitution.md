# Project Constitution

**Project:** Race Result Predictor
**Version:** 1.0.0
**Ratified:** 2026-08-15

This document defines the non-negotiable rules of this project. Every spec, plan,
and task must comply. Where a spec conflicts with this constitution, the
constitution wins and the spec is wrong.

Amending this document requires an explicit, dated entry in the Amendments
section. It is not edited silently.

---

## Article I — Purpose and Honest Framing

**I.1** This is a **portfolio and research project**. Its purpose is to
demonstrate rigorous ML engineering, not to generate revenue or betting advice.

**I.2** The project SHALL NOT be presented, packaged, or distributed as a
commercial product, a betting tool, or a source of wagering advice.

**I.3** All published claims about predictive performance SHALL be reproducible
from the committed code and a documented data-collection procedure.

**I.4** The project's headline question is falsifiable and stated up front:
*does a learned model predict F1 finishing order better than the starting grid
alone?* A negative answer is a valid, publishable result and SHALL NOT be hidden.

> **Rationale.** The predecessor's README claimed performance the code could not
> reproduce. Overstated claims are the single most damaging thing a portfolio
> project can contain, because they are trivially checked by anyone who cares.

---

## Article II — The Baseline Is Sovereign

**II.1** Every evaluation report SHALL include the **grid-order baseline**
(predicted finishing position = starting grid position) computed on the exact
same holdout, using the exact same metrics.

**II.2** No model may be described as "working", "good", or "accurate" in any
artifact unless it beats the grid-order baseline on the primary metric over a
holdout of at least three distinct seasons.

**II.3** Evaluation SHALL additionally report at least one naive control beyond
grid order (e.g. championship-standings order) to guard against a single
baseline being unrepresentative.

**II.4** Where the model loses to a baseline, the losing comparison SHALL appear
in the README, not only in the logs.

> **Rationale.** The v0 model lost to a one-line baseline on five of six metrics
> and nobody noticed for four months, because the baseline was never computed.
> An uncomputed baseline is an unfalsifiable claim.

---

## Article III — No Temporal Leakage

**III.1** A feature is admissible **only if its value is knowable before the
race start** for the race it describes.

**III.2** Any feature derived from the target race's own session data — results,
actual weather, lap times, retirements — is **inadmissible** in the primary
model.

**III.3** Rolling and aggregate features SHALL be computed from strictly prior
races only. Off-by-one inclusion of the current race is a defect, not a nuance.

**III.4** Where an inadmissible feature is genuinely interesting, it MAY be
evaluated as an explicitly labelled **oracle variant** that establishes an upper
bound. Oracle results SHALL NEVER be reported as the model's performance.

**III.5** Every feature SHALL be accompanied, in code, by a declaration of when
its inputs become available. Features that cannot make this declaration do not
ship.

> **Rationale.** In v0, `weather_mixed` was the third most important feature and
> was computed from the actual weather of the race being predicted. Every metric
> the project reported was inflated by knowing whether it rained.

---

## Article IV — Chronological Evaluation Only

**IV.1** Train/test splits SHALL be chronological. Random splits over race data
are prohibited, as they leak future information into the past.

**IV.2** The holdout SHALL span **at least three distinct seasons**. Evaluation
on a single season, or on fewer than 200 driver-race rows, is not evidence.

**IV.3** Reported metrics SHALL be accompanied by an uncertainty estimate
(bootstrap confidence interval or across-fold standard deviation). A bare point
estimate is not a result.

> **Rationale.** v0's holdout was two races and forty rows. Every metric it
> produced was indistinguishable from noise, in both directions.

---

## Article V — Probabilities Must Be Calibrated

**V.1** Any artifact that emits a probability (win, podium, points) SHALL be
accompanied by a calibration assessment on held-out races.

**V.2** Calibration SHALL be measured by, at minimum, a reliability curve and a
Brier score, each compared against a baseline probability model.

**V.3** Stated intervals SHALL report empirical coverage. An interval labelled
80% that contains the truth 55% of the time is a defect.

**V.4** Uncalibrated probabilities SHALL NOT be surfaced in any user-facing
output, including CLI tables and exported CSVs.

> **Rationale.** v0's simulator emitted confident-looking win probabilities to
> four decimal places with no evidence they meant anything. Precision without
> calibration is decoration.

---

## Article VI — Reproducibility

**VI.1** Given the same inputs and seed, every pipeline stage SHALL produce
identical outputs.

**VI.2** Data collection SHALL be **incremental and resumable**. A run
interrupted at any point SHALL resume without refetching completed work.

**VI.3** Every persisted model artifact SHALL record: training data date range,
row count, feature list, hyperparameters, code version, and evaluation metrics.

**VI.4** No committed artifact may depend on a file that the repository cannot
regenerate or documentedly reacquire.

> **Rationale.** v0's collection died partway through 2024 and could not resume,
> because per-season checkpoints were written to paths that were never created.
> Four seasons of intended data quietly became one.

---

## Article VII — Spec-Driven Development

**VII.1** The order of work is **spec → plan → tasks → code**. Code that no task
requires SHALL NOT be written.

**VII.2** Every task SHALL cite the requirement ID(s) it satisfies. A task
tracing to no requirement is out of scope by definition.

**VII.3** When implementation reveals a spec to be wrong, the **spec is amended
first**, then the code follows. Silently diverging from the spec is the failure
mode this project exists to avoid.

**VII.4** Requirements SHALL be written so that a competent reader can determine,
without consulting the code, whether the requirement is met.

**VII.5** Documentation SHALL describe what the code does, verified by running
it — never what it was intended to do.

---

## Article VIII — Data Provenance and Licensing

**VIII.1** Race data is obtained via FastF1, which sources from Formula 1's
timing feeds. The MIT licence on FastF1 covers the software, not the data.

**VIII.2** Bulk collected race data SHALL NOT be redistributed in this
repository. The repository ships the *code to acquire* data, not the data.

**VIII.3** The project SHALL NOT use Formula 1 trademarks in a way that implies
affiliation or endorsement.

---

## Amendments

*None. This is the ratified 1.0.0 text.*
