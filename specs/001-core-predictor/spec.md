# Spec 001 — Core Predictor

**Status:** ☐ DRAFT — NOT RATIFIED. Implementation blocked per Constitution IX.1.
**Drafted by:** Claude (agent). Per Constitution IX.3, the drafting agent may not ratify.
**Ratified by:** _pending — Shiva Kaushik_
**Ratified on:** _pending_
**Created:** 2026-08-15
**Governed by:** `.specify/memory/constitution.md` v1.2.0

> **Ratification checklist** — tick each section once read in full (IX.2):
>
> - [ ] §1 Problem statement — the question is the right one to ask
> - [ ] §2 User stories
> - [ ] §3.1 Data acquisition (FR-101..107)
> - [ ] §3.2 Feature engineering (FR-201..208) — *the leakage rules*
> - [ ] §3.3 Baselines (FR-301..305)
> - [ ] §3.4 Model (FR-401..405)
> - [ ] §3.5 Evaluation (FR-501..507) — *incl. choice of primary metric*
> - [ ] §3.6–3.7 Simulation and calibration (FR-601..605, FR-701..706)
> - [ ] §3.7a **Race randomness** (FR-751..755) — *Article X: noise floor, proper scoring*
> - [ ] §3.8–3.9 Interface and documentation (FR-801..804, FR-901..904)
> - [ ] §4 Non-functional requirements
> - [ ] §5 **Out of scope** — every deletion in P0 was justified from this list
> - [ ] §6 Acceptance criteria — what "done" means
> - [ ] §7 Open questions

> This document states **what** the system must do and **why**. It contains no
> implementation decisions — those live in `plan.md`. If you find yourself
> writing a module name or a library here, it belongs in the plan.

---

## 1. Problem Statement

Formula 1 finishing order is strongly determined by qualifying position. Any
predictive model of race results is therefore competing against a trivially
available signal: the starting grid.

The v0 system trained a 50-feature XGBoost ranker and lost to that trivial
signal on five of six metrics, while reporting inflated numbers derived from
post-race weather information.

**This project asks one falsifiable question and answers it rigorously:**

> Given only information available before lights out, can a learned model
> predict F1 finishing order measurably better than the starting grid alone —
> and can it express its uncertainty in probabilities that are actually
> calibrated?

The deliverable is a defensible answer, whichever way it falls.

---

## 2. User Stories

**US-1 — Reproducible dataset.**
As the author, I can rebuild the full multi-season dataset from a clean machine
with a single documented command, resuming safely if interrupted, so that anyone
can verify my results.

**US-2 — Honest evaluation.**
As the author, I can run one command that reports my model's performance
side-by-side with naive baselines on a multi-season holdout, with uncertainty
bounds, so that I never again make a claim I cannot support.

**US-3 — Calibrated probabilities.**
As a reader of this portfolio, I can see that stated win/podium probabilities
match observed frequencies, so that I trust the numbers rather than merely
reading them.

**US-4 — Pre-race prediction.**
As a user, I can supply a starting grid for an upcoming race and receive ranked
predictions with honest uncertainty, so the system is usable and not merely an
offline experiment.

**US-5 — Legible findings.**
As a hiring manager skimming this repository, I can read the headline result and
its supporting evidence within two minutes, so the work's quality is apparent
without running anything.

---

## 3. Functional Requirements

### 3.1 Data Acquisition (FR-1xx)

| ID | Requirement |
|----|-------------|
| **FR-101** | The system SHALL collect race results, qualifying results, and session metadata for a configurable, contiguous range of seasons, targeting **at least eight seasons** per Constitution IV.2. |
| **FR-102** | The system SHALL persist a per-season checkpoint after each season completes, and SHALL NOT refetch any season for which a valid checkpoint exists. |
| **FR-103** | WHEN collection is interrupted, THE SYSTEM SHALL, on the next run, resume from the first incomplete season without refetching completed seasons. |
| **FR-104** | WHEN a single race session fails to load, THE SYSTEM SHALL record the failure, continue collecting remaining races, and report all failures in a summary at completion. |
| **FR-105** | The system SHALL record, for every collected row, the season, round, and race date, sufficient to establish strict chronological ordering. |
| **FR-106** | The system SHALL report collected row counts per season, and SHALL emit a warning WHERE a season's row count deviates from the expected race-count × entry-count by more than 10%. |
| **FR-107** | The system SHALL NOT commit collected race data to version control. |

### 3.2 Feature Engineering (FR-2xx)

| ID | Requirement |
|----|-------------|
| **FR-201** | Every feature SHALL declare, in machine-readable form, whether its inputs are available **pre-race** or only **post-race**. |
| **FR-202** | The primary feature set SHALL contain only features declared pre-race. |
| **FR-203** | Rolling, form, and aggregate features SHALL be computed exclusively from races strictly preceding the race they describe. |
| **FR-204** | The system SHALL provide an automated leakage check that fails IF any post-race-declared feature appears in the primary feature set. |
| **FR-205** | The system SHALL provide a leakage check that fails IF any rolling feature's value for a given race can be shown to depend on that race's outcome. |
| **FR-206** | WHERE weather is used, the primary feature set SHALL use only pre-race forecast or climatological priors; actual observed race weather is inadmissible per Constitution III.2. |
| **FR-207** | The system SHALL support an explicitly labelled **oracle feature set** including post-race features, usable only for upper-bound analysis. |
| **FR-208** | Feature engineering SHALL be deterministic: identical input data yields byte-identical output. |

### 3.3 Baselines (FR-3xx)

| ID | Requirement |
|----|-------------|
| **FR-301** | The system SHALL implement a **grid-order baseline** predicting finishing position equal to starting grid position. |
| **FR-302** | The system SHALL implement a **standings-order baseline** predicting finishing order by current championship standings. |
| **FR-303** | The system SHALL implement a **random-order control** establishing the random-chance floor. (Distinct from the *irreducible noise floor* of FR-754.) |
| **FR-304** | All baselines SHALL be evaluated by the identical code path and metrics as the model, on the identical holdout. |
| **FR-305** | Baselines SHALL be first-class citizens of the evaluation output, not a footnote. |

### 3.4 Model (FR-4xx)

| ID | Requirement |
|----|-------------|
| **FR-401** | The system SHALL train a model producing a ranking over drivers within each race. |
| **FR-402** | The system SHALL produce, per driver, a point prediction of finishing position and a prediction interval. |
| **FR-403** | Training SHALL use only races chronologically prior to the evaluation holdout. |
| **FR-404** | Every persisted model SHALL record training date range, row count, feature list and their availability declarations, hyperparameters, code commit, and evaluation metrics. |
| **FR-405** | Training SHALL be reproducible: identical data plus identical seed yields identical model predictions. |

### 3.5 Evaluation (FR-5xx)

| ID | Requirement |
|----|-------------|
| **FR-501** | Evaluation SHALL use a chronological split over a corpus of **≥8 seasons**: **≥5 seasons training, ≥3 seasons holdout** (Constitution IV.2). |
| **FR-501a** | IF fewer than 8 seasons are collected, THEN the shortfall SHALL be recorded in every results file and in the README, and the minima SHALL NOT be lowered to fit (Constitution IV.2a). |
| **FR-502** | Evaluation SHALL report, for the model and every baseline: MAE, exact-position accuracy, within-1/2/3 accuracy, Spearman correlation, top-3 and top-5 set accuracy, and podium-set accuracy. |
| **FR-503** | Every reported metric SHALL carry a bootstrap 95% confidence interval. |
| **FR-504** | Evaluation SHALL report the model-minus-grid-baseline delta per metric, with a confidence interval on the delta. |
| **FR-505** | Evaluation SHALL segment results by season and by circuit type, so that aggregate performance cannot conceal systematic failure. |
| **FR-506** | Evaluation SHALL emit a machine-readable results file alongside the human-readable report. |
| **FR-507** | IF the model does not beat the grid baseline on the primary metric with a confidence interval excluding zero, THEN the evaluation output SHALL state this explicitly and prominently. |

**Primary metric:** Spearman rank correlation between predicted and actual
finishing order, averaged per race. Chosen because the task is fundamentally
ordinal, and because position-difference metrics are dominated by DNF noise.

### 3.6 Race Simulation (FR-6xx)

| ID | Requirement |
|----|-------------|
| **FR-601** | The system SHALL convert model output into a Monte Carlo simulation over race outcomes, producing a full finishing-position distribution per driver. |
| **FR-602** | The simulation SHALL model, at minimum: pace uncertainty, stochastic retirement, and position-change dynamics conditioned on circuit overtaking difficulty. |
| **FR-603** | The simulation SHALL emit win, podium, and points-finish probabilities per driver. |
| **FR-604** | Simulation SHALL be deterministic given a seed. |
| **FR-605** | Per-driver probabilities across all finishing positions SHALL sum to 1.0 within a tolerance of 1e-6. |

### 3.7 Calibration (FR-7xx)

| ID | Requirement |
|----|-------------|
| **FR-701** | The system SHALL assess calibration of simulated probabilities against held-out race outcomes. |
| **FR-702** | Calibration SHALL be reported as a reliability curve with binned predicted-vs-observed frequencies, for win, podium, and points-finish events. |
| **FR-703** | The system SHALL report Brier score per event type, compared against a climatological baseline (base rate) and the grid-order baseline. |
| **FR-704** | The system SHALL report empirical coverage of stated prediction intervals; an interval labelled *X%* SHALL be reported against its observed containment rate. |
| **FR-705** | IF calibration is unmeasured or exceeds a documented error threshold, THEN probabilistic outputs SHALL still be displayed but SHALL carry an `UNCALIBRATED` label applied at the rendering layer, so no call site can omit it (Constitution V.4). |
| **FR-706** | Unmeasured or failing-calibration probabilities SHALL NOT support any performance claim in the README or results files (Constitution V.5). |

### 3.7a Race Randomness (FR-75x) — Constitution Article X

| ID | Requirement |
|----|-------------|
| **FR-751** | The system SHALL model retirement, safety-car incidence, pace variability, and position-change volatility as **distributions**, with parameters estimated from prior races only (X.1, X.3). |
| **FR-752** | No prediction SHALL be published as a bare point estimate; every predicted position SHALL carry a distribution or interval (X.2). |
| **FR-753** | Evaluation SHALL include at least one **proper scoring rule** (Brier or log score) that rewards correctly stated uncertainty, not rank accuracy alone (X.4). |
| **FR-754** | The system SHALL estimate and report the **irreducible noise floor**: the share of finishing-order variance no pre-race information can explain. |
| **FR-755** | Reported model error SHALL be interpreted against the noise floor rather than against zero, in both the evaluation report and the README (X.5). |

**On FR-754 method.** The floor is estimated empirically — e.g. by measuring how
much finishing order varies between races that are near-identical in all
pre-race respects (same circuit, near-identical grid and car order), and by
bounding the contribution of mechanical DNFs and first-lap incidents. The
estimate will be crude. A crude floor is still far more informative than
comparing model error to zero, which implicitly asserts that a perfect predictor
of a chaotic sport could exist.

### 3.8 Interface (FR-8xx)

| ID | Requirement |
|----|-------------|
| **FR-801** | The system SHALL expose a command-line interface covering: collect, build features, train, evaluate, simulate, and predict-from-grid. |
| **FR-802** | WHEN a required upstream artifact is missing, THE SYSTEM SHALL fail with a message naming the exact command that produces it. |
| **FR-803** | The predict-from-grid command SHALL accept a documented CSV schema and SHALL validate it, reporting all schema violations at once rather than failing on the first. |
| **FR-804** | All user-facing numeric output SHALL be rendered at a precision justified by its uncertainty. |

### 3.9 Documentation (FR-9xx)

| ID | Requirement |
|----|-------------|
| **FR-901** | The README SHALL state the headline result — model versus grid baseline — above any feature list. |
| **FR-902** | Every performance number in documentation SHALL be traceable to a committed results file. |
| **FR-903** | The README SHALL document the leakage defect found in v0 and the structural controls now preventing it. |
| **FR-904** | The README SHALL state the project's non-commercial, portfolio framing per Constitution I.2. |

---

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| **NFR-01** | A full evaluation run on cached data SHALL complete in under 5 minutes on a developer laptop. |
| **NFR-02** | The system SHALL run on Python 3.13 with the pinned dependency set. |
| **NFR-03** | Automated tests SHALL cover every leakage check (FR-204, FR-205) and every baseline (FR-301..303). |
| **NFR-04** | Test suite SHALL complete in under 60 seconds without network access. |
| **NFR-05** | No pipeline stage SHALL require network access except data collection. |

---

## 5. Out of Scope

Explicitly excluded. Each entry is a deliberate decision, not an oversight.

| Item | Reason |
|------|--------|
| Betting odds comparison / edge detection | Constitution I.2. Requires licensed odds data and lands in gambling regulation. |
| Live timing or in-race prediction | Large scope increase; does not serve the headline question. |
| Web UI or hosted service | Portfolio value is in the analysis, not the frontend. |
| Telemetry-derived features (long-run pace, tyre stints) | Genuinely predictive, but multiplies collection cost. Recorded as a future spec, not this one. |
| Driver-transfer and regulation-change modelling | Insufficient data per regime to learn from. |
| Synthetic-data demo mode | Demonstrates nothing about real performance and previously masked the absence of real data. |
| Per-race bespoke models | v0 shipped a `chinese_gp_predictor`; single-race models cannot be validated. |

---

## 6. Acceptance Criteria

This spec is satisfied when all of the following hold:

- [ ] **AC-1** A clean checkout can rebuild the dataset for **≥8 seasons** via documented commands, and an interrupted run resumes without refetching. Any shortfall is recorded, not accommodated. *(FR-101..107, FR-501a)*
- [ ] **AC-2** The automated leakage check passes, and a deliberately introduced post-race feature causes it to fail. *(FR-204, FR-205, NFR-03)*
- [ ] **AC-3** `evaluate` reports model and all three baselines on a **≥3-season holdout drawn from a ≥8-season corpus** with bootstrap CIs on every metric and on every delta. *(FR-501..506)*
- [ ] **AC-4** The headline model-versus-grid comparison is stated in the README with its confidence interval, in whichever direction it falls. *(FR-507, FR-901)*
- [ ] **AC-5** Reliability curves and Brier scores exist for win, podium, and points events, each against two baselines, and a proper scoring rule appears in the main evaluation. *(FR-701..703, FR-753)*
- [ ] **AC-6** Stated interval coverage is measured and reported. *(FR-704)*
- [ ] **AC-7** Every number in the README resolves to a committed results file. *(FR-902)*
- [ ] **AC-8** Full test suite passes offline in under 60 seconds. *(NFR-04, NFR-05)*
- [ ] **AC-9** The irreducible noise floor is estimated, committed to `results/`, and model error is reported relative to it. *(FR-754, FR-755)*
- [ ] **AC-10** No prediction anywhere in the system is emitted as a bare point estimate. *(FR-752)*

---

## 7. Open Questions

| ID | Question | Blocks | Resolution |
|----|----------|--------|------------|
| **OQ-1** | Which seasons form the holdout? Depends on how many collect cleanly. | FR-501 | Defer to after AC-1; target the three most recent complete seasons. |
| **OQ-2** | Is a pre-race weather forecast source obtainable for historical races? If not, FR-206 forces dropping weather from the primary set entirely. | FR-206 | Investigate during Phase 1; dropping weather is the acceptable fallback. |
| **OQ-3** | What is the documented calibration-error threshold for FR-705? | FR-705 | Set empirically once first reliability curves exist; must be fixed before any public claim. |
| **OQ-4** | Which estimator for the irreducible noise floor? Near-identical-race variance, DNF-bound decomposition, or both? | FR-754 | Decide in P5 with real data; both are cheap, so likely both with the wider bound reported. |
| **OQ-5** | Can 8 seasons actually be collected, given FastF1 coverage for 2018–19? | FR-101, FR-501a | Answered by T-106. If not, IV.2a governs: publish the shortfall. |
