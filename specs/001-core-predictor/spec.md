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
> - [ ] §2 User stories and their traceability map
> - [ ] §3.0 **Target variable** (FR-001..003) — *what a DNF scores; read this first*
> - [ ] §3.1 Data acquisition (FR-101..109) — *incl. sprint weekend formats*
> - [ ] §3.2 Feature engineering (FR-201..212) — *the leakage rules; regime-relative features; cold start*
> - [ ] §3.3 Baselines (FR-301..305)
> - [ ] §3.4 Model (FR-401..406)
> - [ ] §3.5 Evaluation (FR-500..512) — *primary metric, rolling corpus window, regulation-transfer diagnostic and its IV.2 exemption*
> - [ ] §3.6–3.7 Simulation and calibration (FR-601..605, FR-701..706)
> - [ ] §3.7a **Race randomness** (FR-751..755) — *Article X: noise floor, proper scoring*
> - [ ] §3.8–3.9 Interface and documentation (FR-801..804, FR-901..905)
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

**Story traceability.** A user story that no requirement serves is decoration,
and a requirement that serves no story is scope creep. Every story below resolves
to requirements and to an acceptance criterion that proves it:

| Story | Served by | Proven by |
|-------|-----------|-----------|
| US-1 Reproducible dataset | FR-101..109, FR-508..510 | AC-1, AC-11 |
| US-2 Honest evaluation | FR-500..507, FR-301..305 | AC-3, AC-4 |
| US-3 Calibrated probabilities | FR-701..706, FR-751..755 | AC-5, AC-6, AC-9 |
| US-4 Pre-race prediction | FR-801..804, FR-212, FR-402 | **AC-13** |
| US-5 Legible findings | FR-901..905 | AC-7, **AC-14** |

---

## 3. Functional Requirements

### 3.0 Target Variable and Unit of Analysis (FR-0xx)

*This section defines what the system predicts. It comes first because every
metric in §3.5, the noise floor in FR-754, and the retirement model in FR-602
are meaningless without it.*

| ID | Requirement |
|----|-------------|
| **FR-001** | The prediction target SHALL be **classified finishing position**, constructed as: (1) classified finishers in official finishing order, then (2) retirements ordered by laps completed descending — a later retirement scores better than an earlier one — then (3) disqualified entrants last. Ties SHALL be broken deterministically by starting grid position. |
| **FR-002** | The unit of analysis SHALL be the **driver-race row**, for every driver who took the start. Entrants who did not start SHALL be excluded, and every exclusion SHALL be recorded with its reason. |
| **FR-003** | Every row SHALL retain `status` (FINISHED / RETIRED / DISQUALIFIED), `laps_completed`, and `classified_position` as separate stored fields, so that metrics can be recomputed under an alternative target definition **without recollecting data**. |

**Why this definition.** It matches how F1 itself classifies a race, keeps every
starter in every race so field size is constant, and gives retirement an ordered
position rather than a sentinel value that would fabricate ties. Its known cost:
a lap-1 collision scores near-worst even for a genuinely fast car. That cost is
accepted deliberately — it is *real* prediction error caused by *real* race
randomness, which is exactly what FR-754's noise floor exists to quantify.
FR-003 preserves the raw fields so this choice can be revisited empirically
rather than re-argued.

### 3.1 Data Acquisition (FR-1xx)

| ID | Requirement |
|----|-------------|
| **FR-101** | The system SHALL collect race results, qualifying results, and session metadata for a configurable, contiguous range of **completed** seasons, targeting **at least eight** per Constitution IV.2 and resolving to the window of FR-508. |
| **FR-102** | The system SHALL persist a per-season checkpoint after each season completes, and SHALL NOT refetch any season for which a valid checkpoint exists. |
| **FR-103** | WHEN collection is interrupted, THE SYSTEM SHALL, on the next run, resume from the first incomplete season without refetching completed seasons. |
| **FR-104** | WHEN a single race session fails to load, THE SYSTEM SHALL record the failure, continue collecting remaining races, and report all failures in a summary at completion. |
| **FR-105** | The system SHALL record, for every collected row, the season, round, and race date, sufficient to establish strict chronological ordering. |
| **FR-106** | The system SHALL report collected row counts per season, and SHALL emit a warning WHERE a season's row count deviates from the expected race-count × entry-count by more than 10%. |
| **FR-107** | The system SHALL NOT commit collected race data to version control. |
| **FR-108** | Every collected race SHALL record its **weekend format** as one of `CONVENTIONAL`, `SPRINT_SETS_GRID`, or `SPRINT_STANDALONE`. The system SHALL NOT treat "qualifying result" as a single concept across the corpus: the session that sets the Grand Prix grid differs by format, and the field that records the grid SHALL be the grid actually started from, never inferred from a qualifying classification. |
| **FR-109** | Circuit attributes used by FR-505 and FR-602 — at minimum **circuit type** (street / permanent / hybrid) and **overtaking-difficulty tier** — SHALL be maintained as declared data alongside the regime table of FR-210, not inferred at run time. Every circuit in the corpus SHALL have an entry, and a missing entry SHALL be an error rather than a default. |

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
| **FR-209** | WHERE a feature summarises historical performance — team pace, driver form, reliability — it SHALL be expressed **relative to the contemporaneous field** (rank, percentile, or within-season z-score) rather than as an absolute quantity, so that its meaning survives a regulation reset. A feature that cannot be expressed relatively SHALL declare that limitation in the registry, and the declaration SHALL be machine-readable. |
| **FR-210** | The system SHALL maintain a **declared table of regulation regimes** — start season and severity — and SHALL expose regime identity and seasons-elapsed-since-regime-start as `PRE_RACE` features. Regime boundaries SHALL be declared data, never inferred from the calendar: F1 regulation cycles are irregular and no fixed period is correct. |
| **FR-211** | WHERE a sprint race precedes the Grand Prix, its result SHALL be admissible as a `PRE_RACE` feature: it is knowable before lights out and therefore satisfies Constitution III.1. The feature SHALL be null-valued for conventional weekends, and the null SHALL be explicit rather than imputed with a neutral value that would misrepresent a sprint that never happened. |
| **FR-212** | WHERE a driver or constructor has no prior-race history in the corpus — a rookie, a new team, or the first race of a returning entrant — rolling features SHALL take **declared fallback values** rather than silently propagating nulls, and the affected prediction SHALL be flagged `LOW_INFORMATION` in all output. The fallback values SHALL be documented and SHALL NOT be the field mean, which would assert average competitiveness the system has no evidence for. |

### 3.3 Baselines (FR-3xx)

| ID | Requirement |
|----|-------------|
| **FR-301** | The system SHALL implement a **grid-order baseline** predicting finishing position equal to starting grid position. |
| **FR-302** | The system SHALL implement a **standings-order baseline** predicting finishing order by current championship standings. |
| **FR-303** | The system SHALL implement a **random-order control** establishing the random-chance floor, reported as the mean over **at least 1000 seeded random orderings** with its own confidence interval. A single draw SHALL NOT be reported: its variance is large enough to make the floor meaningless. (Distinct from the *irreducible noise floor* of FR-754.) |
| **FR-304** | All baselines SHALL be evaluated by the identical code path and metrics as the model, on the identical holdout. |
| **FR-305** | Baselines SHALL be first-class citizens of the evaluation output, not a footnote. |

### 3.4 Model (FR-4xx)

| ID | Requirement |
|----|-------------|
| **FR-401** | The system SHALL train a model producing a ranking over drivers within each race. |
| **FR-402** | The system SHALL produce, per driver, a point prediction of finishing position and a prediction interval. |
| **FR-403** | Training SHALL use only races chronologically prior to the evaluation holdout. |
| **FR-404** | Every persisted model SHALL record exactly these six fields: (1) training date range, (2) training row count, (3) feature list with each feature's availability declaration, (4) hyperparameters, (5) code commit, (6) evaluation metrics. |
| **FR-405** | Training SHALL be reproducible: identical data plus identical seed yields identical model predictions. |
| **FR-406** | Training SHALL support **recency weighting** of training rows via a configurable decay parameter, recorded among the hyperparameters of FR-404. Uniform weighting (zero decay) SHALL remain expressible, so the weighting's effect can be measured rather than assumed. |

### 3.5 Evaluation (FR-5xx)

| ID | Requirement |
|----|-------------|
| **FR-500** | The **primary metric** SHALL be Spearman rank correlation between predicted and actual classified finishing position (FR-001), averaged unweighted across races. Wherever this spec or the Constitution says "the primary metric", it means this one. Every requirement that gates a claim on model performance SHALL gate on this metric. |
| **FR-501** | Evaluation SHALL use a chronological split over the corpus window defined in FR-508, of **≥8 seasons**: **≥5 seasons training, ≥3 seasons holdout** (Constitution IV.2). |
| **FR-501a** | IF fewer than 8 seasons are collected, THEN the shortfall SHALL be recorded in every results file and in the README, and the minima SHALL NOT be lowered to fit (Constitution IV.2a). |
| **FR-502** | Evaluation SHALL report, for the model and every baseline, exactly these nine metrics: (1) MAE, (2) exact-position accuracy, (3) within-1 accuracy, (4) within-2 accuracy, (5) within-3 accuracy, (6) Spearman correlation, (7) top-3 set accuracy, (8) top-5 set accuracy, (9) podium-set accuracy. |
| **FR-503** | Every reported metric SHALL carry a bootstrap 95% confidence interval. |
| **FR-504** | Evaluation SHALL report the model-minus-grid-baseline delta per metric, with a confidence interval on the delta. |
| **FR-505** | Evaluation SHALL segment results by season, by circuit type, and by overtaking-difficulty tier as declared in FR-109, so that aggregate performance cannot conceal systematic failure. |
| **FR-506** | Evaluation SHALL emit a machine-readable results file alongside the human-readable report. |
| **FR-507** | IF the model does not beat the grid baseline on the primary metric with a confidence interval excluding zero, THEN the evaluation output SHALL state this explicitly and prominently. |
| **FR-508** | The corpus SHALL be a **fixed-width rolling window** of the `W` most recently completed seasons. WHEN a new season completes, the window SHALL advance by one season at **both** ends — the newly completed season enters, the oldest leaves — and all reported results SHALL be regenerated rather than carried forward. |
| **FR-509** | The window SHALL admit **completed seasons only**. An in-progress season SHALL NOT enter the training or evaluation corpus, and its partial row count SHALL NOT trigger the FR-106 deviation warning. An in-progress season MAY still be a *prediction target* under FR-801; being predictable and being admissible to the corpus are distinct roles. |
| **FR-510** | `W` SHALL be a configurable parameter with a documented default, never a constant embedded in code, and its value SHALL be recorded in every results file alongside the concrete season list the window resolved to. **`W` SHALL NOT resolve below 8** for any reported result; a configuration attempting it SHALL fail loudly rather than silently producing a sub-constitutional corpus. The sole exception is the FR-511 diagnostic. |
| **FR-511** | Evaluation SHALL include a **regulation-transfer experiment** quantifying what a regime change costs: the primary metric (FR-500) for a model trained on all seasons strictly prior to a reset season and evaluated on that reset season, compared against the identical protocol applied to **the season immediately preceding that reset**. The difference SHALL be reported as the **measured regime-transfer penalty**, with the number of transitions observed stated alongside it. |
| **FR-511a** | The FR-511 experiment is a **diagnostic and is exempt from the FR-501 training-season minimum**, because training strictly prior to a reset inside an 8-season corpus cannot reach 5 seasons. It SHALL report its actual training-season count, SHALL be labelled `DIAGNOSTIC` in `results/`, and SHALL NOT support any headline claim. Constitution IV.2 continues to govern every reported result; this requirement narrows where it binds, it does not weaken it. |
| **FR-512** | The regulation-transfer experiment SHALL be run at both zero and non-zero recency decay (FR-406), so that any claimed robustness benefit is measured rather than asserted. |

**Why Spearman is the primary metric (FR-500).** The task is fundamentally
ordinal — what matters is who finishes ahead of whom, not the arithmetic gap —
and position-difference metrics such as MAE are dominated by DNF noise under the
FR-001 target, where a lap-1 retirement lands near the back regardless of pace.
Spearman is reported alongside eight other metrics precisely so that a single
metric cannot be gamed, but only one metric can gate a claim, and this is it.

**On the rolling window (FR-508..510).** F1 relevance decays: 2018 cars, tyres,
and driver line-ups may actively mislead a prediction made years later. Dropping
the oldest season as a new one arrives is therefore a modelling decision, not
merely bookkeeping. But whether recency beats volume is an *empirical* question,
which is why FR-510 forbids hard-coding `W` — a fixed constant would foreclose
ever measuring the trade-off. Note two present-day consequences: with 2018 the
earliest season FastF1 covers, an 8-wide window is **pinned** until the 2026
season completes, and a window that always drops its oldest season never
accumulates margin above the Constitution IV.2 floor (see OQ-6).

**On regulation robustness (FR-209, FR-210, FR-406, FR-511, FR-512).** Two
different problems hide behind "handle regulation changes", and only one is
solvable. *Predicting a new regime* — what the 2026 cars will actually do — is
impossible: no data exists, and any claim to the contrary would be fabrication
of exactly the kind Article I exists to prevent. *Surviving a new regime* —
degrading gracefully, knowing you have degraded, and recovering quickly — is
tractable. This spec pursues only the second.

The mechanism is FR-209. At a reset, absolute pace is destroyed: a 2021
constructor's average finishing position says nothing about its 2022 car. But
relative structure persists, because "P3 on the grid finishes near P3" is a fact
about track position and overtaking difficulty, not about which car is quick.
A feature written as `team_avg_finish = 4.2` dies at a reset; the same feature
written as `team_rank_in_field = 2` survives it. This is a constraint on how
features are written, not additional machinery.

**Limitation, stated up front.** The 2018–2025 corpus contains exactly **one**
major regulation transition (2022, ground effect). FR-511 therefore rests on a
single observation. It SHALL be reported as one observation and never
generalised into a rate or a law. A single measured number with an honest `n=1`
attached is still worth far more than an unmeasured assurance that the model
"handles" regulation changes.

The experiment concretely resolves to:

```
reset arm       train 2018-2021 -> test 2022   (4 training seasons)
comparator arm  train 2018-2020 -> test 2021   (3 training seasons)
penalty         = comparator Spearman - reset Spearman
```

Both arms fall below Constitution IV.2's five-season training minimum, which is
unavoidable: no 8-season corpus can put five training seasons strictly before a
reset that sits in its middle. FR-511a grants the exemption explicitly and
confines it to a `DIAGNOSTIC`-labelled artifact rather than letting the conflict
be discovered and quietly rationalised during implementation. **If OQ-7 resolves
favourably and the corpus extends before 2018, this exemption becomes
unnecessary and SHALL be withdrawn** — the preferred outcome.

### 3.6 Race Simulation (FR-6xx)

| ID | Requirement |
|----|-------------|
| **FR-601** | The system SHALL convert model output into a Monte Carlo simulation over race outcomes, producing a full finishing-position distribution per driver. |
| **FR-602** | The simulation SHALL model, at minimum: pace uncertainty, stochastic retirement, **safety-car incidence**, and position-change dynamics conditioned on circuit overtaking difficulty. The four factors here SHALL match the four required by FR-751 and Constitution X.1. |
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
| **FR-804** | Numeric output precision SHALL follow two enforceable rules: (1) probabilities SHALL be rendered to at most **two decimal places**; (2) a metric accompanied by a confidence interval SHALL NOT be rendered to more decimal places than its CI half-width — a metric of 0.412 with a half-width of 0.03 renders as `0.41`, not `0.412`. v0 printed win probabilities to four decimals with no evidence any of them meant anything. |

### 3.9 Documentation (FR-9xx)

| ID | Requirement |
|----|-------------|
| **FR-901** | The README SHALL state the headline result — model versus grid baseline — above any feature list. |
| **FR-902** | Every performance number in documentation SHALL be traceable to a committed results file. |
| **FR-903** | The README SHALL document the leakage defect found in v0 and the structural controls now preventing it. |
| **FR-904** | The README SHALL state the project's non-commercial, portfolio framing per Constitution I.2. |
| **FR-905** | The README SHALL state the data's provenance and licensing position: that race data is obtained via FastF1 from Formula 1's timing feeds, that FastF1's MIT licence covers the software and not the data, that this repository ships the code to acquire data rather than the data itself, and that no Formula 1 trademark is used in a way implying affiliation or endorsement (Constitution VIII.1..3). |

---

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| **NFR-01** | A full evaluation run on cached data SHALL complete in under 5 minutes on the **reference machine declared in `plan.md` §4**, and the measured wall-clock time SHALL be recorded in the results file. "A developer laptop" is not a specification; a named machine plus a recorded measurement is. |
| **NFR-02** | The system SHALL run on Python 3.13 with the pinned dependency set. |
| **NFR-03** | Automated tests SHALL cover every leakage check (FR-204, FR-205) and every baseline (FR-301..303). |
| **NFR-04** | Test suite SHALL complete in under 60 seconds without network access. |
| **NFR-05** | No pipeline stage SHALL require network access except data collection. |
| **NFR-06** | The pinned dependency set SHALL be internally consistent: `pip check` SHALL pass. A declared constraint violated by an installed version is a defect, not a warning. |
| **NFR-07** | Dependency currency SHALL be re-verified against the index at the start of each phase, and the audit date recorded in `requirements.txt`. |

---

## 5. Out of Scope

Explicitly excluded. Each entry is a deliberate decision, not an oversight.

| Item | Reason |
|------|--------|
| Betting odds comparison / edge detection | Constitution I.2. Requires licensed odds data and lands in gambling regulation. |
| Live timing or in-race prediction | Large scope increase; does not serve the headline question. |
| Web UI or hosted service | Portfolio value is in the analysis, not the frontend. |
| Telemetry-derived features (long-run pace, tyre stints) | Genuinely predictive, but multiplies collection cost. Recorded as a future spec, not this one. |
| Driver-transfer modelling | Insufficient data per driver move to learn from. |
| Predicting the *effects* of an unseen regulation regime | No data exists for a regime that has not been raced. Forecasting what 2026's cars will do would be fabrication, not modelling. **Note the narrower boundary:** regime *robustness* — relative features, recency weighting, and measuring the transfer penalty — is **in scope** per FR-209, FR-210, FR-406, FR-511. What is excluded is learning regime-specific dynamics from a regime with no races in it. |
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
- [ ] **AC-11** The corpus window is resolved from completed seasons at run time, `W` is configurable, and every results file records both `W` and the concrete season list used. Advancing the system clock past a season boundary changes the window without a code edit. *(FR-508..510)*
- [ ] **AC-12** The regime table is declared data, every historical-performance feature is either regime-relative or declares why it cannot be, and the measured regime-transfer penalty for 2022 is committed to `results/` at two decay settings, labelled `n=1` and `DIAGNOSTIC`. *(FR-209, FR-210, FR-406, FR-511, FR-511a, FR-512)*
- [ ] **AC-13** `predict-from-grid` accepts a documented CSV for an upcoming race and returns ranked predictions with intervals; a malformed CSV reports every violation at once; a missing upstream artifact names the command that produces it; and an entry with no prior history is served a declared fallback and flagged `LOW_INFORMATION`. *(FR-801..804, FR-212, US-4)*
- [ ] **AC-14** The README states the v0 leakage defect and its controls, the non-commercial framing, and the full data-provenance position. *(FR-903, FR-904, FR-905, US-5)*
- [ ] **AC-15** The target variable is implemented per FR-001, and `status`, `laps_completed`, and `classified_position` are all retained, such that switching to an alternative target definition requires no recollection. *(FR-001..003)*
- [ ] **AC-16** Weekend format is recorded for every race, sprint results are admitted as a `PRE_RACE` feature with explicit nulls on conventional weekends, and every circuit has a declared type and overtaking tier. *(FR-108, FR-109, FR-211)*

---

## 7. Open Questions

| ID | Question | Blocks | Resolution |
|----|----------|--------|------------|
| **OQ-1** | Which seasons form the holdout? Depends on how many collect cleanly. **Note the consequence of the obvious answer:** a 2023–25 holdout sits entirely inside the post-2022 ground-effect regime, so the headline number measures *within-regime* performance and says nothing about behaviour across a reset — despite §3.5's regulation-robustness apparatus. That limitation SHALL be stated wherever the headline result appears. | FR-501, FR-901 | Defer to after AC-1; target the three most recent complete seasons, and state the within-regime caveat explicitly rather than letting a reader infer generality. |
| **OQ-2** | Is a pre-race weather forecast source obtainable for historical races? If not, FR-206 forces dropping weather from the primary set entirely. | FR-206 | Investigate during Phase 1; dropping weather is the acceptable fallback. |
| **OQ-3** | What is the documented calibration-error threshold for FR-705? | FR-705 | Set empirically once first reliability curves exist; must be fixed before any public claim. |
| **OQ-4** | Which estimator for the irreducible noise floor? Near-identical-race variance, DNF-bound decomposition, or both? | FR-754 | Decide in P5 with real data; both are cheap, so likely both with the wider bound reported. |
| **OQ-5** | Can 8 seasons actually be collected, given FastF1 coverage for 2018–19? **This is the project's highest-severity open risk**: FastF1 timing coverage begins in 2018, so 2018–2025 is *exactly* eight complete seasons and there is zero margin. A single failed season puts the project below its own Constitution IV.2 floor. | FR-101, FR-501a | Answered by T-106, which runs early in P1 precisely so the answer arrives before anything is built on it. If short, IV.2a governs: publish the shortfall, do not lower the bar. |
| **OQ-6** | Given zero margin (OQ-5) and F1's regulation resets — 2022 (ground effect) and 2026 (new power units, active aero) — should the corpus window be regime-aware, and should `W` default to 8 or wider? | FR-508, FR-510, §5 | Decide once T-106 reports what actually collected. Three options: (a) keep `W`=8 and accept a mixed-regime corpus, (b) widen `W` beyond 8 as seasons accrue to build margin, (c) treat 2022+ as a separate regime and accept a smaller, cleaner corpus with a published IV.2a shortfall. Option (c) is compatible with §5 as narrowed — §5 now excludes only *predicting an unraced regime*, not restricting the corpus to one — but it trades corpus size for regime purity and would likely trigger IV.2a. |
| **OQ-7** | **Does FastF1 serve race results, qualifying results, and standings for seasons before 2018?** The 2018 floor is believed to bind on session/timing/telemetry data, not on results data, which historically came via the Ergast-compatible backend and reaches back decades. If results data extends earlier, OQ-5's zero-margin problem dissolves, FR-511 rises from n=1 to n=3 (2014 hybrid, 2017 aero, 2022 ground effect), and FR-511a's IV.2 exemption can be withdrawn. **Counterweight:** Ergast was retired after 2024 and its successor is jolpica-f1, which plan D-8 rejected as a direct dependency — so the backend question must be answered, not assumed. | FR-101, FR-511, FR-511a, OQ-5, OQ-6 | Resolved by a spike (T-099) under Constitution VII.1a, run **before** T-106. Highest expected value per hour in the project: one afternoon either removes the project's largest risk or confirms the 8-season plan. |
