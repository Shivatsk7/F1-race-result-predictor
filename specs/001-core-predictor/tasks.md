# Tasks 001 — Core Predictor

**Implements:** `spec.md` via `plan.md`
**Created:** 2026-08-15

Rules of this board, per Constitution VII:

1. A task with no requirement ID in **Satisfies** is out of scope. Delete it.
2. Tasks are done when their **Done when** clause is objectively true — not when
   the code "works".
3. If implementing a task shows the spec is wrong, **amend the spec first**,
   note it in Spec Amendments below, then continue.

Status: ☐ todo · ◐ in progress · ☑ done · ⊘ dropped

---

## Phase P0 — Cull

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☑ **T-001** | Delete files listed in plan §5 "Delete" | Spec §5 | `git status` shows only planned deletions; tree matches plan §5 |
| ☑ **T-002** | Move `data/input/2025_*.csv` → `tests/fixtures/` | FR-803 | Six fixtures relocated, referenced by no production code |
| ☑ **T-003** | Create `src/f1pred/` package skeleton per plan §2 | NFR-02 | All directories exist with `__init__.py`; `import f1pred` succeeds |
| ☑ **T-004** | Pin dependencies, add pytest | NFR-02, NFR-03 | `requirements.txt` fully pinned; `pytest` runs and collects 0 tests without error |
| ☑ **T-005** | Update `.gitignore` for `results/` policy and stale paths | FR-107, FR-902 | Collected data ignored; `results/` **tracked** |
| ☑ **T-006** | Commit the cull with rationale | VII.2 | Single commit referencing plan §5 |

**Exit gate:** repository contains only what plan §5 retains. Nothing else.

**P0 COMPLETE** (2026-08-15). Verified: `import f1pred` succeeds; `pytest`
collects cleanly; tree matches plan §5 + §2. Deleted 24 files across src, tests,
models, plots, data/output, docs, .vscode, .qodo, .sixth.

---

## Phase P1 — Data Acquisition

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☑ **T-099** | **Spike — pre-2018 results availability (OQ-7)** | OQ-7, FR-101 | **DONE 2026-08-24.** Answer: yes, back to 1996. Finding recorded in spec §7.1; spike deleted. Closed OQ-5, opened the OQ-6 corpus-start decision, and exposed six defects now closed as FR-001a, FR-003, FR-003a, FR-108, FR-110, FR-111. |
| ☐ **T-101** | Define canonical row schema + validator in `data/schema.py` | FR-002, FR-003, FR-105, FR-108, FR-803 | Schema rejects missing/mistyped columns; carries `status`, `laps_completed`, `classified_position`, `weekend_format`; unit-tested |
| ☐ **T-101a** | Implement the FR-001 target construction | FR-001, AC-15 | Finishers in official order, retirements by laps descending, DSQ last, ties broken by grid; unit-tested against a hand-built race with all four cases |
| ☐ **T-101b** | Circuit attribute table | FR-109 | Every circuit in the corpus has type and overtaking tier; a missing entry raises |
| ☐ **T-101c** | Validate source ordering against FR-001 | FR-001a | Every race's official order is checked against the FR-001 rule; disagreements are reported, not overridden. Regression test on Australia 2014 (8 retirements incl. a DSQ) and Australia 2016 |
| ☐ **T-101d** | Withdrawal vs. lap-1 retirement discrimination | FR-002, FR-003a | Withdrawn entrants excluded via the source's explicit marker; a zero-lap *retirement* is retained. Test asserts Australia 2016 keeps 21 starters and drops exactly 1 withdrawal |
| ☐ **T-102** | Implement per-season checkpoint write/read | FR-102 | `season_YYYY.parquet` written after each season; second run refetches nothing |
| ☐ **T-103** | Implement resume-from-incomplete logic | FR-103 | Test: kill mid-run, rerun, verify no completed season refetched |
| ☐ **T-104** | Per-race failure tolerance + failure summary | FR-104 | Injected session failure does not abort run; appears in summary |
| ☐ **T-105** | Row-count sanity warnings | FR-106 | Season with >10% row deviation emits warning |
| ☐ **T-105a** | Pagination completeness assertion | FR-110 | Collector pages to exhaustion and asserts against the API's reported total; a truncated season raises. Test: 2023 yields 6 sprint races, not 2 |
| ☐ **T-105b** | Transient-failure retry with backoff | FR-111 | Retries on timeout; exhausted retries surface via FR-104 and are never recorded as absent data |
| ☐ **T-100** | Phase-entry dependency check | NFR-06, NFR-07 | `pip check` exits 0; pinned set re-verified against the index; audit date updated in `requirements.txt`. **Repeats at the start of every phase.** |
| ☐ **T-106** | Run collection for the 8 completed seasons 2018–2025 | FR-101, FR-509 | **≥8 seasons** present; per-season counts reported; shortfall recorded if any (FR-501a). 2026 excluded — in progress. |
| ☐ **T-106a** | Resolve **OQ-5** and report margin | FR-101, FR-501a | Explicit statement of how many of the 8 collected cleanly, and whether the IV.2 floor is met |
| ☐ **T-107** | Resolve **OQ-2** (historical weather forecast availability) | FR-206 | Documented answer in spec; OQ-2 closed |
| ☐ **T-108** | Implement `CorpusWindow` resolution | FR-508, FR-509, FR-510, D-9 | Completed-season test implemented; `W` configurable; in-progress season provably cannot enter; unit-tested with a frozen clock either side of a season boundary |

**Exit gate:** AC-1, AC-11. Also closes OQ-1 (holdout seasons become known) and
OQ-5. **T-106 runs early** — it answers the project's highest-severity risk, and
nothing downstream should be built before its answer is known.

---

## Phase P2 — Feature Registry & Leakage Firewall

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-201** | Implement `@feature` decorator + `Availability` enum | FR-201 | Registry enumerates features with availability; unregistered feature cannot be built |
| ☐ **T-202** | Port v0 feature logic into registered functions | FR-202 | Each v0 feature ported or explicitly dropped with reason |
| ☐ **T-203** | Derive primary feature set from registry | FR-202 | No hand-written feature list exists anywhere in the codebase |
| ☐ **T-204** | Enforce strict-prior computation for rolling features | FR-203 | Rolling windows provably exclude the current race |
| ☐ **T-205** | Static leakage check | FR-204 | Fails if any POST_RACE feature reaches primary matrix |
| ☐ **T-206** | Behavioural leakage check (outcome perturbation) | FR-205 | Perturbing a race's result changes no pre-race feature for that race |
| ☐ **T-207** | **Negative control test** | NFR-03, D-2 | A deliberately leaky feature makes T-205 *and* T-206 fail |
| ☐ **T-208** | Determinism test for feature build | FR-208 | Two builds byte-identical |
| ☐ **T-209** | Oracle feature set (post-race included) | FR-207 | Selectable, labelled; cannot be used by default path |
| ☐ **T-210** | Regime table + `regime_id` / `seasons_since_regime_start` features | FR-210, D-10 | Five-row declared table; both features registered `PRE_RACE`; 2026 row present so the table needs no edit when it completes |
| ☐ **T-211** | Relative-feature constraint in the registry | FR-209, D-10 | Every historical-performance feature declares `relative=True` or carries a written justification; static check scans the field in the same pass as the leakage check |
| ☐ **T-212** | Sprint result as `PRE_RACE` feature | FR-211, FR-108 | Registered pre-race; explicit null on conventional weekends, never imputed; leakage checks pass |
| ☐ **T-213** | Cold-start fallbacks + `LOW_INFORMATION` flag | FR-212 | Declared fallbacks documented and not field-mean; rookie/new-team row produces a flagged prediction rather than a null or a crash |

**Exit gate:** AC-2. T-207 is the gate's gate — an untested test is not a control.

---

## Phase P3 — Baselines & Evaluation Harness

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-301** | Define `Predictor` protocol | D-3, FR-304 | Baselines and model are interchangeable in evaluation |
| ☐ **T-302** | Grid-order baseline | FR-301 | Implemented, unit-tested |
| ☐ **T-303** | Standings-order baseline | FR-302 | Implemented, unit-tested |
| ☐ **T-304** | Random-order control | FR-303 | Seeded, reproducible; mean over ≥1000 orderings with its own CI, never a single draw |
| ☐ **T-304a** | Pin the primary metric in code | FR-500 | One definition, imported everywhere; no call site re-specifies which metric gates a claim |
| ☐ **T-305** | `ChronologicalSplit` object, fed by `CorpusWindow` | FR-403, FR-501, D-4, D-9 | No API anywhere accepts a random split; split consumes a resolved window, never a literal season list |
| ☐ **T-306** | Metric battery | FR-502 | All **nine** metrics enumerated in FR-502 implemented and tested |
| ☐ **T-307** | Race-level bootstrap CIs | FR-503, D-5 | CIs on every metric; resamples races not rows |
| ☐ **T-308** | Paired delta CIs vs grid baseline | FR-504 | Delta CI computed within replicates |
| ☐ **T-309** | Segmentation by season, circuit type, overtaking tier | FR-505, FR-109 | Report breaks out all three |
| ☐ **T-310** | JSON results emission | FR-506, FR-510 | Machine-readable file written to `results/`, recording `W` and the resolved season list |
| ☐ **T-311** | Explicit "does not beat baseline" verdict | FR-507, FR-305 | Verdict line rendered prominently, both directions; baselines appear in the main report body alongside the model, never in a footnote or appendix |

**Exit gate:** AC-3. **Baselines are evaluated here, before the model exists** —
so the bar is set before there is any temptation to move it.

---

## Phase P4 — Model

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-401** | Port XGBoost ranker to `models/ranker.py` | FR-401 | Produces within-race ranking |
| ☐ **T-402** | Point prediction + interval | FR-402 | Both emitted per driver |
| ☐ **T-403** | Training respects chronological boundary | FR-403 | Uses `ChronologicalSplit`; test asserts no holdout row seen |
| ☐ **T-404** | Model metadata recording | FR-404 | All **six** fields enumerated in FR-404 persisted, incl. code commit |
| ☐ **T-405** | Reproducibility test | FR-405 | Same seed + data ⇒ identical predictions |
| ☐ **T-406** | **Run the headline evaluation** | AC-4 | Model vs grid baseline with delta CI, committed to `results/` |
| ☐ **T-407** | Oracle-variant comparison | FR-207 | Quantifies how much v0's leak inflated its numbers |
| ☐ **T-408** | Recency weighting with configurable decay | FR-406, D-10 | `sample_weight` applied; decay persisted as a hyperparameter; zero decay reproduces uniform weighting bit-for-bit |
| ☐ **T-409** | **Regulation-transfer experiment** | FR-511, FR-511a, FR-512, AC-12 | Train ≤2021 → test 2022 vs train ≤2020 → test 2021; penalty at two decay settings; committed to `results/` labelled `n=1` **and** `DIAGNOSTIC`, with actual training-season counts stated. Withdraw the FR-511a exemption if OQ-7 extended the corpus. |

**Exit gate:** AC-4, AC-12. The headline answer exists and is committed,
whichever way it falls — as does the measured cost of a regulation reset.

---

## Phase P5 — Simulation & Calibration

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-501** | Port simulator to `simulate/montecarlo.py` | FR-601, FR-602 | Full position distribution per driver |
| ☐ **T-502** | Seed determinism | FR-604 | Same seed ⇒ identical simulation |
| ☐ **T-503** | Probability-sum assertion | FR-605 | Sums to 1.0 ± 1e-6; test enforces |
| ☐ **T-504** | Emit win/podium/points probabilities | FR-603 | All three per driver |
| ☐ **T-505** | Reliability curves | FR-702 | Binned predicted vs observed, three event types |
| ☐ **T-506** | Brier scores vs two baselines | FR-703 | Against base rate and grid baseline |
| ☐ **T-507** | Interval coverage measurement | FR-704 | Stated vs observed containment reported |
| ☐ **T-508** | Resolve **OQ-3**, set calibration threshold | FR-705 | Threshold documented in spec |
| ☐ **T-509** | UNCALIBRATED render-layer label | FR-705, FR-706, D-7 | Label applied in render layer; test proves no call site can omit it; no README or results-file claim rests on an unlabelled probability |
| ☐ **T-510** | Stochastic parameters from prior races only | FR-751 | Retirement, safety car, pace variance, position volatility all distributional |
| ☐ **T-511** | Proper scoring rule in main evaluation | FR-753 | Brier or log score reported alongside rank metrics |
| ☐ **T-512** | Estimate irreducible noise floor | FR-754, OQ-4 | Floor committed to `results/`; method documented |
| ☐ **T-513** | Report model error relative to noise floor | FR-755 | Evaluation report and README both frame error against the floor |
| ☐ **T-514** | Assert no bare point estimates | FR-752 | Test fails if any output path emits a position without an interval |

**Exit gate:** AC-5, AC-6.

---

## Phase P6 — Interface & Write-up

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-601** | CLI command surface | FR-801 | Six commands present and functional |
| ☐ **T-602** | Actionable missing-artifact errors | FR-802 | Each error names the producing command |
| ☐ **T-603** | Grid CSV schema validation, all errors at once | FR-803 | Multi-error fixture reports every violation |
| ☐ **T-604** | Precision rules on numeric output | FR-804 | Probabilities ≤2 dp; no metric rendered to more dp than its CI half-width supports; both rules unit-tested |
| ☐ **T-605** | README headline result first | FR-901 | Result above feature list |
| ☐ **T-606** | Every number traceable to `results/` | FR-902 | Each figure cites its file |
| ☐ **T-607** | Document the v0 leakage defect and its controls | FR-903 | Section exists; references T-205..207 |
| ☐ **T-608** | Non-commercial framing + data provenance | FR-904, FR-905 | Both stated in README; all four provenance points of FR-905 present |
| ☐ **T-609** | Offline test-suite timing check | NFR-04, NFR-05 | Full suite <60s, no network |
| ☐ **T-610** | Record wall-clock timings against the reference machine | NFR-01 | Evaluation runtime measured and written into the results file; reference machine named in plan §4 |
| ☐ **T-611** | Within-regime caveat on the headline result | OQ-1, FR-901 | README states the holdout sits inside one regulation regime wherever the headline number appears |

**Exit gate:** AC-7, AC-8. Spec 001 complete.

---

## Traceability Check

Every requirement maps to at least one task:

| Group | Requirements | Covered by |
|-------|-------------|------------|
| **Target variable** | FR-001..003 | T-101, T-101a |
| Target variable | FR-001..003a | T-101a, T-101c, T-101d |
| Data | FR-101..111 | T-099, T-101, T-101b, T-102..106, T-105a, T-105b, T-106a, T-005 |
| Features | FR-201..212 | T-201..213 |
| Baselines | FR-301..305 | T-301..304, T-310, T-311 |
| Model | FR-401..406 | T-401..405, T-408 |
| Evaluation | FR-500..507 | T-304a, T-305..311, T-406 |
| Corpus window | FR-508..510 | T-108, T-305, T-310 |
| Regulation robustness | FR-209, FR-210, FR-406, FR-511, FR-511a, FR-512 | T-210, T-211, T-408, T-409 |
| Simulation | FR-601..605 | T-501..504 |
| Calibration | FR-701..706 | T-505..509 |
| Race randomness (Art. X) | FR-751..755 | T-510..514 |
| Interface | FR-801..804 | T-601..604 |
| Docs | FR-901..905 | T-605..608, T-611 |
| Non-functional | NFR-01..07 | T-004, T-100, T-207, T-609, T-610 |

**No orphan requirements. No orphan tasks.**

---

## Spec Amendments

*Log here whenever implementation forces a spec change — per Constitution VII.3,
the spec is amended before the code diverges.*

| Date | Task | Requirement | Change | Reason |
|------|------|-------------|--------|--------|
| 2026-08-20 | T-101a | spec §3.0 | **New §3.0, FR-001..003**: target variable defined as classified finishing position, retirements ordered by laps completed, DSQ last; `status`/`laps_completed`/`classified_position` retained | Adversarial review found the spec never defined what it predicts. Every metric, the noise floor, and the retirement model depended on an undefined target. Owner chose official-classification ordering. |
| 2026-08-20 | T-409 | FR-511a | **New FR-511a**: transfer experiment exempt from the IV.2 training minimum, labelled DIAGNOSTIC, barred from headline claims | FR-511 as written mandated 4- and 3-season training arms against FR-501's 5-season floor — a hard contradiction introduced the same day. Owner chose an explicit carve-out over dropping the experiment. |
| 2026-08-20 | T-101, T-212 | FR-108, FR-211 | **New FR-108, FR-211**: weekend format recorded; sprint results admitted as PRE_RACE | The corpus spans three weekend formats (none / sprint-sets-grid 2021–22 / sprint-standalone 2023+) and "qualifying result" silently meant different things across them. Sprint results are pre-race knowable and admissible. |
| 2026-08-20 | T-304a | FR-500 | **New FR-500**: primary metric given a requirement ID | FR-507 and FR-511 both gated on "the primary metric" while it was defined only in unnumbered prose — untraceable and ungateable. |
| 2026-08-20 | T-099 | OQ-7 | **New OQ-7 + spike T-099**: does FastF1 serve results data before 2018? | The 2018 floor may bind only on session/timing data. If results reach earlier, OQ-5's zero margin dissolves, FR-511 rises to n=3, and FR-511a can be withdrawn. Sequenced before T-106. |
| 2026-08-20 | T-101b, T-213, T-304, T-604, T-610, T-611 | FR-109, FR-212, FR-303, FR-804, NFR-01, OQ-1 | Unfalsifiable requirements made checkable; cold start defined; random control averaged over ≥1000 draws; new AC-13..16; user stories traced | Review flagged "developer laptop", "precision justified by its uncertainty", and undefined "circuit type" as VII.4 violations; §3.8 and most of §3.9 had no acceptance criterion; US-1..5 traced to nothing. |
| 2026-08-20 | T-210, T-211, T-408, T-409 | spec §3.2, §3.4, §3.5, §5 | **New FR-209, FR-210, FR-406, FR-511, FR-512**; new AC-12; new plan D-10; **§5 exclusion narrowed** | Owner: the model must be at least somewhat ready for regulation changes. Split into the impossible half (predicting an unraced regime — still excluded) and the tractable half (surviving one — now in scope via relative features, recency weighting, and a measured transfer penalty). Corpus contains one major reset (2022), so FR-511 is explicitly `n=1`. |
| 2026-08-20 | — | spec §3.5 | **New FR-508..510**: fixed-width rolling corpus window, completed seasons only, `W` configurable. New AC-11, new plan D-9. | Owner decision: the corpus should advance by one season at both ends as each season completes, rather than naming a fixed season list. Also settles whether the in-progress 2026 season is admissible — it is not. |
| 2026-08-20 | — | spec §3.6 | **FR-602 amended**: added safety-car incidence | FR-602 predated Article X and listed three stochastic factors where X.1 and FR-751 require four. Direct internal contradiction. |
| 2026-08-20 | — | spec §3.9 | **New FR-905**: data provenance and licensing in README | T-608 cited Constitution VIII.1..3 directly because no requirement implemented it — a task reaching past the spec, which VII.2 exists to prevent. |
| 2026-08-20 | T-100 | spec §4 | **NFR-06, NFR-07 given a task**; traceability table corrected to NFR-01..07 | Both were added during the 2026-08-16 dependency audit but never propagated to tasks.md, whose table still read NFR-01..05 while asserting "no orphan requirements". |
| 2026-08-20 | T-306, T-404 | FR-502, FR-404 | Requirement lists **explicitly enumerated**; task counts corrected 8→9 and 7→6 | Both "done when" clauses counted items that the requirements did not enumerate, so neither was objectively checkable — a VII.4 violation. |
| 2026-08-20 | — | §7 | **OQ-5 severity raised; new OQ-6** (regime-awareness and choice of `W`) | FastF1 coverage begins in 2018, making 2018–2025 exactly the eight-season IV.2 floor with zero margin. Recorded as the project's highest-severity risk rather than discovered during collection. |
| 2026-08-15 | T-003 | plan §2 | **Plan amendment A-2**: added `pyproject.toml` to the retained tree | Plan §2 omitted a build file, but T-003 requires the package be importable and installable. Standard src-layout packaging. |
| 2026-08-15 | T-001 | plan §5 | **Plan amendment A-1**: v0 sources are deleted from the tree in P0 and ported from tag `v0-vibe-coded` on demand | Plan §5 implied Port/Rewrite files remain in the tree, contradicting P0's exit gate. Porting from the tag resolves it and enforces deliberate reintroduction. |
| 2026-08-24 | T-099 | OQ-5, OQ-6, OQ-7, FR-001a, FR-003, FR-003a, FR-108, FR-110, FR-111 | **Spike T-099 executed.** OQ-7 and OQ-5 closed; OQ-6 reframed as the corpus-start decision; six defects closed; new §7.1 records the finding. | Results data reaches back to 1996, so the 2018 floor never bound the corpus. Probing the real payload also showed FR-002 was unimplementable, FR-003's status enum was far too coarse, "not finished" is not "not classified", pagination truncates silently, and the default timeout trips intermittently. |
