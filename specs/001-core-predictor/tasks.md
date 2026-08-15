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
| ☐ **T-101** | Define canonical row schema + validator in `data/schema.py` | FR-105, FR-803 | Schema rejects missing/mistyped columns; unit-tested |
| ☐ **T-102** | Implement per-season checkpoint write/read | FR-102 | `season_YYYY.parquet` written after each season; second run refetches nothing |
| ☐ **T-103** | Implement resume-from-incomplete logic | FR-103 | Test: kill mid-run, rerun, verify no completed season refetched |
| ☐ **T-104** | Per-race failure tolerance + failure summary | FR-104 | Injected session failure does not abort run; appears in summary |
| ☐ **T-105** | Row-count sanity warnings | FR-106 | Season with >10% row deviation emits warning |
| ☐ **T-106** | Run collection for 2018–2026 | FR-101 | **≥8 seasons** present; per-season counts reported; shortfall recorded if any (FR-501a) |
| ☐ **T-107** | Resolve **OQ-2** (historical weather forecast availability) | FR-206 | Documented answer in spec; OQ-2 closed |

**Exit gate:** AC-1. Also closes OQ-1 (holdout seasons become known).

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

**Exit gate:** AC-2. T-207 is the gate's gate — an untested test is not a control.

---

## Phase P3 — Baselines & Evaluation Harness

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-301** | Define `Predictor` protocol | D-3, FR-304 | Baselines and model are interchangeable in evaluation |
| ☐ **T-302** | Grid-order baseline | FR-301 | Implemented, unit-tested |
| ☐ **T-303** | Standings-order baseline | FR-302 | Implemented, unit-tested |
| ☐ **T-304** | Random-order control | FR-303 | Seeded, reproducible |
| ☐ **T-305** | `ChronologicalSplit` object | FR-403, FR-501, D-4 | No API anywhere accepts a random split |
| ☐ **T-306** | Metric battery | FR-502 | All eight metrics implemented and tested |
| ☐ **T-307** | Race-level bootstrap CIs | FR-503, D-5 | CIs on every metric; resamples races not rows |
| ☐ **T-308** | Paired delta CIs vs grid baseline | FR-504 | Delta CI computed within replicates |
| ☐ **T-309** | Segmentation by season and circuit type | FR-505 | Report breaks out both |
| ☐ **T-310** | JSON results emission | FR-506 | Machine-readable file written to `results/` |
| ☐ **T-311** | Explicit "does not beat baseline" verdict | FR-507 | Verdict line rendered prominently, both directions |

**Exit gate:** AC-3. **Baselines are evaluated here, before the model exists** —
so the bar is set before there is any temptation to move it.

---

## Phase P4 — Model

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☐ **T-401** | Port XGBoost ranker to `models/ranker.py` | FR-401 | Produces within-race ranking |
| ☐ **T-402** | Point prediction + interval | FR-402 | Both emitted per driver |
| ☐ **T-403** | Training respects chronological boundary | FR-403 | Uses `ChronologicalSplit`; test asserts no holdout row seen |
| ☐ **T-404** | Model metadata recording | FR-404 | All seven fields persisted incl. code commit |
| ☐ **T-405** | Reproducibility test | FR-405 | Same seed + data ⇒ identical predictions |
| ☐ **T-406** | **Run the headline evaluation** | AC-4 | Model vs grid baseline with delta CI, committed to `results/` |
| ☐ **T-407** | Oracle-variant comparison | FR-207 | Quantifies how much v0's leak inflated its numbers |

**Exit gate:** AC-4. The headline answer exists and is committed, whichever way
it falls.

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
| ☐ **T-509** | UNCALIBRATED render-layer label | FR-705, D-7 | Label applied in render layer; test proves no call site can omit it |
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
| ☐ **T-604** | Precision rules on numeric output | FR-804 | No probability printed to more digits than justified |
| ☐ **T-605** | README headline result first | FR-901 | Result above feature list |
| ☐ **T-606** | Every number traceable to `results/` | FR-902 | Each figure cites its file |
| ☐ **T-607** | Document the v0 leakage defect and its controls | FR-903 | Section exists; references T-205..207 |
| ☐ **T-608** | Non-commercial framing + data provenance | FR-904, VIII.1..3 | Stated in README |
| ☐ **T-609** | Offline test-suite timing check | NFR-04, NFR-05 | Full suite <60s, no network |

**Exit gate:** AC-7, AC-8. Spec 001 complete.

---

## Traceability Check

Every requirement maps to at least one task:

| Group | Requirements | Covered by |
|-------|-------------|------------|
| Data | FR-101..107 | T-101..106, T-005 |
| Features | FR-201..208 | T-201..209 |
| Baselines | FR-301..305 | T-301..304, T-310 |
| Model | FR-401..405 | T-401..405 |
| Evaluation | FR-501..507 | T-305..311, T-406 |
| Simulation | FR-601..605 | T-501..504 |
| Calibration | FR-701..706 | T-505..509 |
| Race randomness (Art. X) | FR-751..755 | T-510..514 |
| Interface | FR-801..804 | T-601..604 |
| Docs | FR-901..904 | T-605..608 |
| Non-functional | NFR-01..05 | T-004, T-207, T-609 |

**No orphan requirements. No orphan tasks.**

---

## Spec Amendments

*Log here whenever implementation forces a spec change — per Constitution VII.3,
the spec is amended before the code diverges.*

| Date | Task | Requirement | Change | Reason |
|------|------|-------------|--------|--------|
| 2026-08-15 | T-003 | plan §2 | **Plan amendment A-2**: added `pyproject.toml` to the retained tree | Plan §2 omitted a build file, but T-003 requires the package be importable and installable. Standard src-layout packaging. |
| 2026-08-15 | T-001 | plan §5 | **Plan amendment A-1**: v0 sources are deleted from the tree in P0 and ported from tag `v0-vibe-coded` on demand | Plan §5 implied Port/Rewrite files remain in the tree, contradicting P0's exit gate. Porting from the tag resolves it and enforces deliberate reintroduction. |
