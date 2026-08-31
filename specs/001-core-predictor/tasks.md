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
| ☑ **T-101** | Define canonical row schema + validator in `data/schema.py` | FR-002, FR-003, FR-105, FR-108, FR-803 | Schema rejects missing/mistyped columns; carries `status`, `laps_completed`, `classified_position`, `weekend_format`; unit-tested — **DONE 2026-08-25.** `data/schema.py`: canonical row, dtype/null/duplicate-key validation, `ClassificationOutcome`, `WeekendFormat` mapping. 19 tests. |
| ☑ **T-101a** | Implement the FR-001 target construction | FR-001, FR-001b, AC-15 | Finishers in official order, retirements by laps descending, DSQ last; remaining ties by final-lap track position per FR-001b, grid only as a flagged fallback; unit-tested against a hand-built race with all four cases — **DONE 2026-08-25.** `data/target.py::compute_order` — all four cases tested on Australia 2014 and 2016. |
| ☑ **T-101b** | Circuit attribute table | FR-109 | Every circuit in the corpus has type and overtaking tier; a missing entry raises — **DONE 2026-08-26.** `reference/circuits.py`: all 35 circuits raced 2010–2025, enumerated from the source; type + overtaking tier declared; `get`/`assert_all_declared` raise on any undeclared id. |
| ☑ **T-101e** | Fetch final-lap position for tied races | FR-001b | Lap-times detail fetched **only** for races where `races_needing_final_lap_position` reports a tie; rows left to the grid fallback are flagged — **DONE 2026-08-30.** `collect.py::add_final_lap_positions`, reusing `ErgastClient`. 40 requests, one per tied lap rather than ~11 per race: the endpoint takes a `lap_number`. 81 of 83 resolvable rows filled, 0 fetch failures; grid fallback down 194 → 113. The remaining 111 are **zero-lap retirements the rule cannot reach** — new OQ-8, spec §7.3. |
| ☑ **T-101c** | Validate source ordering against FR-001 | FR-001a | Every race's official order is checked against the FR-001 rule; disagreements are reported, not overridden. Regression test on Australia 2014 (8 retirements incl. a DSQ) and Australia 2016 — **DONE 2026-08-25.** `validate_against_source` reports, never overrides. Australia 2014 agrees; Australia 2016 disagrees on the Gutierrez/Alonso lap-16 tie — see note below. |
| ☑ **T-101d** | Withdrawal vs. lap-1 retirement discrimination | FR-002, FR-003a | Withdrawn entrants excluded via the source's explicit marker; a zero-lap *retirement* is retained. Test asserts Australia 2016 keeps 21 starters and drops exactly 1 withdrawal — **DONE 2026-08-25.** Australia 2016 keeps 21 starters, drops exactly 1 withdrawal; zero-lap *retirements* retained. |
| ☑ **T-102** | Implement per-season checkpoint write/read | FR-102 | `season_YYYY.parquet` written after each season; second run refetches nothing — **DONE 2026-08-26.** Atomic parquet checkpoint per season (temp file + replace), so an interrupted write cannot masquerade as a checkpoint. |
| ☑ **T-103** | Implement resume-from-incomplete logic | FR-103 | Test: kill mid-run, rerun, verify no completed season refetched — **DONE 2026-08-26.** `seasons_to_collect` skips only seasons with a valid checkpoint; tested with a truncated `.partial` present. |
| ☑ **T-104** | Per-race failure tolerance + failure manifest | FR-104, FR-104a, FR-104b, FR-104c, AC-21 | Injected session failure does not abort run and appears in the manifest; a partially collected race is discarded whole; rolling features treat the gap as absent and retain their true race counts; exceeding the threshold aborts the run — **DONE 2026-08-26.** `FailureManifest`: reason recorded, run continues, gaps queryable per season, threshold breach aborts. |
| ☑ **T-105** | Row-count sanity warnings | FR-106 | Season with >10% row deviation emits warning — **DONE 2026-08-26.** `check_season_counts` against the season's modal field size — the median absorbs the very anomaly the check exists to catch. |
| ☑ **T-105a** | Pagination completeness assertion | FR-110 | Collector pages to exhaustion and asserts against the API's reported total; a truncated season raises. Test: 2023 yields 6 sprint races, not 2 — **DONE 2026-08-26.** `paginate` asserts against the reported total; stalled page, shifting total, and the 2023 six-sprint case all covered. |
| ☑ **T-105b** | Transient-failure retry with backoff | FR-111 | Retries on timeout; exhausted retries surface via FR-104 and are never recorded as absent data — **DONE 2026-08-26.** `with_retry` exponential backoff; exhausted retries raise and are recorded as failure, never as absent data. |
| ☑ **T-100** | Phase-entry dependency check | NFR-06, NFR-07 | `pip check` exits 0; pinned set re-verified against the index; audit date updated in `requirements.txt`. **Repeats at the start of every phase.** — **DONE 2026-08-25.** `pip check` exits 0; direct pins re-verified against the index; scipy 1.18.0→1.18.1; pandas held at 2.3.3 by fastf1's documented `<3.0.0` cap; audit date updated. **Re-run 2026-08-30 (P2 entry):** `pip check` exits 0; click 8.4.2→8.5.0; pandas 3.0.5 exists on the index and stays refused by fastf1's cap; audit date updated. |
| ☑ **T-106** | Run collection for the 16 completed seasons 2010–2025 | FR-101, FR-509 | **16 seasons** present (OQ-6 resolved: 2010 start); per-season counts reported; shortfall recorded if any (FR-501a). 2026 excluded — in progress. — **DONE 2026-08-30.** 6,911 rows over 329 races, 2010–2025. Zero failures in 329 attempts. `validate_frame` passes over all 16 seasons concatenated; no duplicate `(season, round, driver)` keys. Race counts reconcile against the historical calendar, including 2020's COVID-shortened 17. Counts in `results/collection_2010_2025.json`. |
| ☑ **T-106a** | Report the collected corpus against the IV.2 floor | FR-101, FR-501a, FR-902 | Explicit statement of how many of the **16** seasons collected cleanly, whether the Constitution IV.2 floor (8 total / 5 training / 3 holdout) is met **and by what margin**, and any shortfall stated plainly per IV.2a rather than absorbed. Written to a committed file in `results/`. — **DONE 2026-08-30.** All 16 collected cleanly. IV.2 met with **8 seasons of margin** (16 vs. floor of 8) and 6,911 rows vs. the 200-row floor. No shortfall, so IV.2a does not engage. `results/collection_2010_2025.json`. |
| ☑ **T-107** | Resolve **OQ-2** (historical weather forecast availability) | FR-206 | Documented answer in spec; OQ-2 closed — **DONE 2026-08-30.** Archived forecasts exist from 2017-01-01 only (9 of 16 seasons); boundary bisected, not assumed. ERA5 climatological priors cover the full corpus, verified at the 2010 opener. FR-206 admits a prior, so weather is **not** dropped. Spec §7.2. |
| ☑ **T-108** | Implement `CorpusWindow` resolution | FR-508, FR-509, FR-510, D-9 | Completed-season test implemented; `W` defaults to 16 and is configurable; resolves to 2010–2025 as of 2025; in-progress season provably cannot enter; unit-tested with a frozen clock either side of a season boundary — **DONE 2026-08-25.** `corpus.py::CorpusWindow`; W defaults to 16, resolves 2010–2025; frozen-clock tests either side of the 2026 finale; sub-floor width raises. |

**Exit gate:** AC-1, AC-11. Also closes OQ-1 (holdout seasons become known).
OQ-5 is **not** part of this gate: T-099 closed it on 2026-08-24, before
collection ran. **T-106 runs early** — it answered the project's
highest-severity risk, and nothing downstream should be built before its answer
is known.

---

## Phase P2 — Feature Registry & Leakage Firewall

| ID | Task | Satisfies | Done when |
|----|------|-----------|-----------|
| ☑ **T-201** | Implement `@feature` decorator + `Availability` enum | FR-201 | Registry enumerates features with availability; unregistered feature cannot be built — **DONE 2026-08-30.** `features/registry.py`: `@feature` decorator, `Availability`, named groups, and a `Registry` that refuses a duplicate name, an undocumented feature, or a name shadowing a raw schema column. `Registry.get` raises on an unregistered name, so an unregistered feature cannot be built. |
| ☑ **T-202** | Port v0 feature logic into registered functions | FR-202 | Each v0 feature ported or explicitly dropped with reason — **DONE 2026-08-30.** All 50 v0 features dispositioned in `features/disposition.py`: 33 ported, 17 dropped with a written reason. Tested both ways — every port must name a registered feature, every registered feature must descend from a v0 one or be listed as new. |
| ☑ **T-203** | Derive primary feature set from registry | FR-202 | No hand-written feature list exists anywhere in the codebase — **DONE 2026-08-30.** `build_primary` takes no feature list; the set is `registry.primary()`. A test scans every module outside the registry for feature names appearing as string literals and fails if any module names more than two. |
| ☑ **T-204** | Enforce strict-prior computation for rolling features | FR-203 | Rolling windows provably exclude the current race — **DONE 2026-08-30.** `features/priors.py`: every rolling quantity goes through `prior_stat`/`prior_rate`, which aggregate to the (entity, race) grain *before* shifting. That closes the teammate trap — a row-level shift inside a constructor group hands a driver their teammate's result from the race being predicted. Unsorted frames are refused rather than silently reading the future. |
| ☑ **T-205** | Static leakage check | FR-204 | Fails if any POST_RACE feature reaches primary matrix — **DONE 2026-08-30.** `checks.static_leakage_check`: fails on a post-race feature in the primary set, on a non-primary group reaching it, and on any *transitive* dependence on a post-race feature. Passes on the real 2010-2025 corpus. |
| ☑ **T-206** | Behavioural leakage check (outcome perturbation) | FR-205 | Perturbing a race's result changes no pre-race feature for that race — **DONE 2026-08-30.** `checks.behavioural_leakage_check`: permutes one race's outcomes among its own starters and asserts no pre-race feature moves for that race. Non-starters are held fixed, since moving a withdrawal changes which cars exist. Passes over 12 races spread across the corpus. |
| ☑ **T-207** | **Negative control test** | NFR-03, D-2 | A deliberately leaky feature makes T-205 *and* T-206 fail — **DONE 2026-08-30.** Two controls in `tests/test_leakage.py`. One feature declares PRE_RACE and depends on a post-race feature — both checks catch it. A second declares only raw-column inputs and adds the current race's points to an honest prior; the static check provably cannot see it and the behavioural check does. That second control is the written justification for keeping two layers. |
| ☑ **T-208** | Determinism test for feature build | FR-208 | Two builds byte-identical — **DONE 2026-08-30.** `checks.determinism_check` builds twice, the second time from a row-shuffled copy, and compares a content hash over values, dtypes, and column order. Column order follows registration order. |
| ☑ **T-209** | Oracle feature set (post-race included) | FR-207 | Selectable, labelled; cannot be used by default path — **DONE 2026-08-30.** `build_oracle` returns a matrix labelled `oracle`; `build_primary` has no keyword that could widen it, asserted by signature inspection. Two post-race features registered so the upper bound and the leakage checks both have something real to work with. |
| ☑ **T-210** | Regime table + `regime_id` / `seasons_since_regime_start` features | FR-210, D-10 | Five-row declared table; both features registered `PRE_RACE`; 2026 row present so the table needs no edit when it completes — **DONE 2026-08-30.** `regime_id`, `seasons_since_regime`, and `regime_is_major` registered PRE_RACE, reading `config.REGIMES`. The 2026 row is present and tested for, so the table needs no edit when that season completes. |
| ☑ **T-211** | Relative-feature constraint in the registry | FR-209, D-10 | Every historical-performance feature declares `relative=True` or carries a written justification; static check scans the field in the same pass as the leakage check — **DONE 2026-08-30.** `summarises_history` / `relative` / `relative_justification` on every declaration; the registry refuses an absolute historical feature at registration, and the static check rescans the field in the same pass as the leakage fields. Satisfied in practice by computing each historical quantity twice — absolute in a new INTERMEDIATE group that reaches no model, and as a within-race percentile that does. |
| ☑ **T-212** | Sprint result as `PRE_RACE` feature | FR-211, FR-108 | Registered pre-race; explicit null on conventional weekends, never imputed; leakage checks pass — **DONE 2026-08-30.** Two new sparse schema columns (`sprint_position`, `sprint_grid_position`) and a `sprint_result_pages` endpoint on the client, paged and asserted against the reported total per FR-110 — 2023 returns 2 of its 6 sprints unpaginated. 24 sprint weekends collected, 480 rows filled, and the results endpoint agreed with FR-108's declared weekend format in both directions: no declared sprint without results, no results for an undeclared round. Features `sprint_finish_pct` and `sprint_places_gained_pct` registered PRE_RACE. The null is enforced, not merely intended: a new `never_impute` declaration makes FR-212's cold-start fallbacks skip these features, so a rookie on a conventional weekend gets fallbacks for their form and nothing at all for a sprint that did not happen. Counts in `results/sprint_collection_2021_2025.json`. |
| ⊘ **T-213a** | Junior-formula supplement with hard date filter | FR-217, FR-218, AC-22 | Date-filtered admission; undated records discarded; source/licence/retrieval date in README; cached and version-pinned so features are byte-reproducible; leakage test fails on a record dated on or after the race — **DROPPED 2026-08-31, owner decision.** FR-217 says the system *may* supplement, so declining is compliant rather than a shortfall. Measured before deciding: the supplement could touch 100 of 6,868 rows (1.46%), and 27 of those are the 2010 opener where every entrant is a debutant because the corpus starts there, not because anyone was a rookie — so the real population is ~56 rows, under 1%. Wikidata's query service returns empty results for even trivial queries; Wikipedia is licensed and reachable but means scraping ~45 season pages of inconsistent tables. Against that, this is the one task in the phase with real leakage exposure, since junior records are written and revised after the fact and FR-218 would require proving every admitted record predates the race. Cost and risk both high, population under 1%. **Consequence: FR-217, FR-218 and AC-22 are deliberately unexercised — see the note under the P2 exit gate.** |
| ☑ **T-213** | Cold-start fallbacks + `LOW_INFORMATION` flag | FR-212 | Declared fallbacks documented and not field-mean; rookie/new-team row produces a flagged prediction rather than a null or a crash — **DONE 2026-08-30.** Fallbacks are *measured*, not invented: the mean finishing percentile actually achieved on debut, 0.35 for drivers and 0.33 for constructors, taken from 2011-2014 only so the constant cannot carry holdout information into a training feature. Field mean is 0.52 and is explicitly not used. Every affected row carries `low_information`; 100 rows in the corpus. |
| ◐ **T-214** | Telemetry collection for covered seasons | FR-213, FR-214 | Practice long-run pace and tyre-stint data collected for 2018+; per-season coverage reported; pre-2018 rows explicitly null, never imputed — **BLOCKED 2026-08-30** — needs a FastF1 practice-session collection pass over 2018+. No work started. |
| ◐ **T-215** | Telemetry feature group, excluded from primary set | FR-213, FR-215, AC-19 | Named group in the registry; leakage/registry test fails if any telemetry feature reaches the primary set — **BLOCKED 2026-08-30** — the `TELEMETRY` group exists in the registry and `build_group` assembles the separate arm; the features themselves wait on T-214. |
| ☑ **T-217** | Driver-transfer robustness | FR-216, AC-20 | Constructor rolling features reset on a move, driver features carry across, transfer recorded, next race flagged `POST_TRANSFER`; regression test on a known mid-corpus transfer — **DONE 2026-08-30.** Structural rather than patched: constructor priors group by `constructor_id`, so a move recomputes against the new team by construction, while driver priors group by `driver_id` and carry across. `detect_transfers` records all 90 mid-corpus moves; the first race after one is flagged `post_transfer`. Regression test asserts a mover and their new teammate see identical constructor history on the day of the move. |
| ☑ **T-221** | **Behavioural check must perturb race-level values, not only driver-level** | FR-205, NFR-03 | Perturbation moves race-level post-race quantities; a feature reading race-day weather while declaring itself pre-race fails the check. — **DONE 2026-08-31.** Found by T-218's negative control passing when it should have failed. The check perturbed a race by permuting outcomes *among its drivers*, which is correct for finishing positions but a no-op for anything the whole race shares: weather has one value per race, so shuffling it changed nothing and a race-level leak was invisible. Race-level columns are now *substituted* with a value from a different race, detected automatically rather than listed so a post-race column added later cannot miss out. Columns that decide whether a row exists (`classification_outcome`, `position_text`) are exempt: substituting "withdrawn" into a race deletes its whole field under FR-002. |
| ☑ **T-219** | **Shrinkage for thin estimates** | FR-212a, FR-203 | A feature's value is blended toward its declared prior in proportion to the evidence behind it; the counter is declared per feature; no-history rows return the prior exactly. — **DONE 2026-08-31, owner-approved.** `priors.shrink_towards`, a per-feature `evidence` declaration in the registry, and four new counters (season-scoped and circuit-scoped as well as career). Strength k=5, derived not tuned: a driver's finishing percentile after one race correlates 0.41 with where they settle over ten or more, so the observation earns weight r²≈0.17, giving 1/(1+k)=0.17. Replaces a cliff with a slope — the old code took a single race at full confidence. Reaches the 26% of rows whose season features rest on four races or fewer, against the 1.2% a cold-start fallback alone touches. Rank-scaled features are exempt and tested for. |
| ⊘ **T-220** | **Teammate anchoring for cold-start rows** | FR-212 | Rookie fallback anchored to the car rather than a flat constant. — **DROPPED 2026-08-31: it failed its own measurement.** The idea was that a rookie's car is known even when the driver is not. The first measurement supported it strongly — correlation 0.83 between a debutant's result and their teammate's. **That figure was itself leakage**: it used the teammate's finishing position *in the race being predicted*. Rebuilt on strictly pre-race data the correlation falls to 0.37, and mean absolute error against the actual result is **worse** than the flat constant (0.192 vs 0.167 on 56 debut rows). The anchor is also biased low, because teams that field rookies are usually scoring no points, so the team's points percentile understates where the rookie finishes. Decisive point: the team's standing is **already a feature in its own right**, so anchoring would duplicate an existing signal, add a bias, and destroy the 'no evidence' marker that the low-information flag depends on. |
| ☑ **T-218** | **Collect the climatological weather prior and register the pre-race weather features** | FR-206, FR-751a, AC-18 | ERA5 climatological priors collected for every race in the corpus; registered `PRE_RACE`; observed race weather registered `POST_RACE` and reachable only from the oracle set; the leakage check fails if a same-race observation reaches the primary matrix. — **DONE 2026-08-31.** ERA5 archive collected for all 35 circuits — 396,305 daily records, zero failures, ~47 min under a per-minute rate limit the client now waits out. Circuit coordinates added to the declared table, fetched from the same upstream source as the race results rather than typed by hand. The prior reads only years strictly before the race's own, a cut-off derived from the race date that no caller can widen; verified by rewriting a race day's rainfall to 999 mm and confirming its prior does not move, and by the mirror test that an *earlier* year does move it. Sepang tops the wet-probability prior at 0.87, which is the right answer. Three prior features registered PRE_RACE, two observed features POST_RACE. **The negative control found a hole in the firewall — see T-221.** |
**Exit gate:** AC-2. T-207 is the gate's gate — an untested test is not a control.

**P2 PARTIAL** (2026-08-31). The firewall, the feature port, the sprint result
and the practice-telemetry group are complete. On the real 2010-2025 corpus: 37
primary features over 6,868 rows, static and behavioural leakage clean over 12
perturbed races, two builds byte-identical. 296 tests pass.

**What remains, and why:**

* **T-218 (weather)** — a gap this board did not previously record. Newly added
  by the 2026-08-31 audit and awaiting owner sign-off before implementation.
* **T-214** — collection is running; the feature code and its tests are done.
* **T-213a (junior formula)** — ⊘ dropped by owner decision. FR-217 says *may*,
  so this is a declined option, not a shortfall. **FR-217, FR-218 and AC-22 are
  therefore deliberately unexercised.** They are recorded here rather than
  quietly dropped, because a requirement with no task and no explanation is
  indistinguishable from one nobody noticed. If rookie predictions turn out to
  be visibly poor once the model exists, this is the first thing to revisit.
* **T-216** — moved to P4. It was never completable here: measuring the
  telemetry arm's benefit needs a model and bootstrap CIs.

**AC-2 is therefore not yet claimed.** What is proven is that the firewall
works, including that it fails when it should.

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
| ☐ **T-216** | Telemetry benefit as a separate arm | FR-215, AC-19 | Primary vs primary+telemetry on telemetry-covered seasons only; delta with CI; reduced season count stated in the results file — **MOVED FROM P2 on 2026-08-31.** It cannot be done in P2: its own "done when" needs a trained model (T-401), paired bootstrap delta CIs (T-308) and the results emitter (T-310), none of which exist until P3/P4. Left in P2 it made that phase's exit gate unreachable by construction. The data and the feature group are ready; only the measurement waits. |
| ☐ **T-408** | Recency weighting with configurable decay | FR-406, D-10 | `sample_weight` applied; decay persisted as a hyperparameter; zero decay reproduces uniform weighting bit-for-bit |
| ☐ **T-409** | **Regulation-transfer experiment** | FR-511, FR-511a, FR-512, AC-12 | Run over the **full archive, not the FR-508 window**, for all three resets (2014, 2017, 2022) against their preceding-season comparators; penalty at two decay settings; each result carries its actual training-season count; 2014 labelled `DIAGNOSTIC`, headline stated as `n=2`. |

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
| ☐ **T-505** | Reliability curves | FR-701, FR-702 | Binned predicted vs observed, three event types |
| ☐ **T-506** | Brier scores vs two baselines | FR-703 | Against base rate and grid baseline |
| ☐ **T-507** | Interval coverage measurement | FR-704 | Stated vs observed containment reported |
| ☐ **T-508** | Resolve **OQ-3**, set calibration threshold | FR-705 | Threshold documented in spec |
| ☐ **T-509** | UNCALIBRATED render-layer label | FR-705, FR-706, D-7 | Label applied in render layer; test proves no call site can omit it; no README or results-file claim rests on an unlabelled probability |
| ☐ **T-510** | Stochastic parameters from prior races only | FR-751 | Retirement, safety car, pace variance, position volatility, **weather volatility** all distributional |
| ☐ **T-510a** | Separate weather's two roles | FR-751a, FR-206, AC-18 | Registry test: forecast/prior admissible pre-race, weather distribution used in simulation, observed weather reachable only from the oracle set |
| ☐ **T-511** | Proper scoring rule in main evaluation | FR-753 | Brier or log score reported alongside rank metrics |
| ☐ **T-512** | Estimate irreducible noise floor | FR-754, OQ-4, AC-9 | Floor committed to `results/`; method documented |
| ☐ **T-513** | Report model error relative to noise floor | FR-755, AC-9 | Evaluation report and README both frame error against the floor |
| ☐ **T-514** | Assert no bare point estimates | FR-752, AC-10 | Test fails if any output path emits a position without an interval |

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
| ☐ **T-607** | Document the v0 leakage defect and its controls | FR-903, AC-14 | Section exists; references T-205..207 |
| ☐ **T-608** | Non-commercial framing + data provenance | FR-904, FR-905, AC-14 | Both stated in README; all four provenance points of FR-905 present |
| ☐ **T-609** | Offline test-suite timing check | NFR-04, NFR-05 | Full suite <60s, no network |
| ☐ **T-610** | Record wall-clock timings against the reference machine | NFR-01 | Evaluation runtime measured and written into the results file; reference machine named in plan §4 |
| ☐ **T-611** | Within-regime caveat on the headline result | OQ-1, FR-901 | README states the holdout sits inside one regulation regime wherever the headline number appears |

**Exit gate:** AC-7, AC-8. Spec 001 complete.

---

## Traceability Check

Every requirement maps to at least one task:

| Group | Requirements | Covered by |
|-------|-------------|------------|
| **Target variable** | FR-001..003a | T-101, T-101a, T-101c, T-101d, T-101e |
| Data | FR-101..111 | T-099, T-101, T-101b, T-102..106, T-105a, T-105b, T-106a, T-005 |
| Features | FR-201..216 | T-201..215, T-217 |
| Features (optional, declined) | FR-217, FR-218 | ⊘ T-213a — deliberately unexercised, see P2 note |
| Telemetry benefit | FR-215 | T-216 (P4) |
| Baselines | FR-301..305 | T-301..304, T-310, T-311 |
| Model | FR-401..406 | T-401..405, T-408 |
| Evaluation | FR-500..507 | T-304a, T-305..311, T-406 |
| Corpus window | FR-508..510 | T-108, T-305, T-310 |
| Regulation robustness | FR-209, FR-210, FR-406, FR-511, FR-511a, FR-512 | T-210, T-211, T-408, T-409 |
| Simulation | FR-601..605 | T-501..504 |
| Calibration | FR-701..706 | T-505..509 |
| Calibration | FR-701..706 | T-505..509 |
| Race randomness (Art. X) | FR-751..755, FR-751a | T-510, T-510a, T-511..514 |
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
| 2026-08-25 | T-106, T-108, T-409 | OQ-6, FR-508, FR-510, FR-511, FR-511a, AC-12, D-9, D-10 | **OQ-6 resolved by the owner: corpus starts 2010.** `W` default 8 → 16. FR-511 runs over the full archive rather than the rolling window, covering 2014/2017/2022. FR-511a's blanket IV.2 exemption withdrawn. Regime table gains 2010 and 2011 rows. | 2010 is the first post-refuelling season, so every grid in the corpus was set on low fuel — necessary because §1's question is whether a model beats the grid, and 2006–09 Q3 ran on race fuel. Binding FR-511 to the window would have let the 2017 observation age out silently in 2028, so the experiment reads the archive instead. |
| 2026-08-25 | T-510, T-510a | FR-602, FR-751, FR-751a, AC-18 | **Weather added as a fifth randomness factor**; new FR-751a separates weather's feature role from its simulation role | Owner flagged that FR-751..755 never mentions weather. Article X's own rationale names "weather swings" as irreducible randomness, but X.1's enumerated minimum omits it and FR-751 and FR-602 inherited the omission. **Constitution X.1 needs the same amendment — owner action, not the agent's.** |
| 2026-08-25 | T-214..217 | §5, FR-213..216, AC-19, AC-20 | **Telemetry features and driver-transfer robustness moved into scope**; live/in-race prediction stays excluded | Owner: these improve accuracy. Telemetry does, but its source starts in 2018 while the corpus starts in 2010, so it is a separate arm on half the corpus rather than a primary feature. Transfers were excluded for the wrong reason — learning transfer *effects* needs data we lack, but not carrying a driver's old constructor form across a move is a correctness fix. Live prediction consumes in-race data the pre-race model cannot have, so it cannot move the headline metric; recorded as a candidate spec 002. |
| 2026-08-25 | T-104, T-213a | FR-104a..c, FR-217, FR-218, AC-21, AC-22, Constitution X.1 | **FR-104 failure handling made concrete**; **junior-formula supplement admitted under a hard date filter**; **Constitution amended to v1.3.0** adding weather to X.1 and new X.1a | FR-104 said "record, continue, report" but never said what happens to the data, whether rolling features see a gap, or when to stop. Owner approved scraping junior-formula history for rookies; the leakage risk is that such sources reflect post-race knowledge, so admission is date-filtered, undated records are discarded, and the cache is version-pinned to preserve FR-208 determinism. |
| 2026-08-30 | T-106a | tasks.md T-106a, P1 exit gate | **T-106a's text corrected before running it.** "Resolve **OQ-5** and report margin" → report the collected corpus against the IV.2 floor; "how many of the **8** collected cleanly" → **16**; the P1 exit gate no longer claims to close OQ-5. No requirement changed. | The row was written on 2026-08-20, when the corpus was expected to be the eight seasons 2018–2025 with zero margin. T-099 closed OQ-5 on 2026-08-24 (results reach back to 1996, so the eight-season ceiling was never real) and the owner set the corpus to 16 seasons from 2010 on 2026-08-25. Neither change propagated to this row, so the board asked an already-answered question and counted the wrong number of seasons — a VII.4 violation, since a reader could not tell from the task whether it was met. |
| 2026-08-30 | T-101e | **OQ-8**, FR-001b | **New OQ-8 and spec §7.3: FR-001b cannot separate zero-lap retirements.** No requirement changed; the gap is recorded and the owner decision left open. | Implementing the tie-break showed the rule is silent about its most common case. 111 of 194 tied rows (57%) are retirements with zero completed laps, and such a driver appears in no lap's timing data, so there is nothing to fetch — first-lap collisions are the usual cause. They fall to the grid fallback permanently rather than pending a fetch. Grid is arguably their *correct* ordering, being their last known track position, but FR-001b calls grid a last resort and flags the rows as weak, which understates what is known. Ordering is identical either way, so nothing is blocked; only the label and flag differ. |
| 2026-08-30 | T-107 | OQ-2, FR-206, FR-204, FR-205 | **OQ-2 closed, and its stated fallback withdrawn.** The question assumed that no historical forecast forces dropping weather from the primary set. FR-206 admits *either* a forecast *or* a climatological prior, and only the forecast is scarce. Primary set takes the prior. New spec §7.2. | Probing the live APIs showed archived forecasts begin 2017-01-01 — 9 of 16 seasons — while ERA5 priors cover the whole corpus. A 2017+ forecast would make primary-set coverage non-uniform, the same defect that made telemetry a separate arm under T-214..216. Two items recorded for the owner rather than decided: whether to build the forecast arm at all, and that the same ERA5 endpoint serves both the admissible prior and inadmissible observed weather separated only by date range — FR-204/FR-205 should treat a same-race read as a hard failure. |
| 2026-08-25 | T-101a, T-101e | FR-001, **FR-001b**, AC-15 | **FR-001's tie-break replaced.** Retirements tied on laps completed are now ordered by track position on their final completed lap; grid position is a flagged last resort. | T-101c's validator disagreed with the official order on Australia 2016, where Gutiérrez and Alonso collided and retired on the same lap — laps completed cannot separate them and grid position reverses the official result. A probe confirmed the lap-times endpoint carries per-lap position back to at least 2005 and gives P12/P13, reproducing the official order exactly. Owner chose the finer signal. Fetched lazily (~11 requests/race) only where a tie exists. |
| 2026-08-30 | T-212 | FR-211, FR-108, T-101 schema | **Canonical row schema extended by two sparse columns** for the sprint result. No requirement changed — FR-211 already required the sprint result as a feature; T-101's schema simply did not carry it. | The sprint is a separate endpoint with its own payload key, so the columns could not be filled by the race pass. They are nullable and null on 305 of 329 weekends, which is why they are exempt from the non-null rule. All 16 season checkpoints were rewritten in place rather than recollected, following the T-101e precedent. One genuine gap the extension exposed: a null in these columns means "no sprint ran" only *after* the sprint pass has run — before it, the same null means "not fetched yet". The two are indistinguishable in the data, so the distinction lives in the collection order and in `SprintReport`, not in the frame. |
| 2026-08-31 | T-213a, T-216, T-218, T-505, T-210 | FR-206, FR-217, FR-218, FR-701, FR-751a, AC-16, AC-17, AC-18, AC-22 | **Board audit.** Five defects found by checking the spec against the board mechanically rather than by eye, and fixed: (1) **T-213a dropped** by owner decision, with FR-217/FR-218/AC-22 recorded as deliberately unexercised rather than left orphaned; (2) **T-216 moved to P4** — its "done when" needs a model and bootstrap CIs, so sitting in P2 it made that phase's exit gate unreachable; (3) **new T-218** — no task collected weather at all, leaving FR-206 unimplemented and T-510a/AC-18 with nothing to test; (4) **FR-701 had no task** — added to T-505, which is what actually assesses calibration against held-out outcomes; (5) **AC-16 and AC-17 were already satisfied but unticked**, and five acceptance criteria were cited nowhere on the board. | The traceability section claimed every requirement maps to a task, and it had drifted: 97 requirements, 89 tasks, and no mechanical check that the two agreed. Reading a 250-line board by eye does not catch an orphaned requirement — a script comparing the spec's defined IDs against the board's Satisfies column catches all of them in one pass. Also renamed the `seasons_since_regime` feature to `seasons_since_regime_start` to match FR-210 and the config helper; the board and the code had drifted apart, which is precisely the kind of gap that makes a "done when" unverifiable. |
| 2026-08-31 | T-219, T-220, T-218 | **FR-212a (new)**, FR-206, FR-212, Constitution IX.5 | **Shrinkage added as FR-212a; teammate anchoring proposed, measured, and rejected; the weather task approved.** Spec §3.2 is returned to Draft for the owner's read of FR-212a's text — the owner approved the change in conversation, but IX.2 says approving a summary is not ratification. | The cold-start fallback only ever addressed rows with *no* history — 1.2% of the corpus. Measurement showed the real problem is one step along: a driver's form after a single race correlates just 0.41 with where they settle, and 26% of rows carry season-scoped features built on four races or fewer, because those reset every January. The pipeline had a cliff there and now has a slope. **T-220 is the more useful entry.** Anchoring rookies to their car looked excellent — correlation 0.83 — until the number was checked, at which point it turned out to be measuring the teammate's finish *in the race being predicted*. This is the project's own stated failure mode reappearing in the analysis used to justify a feature rather than in the feature itself, and the leakage firewall would not have caught it, because no feature was built. Recorded because the next person to have this idea should find the measurement, not repeat it. |
| 2026-08-31 | T-218, T-221 | FR-205, FR-206, NFR-03 | **A hole in the behavioural leakage check, found by the control that was supposed to be routine.** The check permuted post-race values among a race's drivers. That is right for anything that varies per car and useless for anything the race shares — so observed weather, which has one value per race, could be read by a feature declaring itself pre-race and neither layer would object. | The static layer cannot see it (the feature declares only a raw column) and the behavioural layer's perturbation was a no-op on it. Every race-level post-race quantity had the same exposure, not just weather. Worth recording because the control was written expecting a green tick: the check had been passing for eleven days against a class of leak it could not detect. Also added `absent_sources` to the built matrix — a frame built without weather or telemetry now says so, since a matrix whose weather features are entirely null is a different experiment from one where they are present. |
