# Plan 001 — Core Predictor

**Implements:** `spec.md` (Spec 001)
**Governed by:** `.specify/memory/constitution.md` v1.0.0
**Created:** 2026-08-15

> This document states **how** the spec will be built. Every design decision
> below cites the requirement that forces it. A decision citing no requirement
> is scope creep and must be removed or a requirement added.

---

## 1. Architecture Overview

```
                  ┌────────────────────────────────────────────┐
                  │  FastF1 API  (network, collection only)     │
                  └────────────────────┬───────────────────────┘
                                       │  FR-101..107
                          ┌────────────▼────────────┐
                          │  data/collect.py         │
                          │  resumable, checkpointed │
                          └────────────┬────────────┘
                                       │  season_YYYY.parquet
                          ┌────────────▼────────────┐
        FR-201..208       │  features/               │
        ┌─────────────────┤   registry.py  ← the     │
        │                 │   build.py       leakage │
        │                 │   checks.py      firewall│
        │                 └────────────┬────────────┘
        │                              │  features.parquet
        │                 ┌────────────▼────────────┐
        │                 │  models/                 │
        │   FR-301..305   │   baselines.py           │
        │   FR-401..405   │   ranker.py              │
        │                 └────────────┬────────────┘
        │                              │  predictions
        │              ┌───────────────┴───────────────┐
        │              │                               │
        │  ┌───────────▼──────────┐      ┌─────────────▼──────────┐
        │  │  evaluate/            │      │  simulate/              │
        │  │   metrics.py          │      │   montecarlo.py         │
        │  │   bootstrap.py        │      │   calibration.py        │
        │  │   report.py           │      │                         │
        │  │  FR-501..507          │      │  FR-601..605, FR-701..705│
        │  └───────────┬──────────┘      └─────────────┬──────────┘
        │              │                               │
        │              └───────────────┬───────────────┘
        │                              │
        └──────────────────► results/  ▼  (committed, FR-902)
                              cli.py  FR-801..804
```

**The load-bearing idea is the feature registry.** Leakage is prevented
structurally rather than by vigilance: a feature cannot enter the pipeline
without declaring when its inputs become available, and the primary feature set
is *derived* from that declaration rather than hand-maintained.

---

## 2. Target Module Layout

```
src/f1pred/
├── __init__.py
├── config.py              paths, seeds, season ranges
├── data/
│   ├── __init__.py
│   ├── schema.py          canonical row schema + validation      FR-105, FR-803
│   └── collect.py         resumable FastF1 collection            FR-101..107
├── features/
│   ├── __init__.py
│   ├── registry.py        @feature decorator + availability enum FR-201
│   ├── build.py           assembles frames from registry         FR-202,203,208
│   └── checks.py          automated leakage detection            FR-204,205
├── models/
│   ├── __init__.py
│   ├── baselines.py       grid / standings / random              FR-301..305
│   └── ranker.py          XGBoost ranker + interval regressor    FR-401..405
├── evaluate/
│   ├── __init__.py
│   ├── metrics.py         the metric battery                     FR-502
│   ├── bootstrap.py       resampling CIs on metrics and deltas   FR-503,504
│   └── report.py          rich tables + JSON results             FR-506,507
├── simulate/
│   ├── __init__.py
│   ├── montecarlo.py      race simulation                        FR-601..605
│   └── calibration.py     reliability, Brier, coverage           FR-701..705
├── reference/
│   ├── __init__.py
│   └── track_metadata.py  circuit priors                         (supports FR-602)
└── cli.py                 command surface                        FR-801..804

tests/                     mirrors src/f1pred/                    NFR-03,04
results/                   committed evaluation outputs           FR-506,902
specs/                     this spec
.specify/memory/           constitution
```

**Why a package rename (`src/` → `src/f1pred/`).** v0 used a bare namespace
package with no `__init__.py`, so `python -m src.cli` worked only from the repo
root. A named package is importable, testable, and installable. Cites NFR-02.

---

## 3. Key Design Decisions

### D-1 — Feature registry with availability declarations
**Forced by:** FR-201, FR-202, FR-204, Constitution III.5

Every feature is registered with an explicit availability class:

```python
class Availability(Enum):
    PRE_RACE  = "pre_race"    # admissible in primary model
    POST_RACE = "post_race"   # oracle variant only

@feature(Availability.PRE_RACE, depends_on=["grid_position"])
def grid_position_sq(df): ...
```

The primary feature set is computed as
`[f for f in registry if f.availability is PRE_RACE]`. It is never written by
hand, so it cannot drift out of sync with reality — the v0 failure mode.

**Rejected alternative:** a hand-maintained allowlist in `config.py`. That is
exactly what v0 had, and `weather_mixed` sat in it undetected for four months.

### D-2 — Two-layer leakage testing
**Forced by:** FR-204, FR-205, NFR-03

- *Static check:* assert no `POST_RACE` feature reaches the primary matrix.
- *Behavioural check:* for a sample of races, perturb the target race's outcome
  and assert every pre-race feature value is unchanged. This catches off-by-one
  errors in rolling windows that a static check cannot see.

The test suite includes a **negative control**: a deliberately leaky feature
that must cause the check to fail. A leakage test that has never failed is not
known to work.

### D-3 — Baselines share the model's code path
**Forced by:** FR-304, Constitution II.1

Baselines implement the same `Predictor` protocol as the model
(`predict(race_frame) -> ranked positions`). Evaluation is written once against
the protocol. It becomes impossible to evaluate the model on terms the baselines
did not face — which is how v0's baseline gap went unnoticed.

### D-4 — Chronological splitting as a first-class object
**Forced by:** FR-403, FR-501, Constitution IV.1

A `ChronologicalSplit` object owns the train/holdout boundary and exposes the
season list on each side. No function anywhere accepts a `test_size` float.
Random splitting is not merely discouraged; the API makes it unexpressible.

### D-5 — Bootstrap at race granularity
**Forced by:** FR-503, FR-504

Resampling resamples *races*, not driver-rows, because driver-rows within a race
are strongly dependent. Row-level bootstrap would produce dishonestly narrow
intervals. Deltas are computed within each bootstrap replicate, preserving the
pairing between model and baseline.

### D-6 — Weather handling pending OQ-2
**Forced by:** FR-206, Constitution III.2

Default position: actual race weather is `POST_RACE` and therefore excluded from
the primary set. It remains registered so the oracle variant (FR-207) can
quantify how much the v0 numbers were inflated — a genuinely interesting result
for the write-up.

### D-7 — Port the simulator, gate it on calibration
**Forced by:** FR-601..605, FR-705, Constitution V.4

v0's `race_simulator.py` logic is sound in structure and is ported largely
intact. What is added is the calibration harness and the display gate: until
calibration is measured and passes threshold, every probability the CLI prints
carries an `UNCALIBRATED` label. The gate is enforced in the render layer, so it
cannot be forgotten at a call site.

---

## 4. Technology

Versions audited 2026-08-16 against PyPI; this is the latest mutually-compatible
set on Python 3.13, and `pip check` passes.

| Concern | Choice | Notes |
|---------|--------|-------|
| Language | Python 3.13.5 | NFR-02. 3.14 exists but is not installed locally; 3.13 is fully supported and the gain does not justify a venv rebuild mid-project. |
| Collection | fastf1 3.8.3 | Cache retained at `data/fastf1_cache/`. **Caps `pandas<3.0.0`** — see D-8. |
| Data | pandas **2.3.3**, pyarrow 25.0.1, numpy 2.5.2 | pandas held at 2.x by fastf1's cap |
| Model | xgboost 3.4.1, scikit-learn 1.9.0 | ranker + interval regressor |
| Metrics | scipy 1.18.0, numpy | Spearman, bootstrap, proper scoring rules |
| Plots | matplotlib 3.11.1 | reliability curves (FR-702) |
| CLI | click 8.4.2 + rich 15.0.0 | v0's choice, works well |
| Tests | pytest 9.1.1 | absent from v0's venv — its tests were never runnable |

**Dependencies are pinned**, not floated. v0 used `>=` throughout and drifted to
pandas 3 / numpy 2.4 / xgboost 3.2. It happened to survive; that was luck, and
luck is not a dependency strategy.

### D-8 — pandas held at 2.x by fastf1
**Forced by:** NFR-02, Constitution VI.1, VI.4

fastf1 3.8.3 declares `pandas<3.0.0,>=2.1.1`. The v0 environment had pandas
3.0.2 installed over that cap, so `pip check` failed and the environment was
unsupported — undetected because the only code path exercising fastf1 heavily is
collection, which is precisely what never completed.

**Decision:** follow the resolver. pandas 2.3.3, everything else latest. pandas
2.3 covers every operation this spec needs (groupby, rolling, parquet, merge),
and a `pip check`-clean environment is worth more to a reproducibility-governed
project than a major version number.

**Revisit when** fastf1 lifts the cap. Do not resolve this by force-installing
pandas 3 — that recreates the exact state this decision exists to fix.

**Rejected alternative:** dropping fastf1 for the jolpica-f1 REST API. It would
free pandas, but invalidates the 365M local cache, loses session/telemetry
access (closing the door on future telemetry features), and requires rewriting
all of P1. Reasonable later; not now.

---

## 5. Disposition of v0 Code

Every existing file, with its fate and the justification. **Nothing is deleted
for being ugly — only for tracing to no requirement.**

> **Amendment A-1 (2026-08-15).** All v0 source files are removed from the
> working tree during P0, including those marked *Port* and *Rewrite* below.
> Their content is preserved at tag `v0-vibe-coded` and retrieved on demand:
>
> ```bash
> git show v0-vibe-coded:src/feature_engineer.py
> ```
>
> *Reason:* the P0 exit gate requires a tree containing only planned files,
> which conflicts with leaving v0 sources in place as porting material. Porting
> from the tag keeps the tree clean and forces each ported line to be
> deliberately reintroduced by a task, rather than surviving by inertia.

### Port (logic survives, adapted to new structure — retrieved from tag per A-1)

| File | Destination | Changes required |
|------|-------------|------------------|
| `src/race_simulator.py` | `simulate/montecarlo.py` | Add seed determinism (FR-604), probability-sum assertion (FR-605), calibration gate (FR-705) |
| `src/track_metadata.py` | `reference/track_metadata.py` | Largely as-is; supports FR-602 |
| `src/model.py` (training half) | `models/ranker.py` | Strip evaluation code out; add metadata recording (FR-404) |
| `src/model.py` (eval half) | `evaluate/metrics.py` | Rewrite against `Predictor` protocol (D-3); add bootstrap (FR-503) |
| `src/config.py` | `config.py` | Drop the hand-maintained `FEATURES` list — replaced by registry (D-1) |
| `data/reference/track_metadata.csv` | unchanged | Supports FR-602 |
| `data/input/2025_*.csv` (6 files) | `tests/fixtures/` | Reused as schema-validation fixtures (FR-803) |

### Rewrite (requirement cannot be met by patching)

| File | Reason |
|------|--------|
| `src/data_collector.py` | FR-102/103 resumability is architectural, not a patch. v0 wrote checkpoints to paths never created. |
| `src/feature_engineer.py` | FR-201 registry inverts the design. Feature logic is salvaged function-by-function during the port. |
| `src/cli.py` | New command surface (FR-801), actionable errors (FR-802), precision rules (FR-804) |
| `README.md` | FR-901..904. Current text makes claims the code cannot reproduce. |
| `requirements.txt` | Pinning + pytest |
| `tests/*` | Rewritten against new structure; leakage and baseline tests are new (NFR-03) |

### Delete

| Path | Justification |
|------|---------------|
| `demo.py` | Spec §5 — synthetic-data demo explicitly out of scope; masked the absence of real data |
| `src/manual_predict.py` | Duplicates `predict.py`; single predict path under FR-803 |
| `src/predict.py` | Superseded by CLI predict command + models layer |
| `models/chinese_gp_predictor*` (3 files) | Spec §5 — per-race bespoke models cannot be validated |
| `models/race_predictor*` (4 files) | Trained on leaked features (Constitution III.2) over one season. Unusable and actively misleading. |
| `plots/*.png` (3 files) | Generated from the above; regenerated by `evaluate` |
| `data/output/*.csv` (5 files) | Uncalibrated probabilities; violates Constitution V.4 |
| `data/input/2026_miami_grid_template.csv` | Header-only; Miami workflow out of scope |
| `docs/workflows/2026-miami-*.md` | Workflow out of scope per Spec §5 |
| `.vscode/tasks.json` | Sole task runs `demo.py` |
| `.qodo/`, `.sixth/` | Empty tool directories |
| `data/raw/*.parquet`, `data/processed/*.parquet` | 2022-only, built with leaked features. Regenerated by FR-101. Gitignored, so deletion is local only. |

**Retained on disk, untouched:** `data/fastf1_cache/` (365M). It is gitignored,
and it saves hours of refetching for 2022/2023 under FR-102.

---

## 6. Build Phases

| Phase | Delivers | Gate to exit |
|-------|----------|--------------|
| **P0 — Cull** | Clean tree, package skeleton | Tree contains only files this plan retains |
| **P1 — Data** | Resumable collection, ≥6 seasons | AC-1 |
| **P2 — Features** | Registry + leakage firewall | AC-2 (incl. negative control) |
| **P3 — Baselines & Eval** | Three baselines, bootstrap harness | AC-3 |
| **P4 — Model** | Trained ranker, honest comparison | AC-4 — *the headline answer lands here* |
| **P5 — Simulation** | Monte Carlo + calibration | AC-5, AC-6 |
| **P6 — Write-up** | README, results, polish | AC-7, AC-8 |

**P4 is the moment of truth.** If the model loses to the grid baseline, that is
the finding, Constitution II.4 requires publishing it, and P5–P6 proceed
unchanged — a well-engineered negative result is a stronger portfolio piece than
an unexamined positive claim.

---

## 7. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| FastF1 collection is slow (2 sessions × ~22 races × 6+ seasons) | P1 stalls, as in v0 | FR-102 checkpointing; 2022/23 already cached; run unattended |
| FastF1 API drift for older seasons (2018–19) | Missing seasons | FR-104 per-race failure tolerance; narrow the range if needed |
| Model genuinely does not beat baseline | "Failure" | Not a risk. It is a valid answer (Constitution I.4) and the write-up is designed for it |
| Simulator proves badly calibrated | P5 weakens | FR-705 labelling; miscalibration honestly reported is still a real finding |
| Scope creep toward telemetry features | Never finishing | Spec §5 excludes them; add as Spec 002 if ever |
