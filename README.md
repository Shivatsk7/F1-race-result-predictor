# Race Result Predictor

> **Status: Spec 001 complete.** All six phases delivered — corpus, feature
> registry and leakage firewall, baselines, model, simulation and calibration,
> and the command-line surface.
>
> **Governing documents** (note: `.specify/` is a hidden directory — `Cmd+Shift+.`
> in Finder, or use the links):
> [constitution](.specify/memory/constitution.md) ·
> [spec](specs/001-core-predictor/spec.md) ·
> [plan](specs/001-core-predictor/plan.md) ·
> [tasks](specs/001-core-predictor/tasks.md)
>
> Every figure on this page resolves to a committed file in `results/`, under
> [Constitution I.3](.specify/memory/constitution.md). The previous version of
> this README claimed numbers the code could not reproduce.

---

## The question

Formula 1 finishing order is largely set by qualifying. So any race-result model
is competing against a signal anyone can read off the grid sheet for free.

**Can a model that knows only pre-race information predict finishing order
measurably better than the starting grid alone — and can it state its
uncertainty in probabilities that are actually calibrated?**

A negative answer is a valid result and will be published as one.

## The headline result

**The model does not beat the starting grid.** *(AC-4, FR-507; source:
[`results/model_vs_baselines_2023_2025.json`](results/model_vs_baselines_2023_2025.json))*

Held out: the 70 races of 2023-2025, never seen in training. Trained on the 13
seasons 2010-2022. The metric is Spearman rank correlation between predicted
and actual finishing order — 1.0 is a perfect ordering, 0.0 is a coin toss.

| predictor | Spearman | 95% CI |
|---|---|---|
| **model** (XGBoost ranker) | **0.66** | 0.61 to 0.71 |
| grid order | 0.66 | 0.60 to 0.71 |
| championship-standings order | 0.61 | 0.56 to 0.66 |
| random order (1000 seeded draws) | 0.00 | -0.05 to 0.06 |

**Model minus grid: +0.00, 95% CI -0.03 to +0.04.** The interval contains zero,
so the difference is not distinguishable from noise. Forty features, a
16-season corpus and a trained ranker land in the same place as reading the
grid sheet.

The two decimal places are the result, not a rounding of it. Each score's
interval is about ±0.05 wide, and FR-804 forbids printing a metric to more
decimals than its interval supports — so at the precision this evidence
actually carries, the model and the grid are **the same number**. The
underlying figures (model 0.6599, grid 0.6555, difference +0.0044) are in the
results file, and stating them to four decimals in prose would imply a
resolution the bootstrap did not find.

Three things this does *not* say. It is not a claim that the model is worthless
— it is far above the random floor, so it has learned something real; it has
learned roughly what the grid already knew. It is not a claim that no model
could do better. And it is measured **inside a single regulation regime** —
see the caveat below.

The random control is the check on the harness itself: a metric that flattered
random orderings would be measuring something other than skill.

### The within-regime caveat

The 2023-2025 holdout sits **entirely inside one regulation regime** — the
ground-effect rules introduced in 2022 *(OQ-1, FR-210; regime table in
[`config.py`](src/f1pred/config.py))*. The headline number therefore says how
the model does against the grid when the cars are broadly stable, and says
nothing about a season following a rule reset.

That question was measured separately and is reported below, on two compliant
transitions. It is a smaller, weaker result than the headline, and the two must
not be read as one.

### How close is this to the ceiling?

Some of the remaining error is not the model's to remove. Measuring the gap
between *teammates* — identical car, identical circuit, identical day, starting
within three grid slots of each other — gives an irreducible spread of **2.79
positions** across 1,540 such pairs. The model's mean absolute error is **3.37
positions**, so it sits **0.58 positions** above a floor set by chance rather
than by prediction *(FR-754, FR-755; source:
[`results/noise_floor.json`](results/noise_floor.json))*.

Read against that floor rather than against zero, much of the distance between
"beats the grid" and "does not" may be asking a chaotic sport for precision it
does not contain.

## Are the probabilities trustworthy?

Partly, and the two halves have different answers *(FR-701..706; source:
[`results/calibration_2023_2025.json`](results/calibration_2023_2025.json))*.

| output | measured | verdict |
|---|---|---|
| **position intervals** (stated 80%) | contain the truth **81%** of the time (79% to 84%) | **CALIBRATED** |
| win probability | worst reliability gap **0.19** | **UNCALIBRATED** |
| podium probability | worst reliability gap **0.16** | **UNCALIBRATED** |
| points probability | worst reliability gap **0.16** | **UNCALIBRATED** |

The documented threshold is ten percentage points, fixed *before* the first
reliability curve was read rather than after *(OQ-3, FR-705)*. So an interval
this project prints means what it says; an event probability does not yet, and
carries an `UNCALIBRATED` label applied at the rendering layer where no call
site can omit it. Under FR-706 those three figures support no claim anywhere in
this repository.

Brier scores put the model level with the grid baseline on all three events and
well ahead of the base-rate baseline — the same shape as the headline result.

## The three supporting arms

Each was run because it might have changed the headline. None did.

| question | answer | source |
|---|---|---|
| Does practice telemetry help? | **+0.02**, interval contains zero. Available for only 8 of 16 seasons, so it is a separate arm rather than a primary feature (FR-214) | [`telemetry_arm_2023_2025.json`](results/telemetry_arm_2023_2025.json) |
| Does the model survive a regulation reset? | No compliant transition's penalty excludes zero (n=2: 2017, 2022; 2014 is `DIAGNOSTIC`). Recency weighting *raised* the penalty on every transition | [`regime_transfer.json`](results/regime_transfer.json) |
| What would knowing the answer be worth? | **+0.23** over the grid — the oracle arm, which reads post-race features and is **INADMISSIBLE** as a forecast (FR-207) | [`oracle_vs_primary_2023_2025.json`](results/oracle_vs_primary_2023_2025.json) |

The regulation-transfer arm contains the result most likely to be misread. The
model's raw penalty across the 2022 reset is +0.07, which looks like a failure
to transfer — until the grid baseline turns out to have lost +0.11 across the
same boundary. Differenced within each arm, where both predictors faced the
same races, the model transfers *better* than the grid across the biggest reset
in the corpus. The raw number alone would have been published as a weakness the
data does not support.

## Why this was rebuilt

The v0 system (preserved at tag `v0-vibe-coded`) ran end to end and looked
convincing. An audit found otherwise:

| Metric, held-out races | v0 model | Grid-order baseline |
|---|---|---|
| MAE | 3.675 | **3.550** |
| Exact position | 15.0% | **17.5%** |
| Within 2 positions | 42.5% | **55.0%** |
| Top-3 hit rate | 50.0% | **66.7%** |
| Spearman | **0.621** | 0.569 |

*(Source: [`results/v0_audit.json`](results/v0_audit.json). These describe code
that no longer exists in this tree — v0's sources were deleted in P0 and survive
only at the tag — so they are a record of a past measurement rather than an
output of the current pipeline. That is stated in the file itself. They are the
only numbers on this page of that kind.)*

It lost to a one-line baseline on five of six metrics. Two root causes:

1. **Temporal leakage.** `weather_mixed` was the third most important feature
   and was computed from the *actual weather of the race being predicted*.
   Every reported number was inflated by knowing whether it rained.
2. **A meaningless holdout.** Two races, forty rows. Every metric was
   indistinguishable from noise.

Neither was a coding error. Both were process failures — nothing in the project
made them detectable. The rebuild fixes the process, not just the code.

## How the rebuild prevents recurrence

| v0 failure | Structural control |
|---|---|
| Leaky feature sat undetected | Features declare input availability; the primary set is *derived*, not hand-listed (FR-201, D-1) |
| — the static half | Every feature's declared dependencies are walked transitively; one reading a post-race column fails the build (T-205) |
| — the behavioural half | Post-race values are permuted between drivers, and again *between races*, and any feature whose output moves is caught (T-206, T-221) |
| — and it is not optional | The firewall runs in `f1pred features` by default and in CI; a control you must remember to switch on is the control v0 lacked (T-207, NFR-03) |
| Baseline never computed | Baselines share the model's evaluation code path and are mandatory output (FR-304, Constitution II) |
| Forty-row holdout | Minimum three-season chronological holdout, bootstrap CIs on every metric (FR-501, FR-503) |
| Uncalibrated probabilities shown | Reliability curves + Brier scores required; uncalibrated output is labelled at the render layer (FR-701, D-7) |
| Collection died and lost 3 seasons | Per-season checkpointing; interrupted runs resume (FR-102, FR-103) |
| README claimed unreproducible numbers | Every figure must cite a committed results file, and a test fails the build on any four-decimal figure here that resolves to none (FR-902) |
| Leaked weather could re-enter by hand | `f1pred predict` **refuses** a grid CSV carrying observed weather rather than ignoring the column (FR-803) |

The behavioural check is worth one more sentence, because it was passing for
eleven days against a class of leak it could not see. It permuted post-race
values *among a race's drivers*, which is correct for anything varying per car
and useless for anything the race shares — so observed weather, one value per
race, could be read by a feature declaring itself pre-race and neither layer
would object. Every race-level post-race quantity had the same exposure.

## Using it

```bash
pip install -e .

f1pred collect --weather      # the only stage that touches the network
f1pred features               # build the matrix, run the leakage firewall
f1pred train                  # fit the ranker, save it with its provenance
f1pred evaluate               # score against all three baselines
f1pred simulate               # Monte Carlo the holdout, measure calibration
f1pred predict grid.csv       # rank a starting grid for a race not yet run
```

`f1pred predict --schema` prints the grid CSV format. Any command run before
its inputs exist names the command that produces them, rather than reporting a
missing file (FR-802):

```
missing a trained model (looked in data/processed/ranker).

Produce it with:
    f1pred train
```

A full evaluation run takes **24 seconds** on the reference machine declared in
[plan §4](specs/001-core-predictor/plan.md) — Apple Silicon, macOS 25.6, Python
3.13 — against a five-minute budget, and every run records its own wall-clock
into the results file it writes *(NFR-01)*. The test suite is 700-odd tests in
under 20 seconds with no network access *(NFR-04, NFR-05)*.

## Project structure

```
.specify/memory/constitution.md   non-negotiable rules; specs must comply
specs/001-core-predictor/
  spec.md    WHAT and WHY  — 97 functional + 7 non-functional requirements
  plan.md    HOW           — architecture and design decisions
  tasks.md   ordered work  — every task traced to a requirement
src/f1pred/                       implementation
results/                          committed evaluation outputs
```

Read the spec before the code. That is the point of it.

## Development method

Spec-driven: **spec → plan → tasks → code.** Code that no task requires does not
get written. When implementation shows a spec is wrong, the spec is amended
first and the change is logged in `tasks.md` — thirty-odd such amendments are
recorded there, including the ones that cost the most to accept.

## Data and licensing

Race data is obtained via [FastF1](https://github.com/theOehrly/Fast-F1) and the
Ergast-compatible timing API, which source from Formula 1's timing feeds.
FastF1's MIT licence covers **the software, not the data** — so this repository
ships the code to acquire data, never bulk data itself. Circuit weather priors
come from the ERA5 reanalysis via Open-Meteo.

**This is a portfolio and research project.** It is not a commercial product and
not betting advice. It is unaffiliated with, and unendorsed by, Formula 1;
no Formula 1 trademark is used here in any way implying affiliation or
endorsement.
