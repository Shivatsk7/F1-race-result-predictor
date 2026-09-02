# Race Result Predictor

> **Status: Phase P4 (model) in progress. Phases P0-P3 complete.**
> Spec 001 is ratified. The corpus is collected (16 seasons, 2010-2025, 6,868
> classified starts across 329 races), the feature registry and its leakage
> firewall are in place, and the three baselines were scored *before* the model
> existed so the bar could not move afterwards.
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
| **model** (XGBoost ranker) | **0.6599** | 0.6169 to 0.7016 |
| grid order | 0.6555 | 0.6050 to 0.7047 |
| championship-standings order | 0.6128 | 0.5683 to 0.6576 |
| random order (1000 seeded draws) | 0.0012 | -0.0494 to 0.0516 |

**Model minus grid: +0.0044, 95% CI -0.0244 to +0.0318.** The interval contains
zero, so the difference is not distinguishable from noise. Forty features, a
16-season corpus and a trained ranker land in the same place as reading the
grid sheet.

Three things this does *not* say. It is not a claim that the model is worthless
— it is far above the random floor, so it has learned something real; it has
learned roughly what the grid already knew. It is not a claim that no model
could do better. And its prediction intervals are **UNCALIBRATED**: their width
has been fitted but never checked against how often the truth actually lands
inside them, which is Phase P5's work.

The random control is the check on the harness itself: a metric that flattered
random orderings would be measuring something other than skill.

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
| Baseline never computed | Baselines share the model's evaluation code path and are mandatory output (FR-304, Constitution II) |
| Forty-row holdout | Minimum three-season chronological holdout, bootstrap CIs on every metric (FR-501, FR-503) |
| Uncalibrated probabilities shown | Reliability curves + Brier scores required; uncalibrated output is labelled at the render layer (FR-701, D-7) |
| Collection died and lost 3 seasons | Per-season checkpointing; interrupted runs resume (FR-102, FR-103) |
| README claimed unreproducible numbers | Every figure must cite a committed results file (FR-902) |

## Project structure

```
.specify/memory/constitution.md   non-negotiable rules; specs must comply
specs/001-core-predictor/
  spec.md    WHAT and WHY  — 91 functional + 7 non-functional requirements
  plan.md    HOW           — architecture and design decisions
  tasks.md   ordered work  — every task traced to a requirement
src/f1pred/                       implementation
results/                          committed evaluation outputs
```

Read the spec before the code. That is the point of it.

## Development method

Spec-driven: **spec → plan → tasks → code.** Code that no task requires does not
get written. When implementation shows a spec is wrong, the spec is amended
first and the change is logged in `tasks.md`.

## Data and licensing

Race data is obtained via [FastF1](https://github.com/theOehrly/Fast-F1), which
sources from Formula 1's timing feeds. FastF1's MIT licence covers the software,
not the data — so this repository ships the code to acquire data, never bulk
data itself.

**This is a portfolio and research project.** It is not a commercial product and
not betting advice. It is unaffiliated with and unendorsed by Formula 1.
