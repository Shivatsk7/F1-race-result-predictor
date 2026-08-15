# Race Result Predictor

> **Status: mid-rebuild (Phase P0 complete). Phase P1 is BLOCKED.**
> Spec 001 is drafted but **not ratified**, and Constitution IX.1 prohibits
> implementing against a Draft spec. The gate lifts when the owner has read
> [`spec.md`](specs/001-core-predictor/spec.md) in full and ticked its
> ratification checklist.
>
> This project is being rebuilt spec-first. The README will carry the headline
> result once Phase P4 produces it.
>
> **Governing documents** (note: `.specify/` is a hidden directory — `Cmd+Shift+.`
> in Finder, or use the links):
> [constitution](.specify/memory/constitution.md) ·
> [spec](specs/001-core-predictor/spec.md) ·
> [plan](specs/001-core-predictor/plan.md) ·
> [tasks](specs/001-core-predictor/tasks.md)
>
> Deliberately, there are **no performance claims on this page yet.** The
> previous version of this README claimed numbers the code could not reproduce.
> Under [Constitution I.3](.specify/memory/constitution.md), a claim appears
> here only once it resolves to a committed file in `results/`.

---

## The question

Formula 1 finishing order is largely set by qualifying. So any race-result model
is competing against a signal anyone can read off the grid sheet for free.

**Can a model that knows only pre-race information predict finishing order
measurably better than the starting grid alone — and can it state its
uncertainty in probabilities that are actually calibrated?**

A negative answer is a valid result and will be published as one.

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
  spec.md    WHAT and WHY  — 46 functional requirements
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
