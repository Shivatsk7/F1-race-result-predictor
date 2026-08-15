# Project Constitution

**Project:** Race Result Predictor
**Version:** 1.2.0
**Ratified:** 2026-08-15 (v1.0.0) · Amended 2026-08-15 (v1.1.0, v1.2.0)

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

**III.6** This Article prohibits **knowing the outcome**, not **modelling
randomness**. The two are opposites and SHALL NOT be conflated:

| Admissible — uncertainty *modelled* | Inadmissible — uncertainty *deleted* |
|---|---|
| Pre-race probability of rain | Whether it actually rained |
| Historical DNF rate for a team | Whether that car actually retired |
| Circuit safety-car base rate | Whether a safety car actually deployed |
| Distribution over finishing positions | The actual finishing position |

Supplying an actual outcome does not capture unpredictability — it removes the
model's need to handle it. The affirmative obligation to represent race
randomness is Article X.

> **Rationale.** In v0, `weather_mixed` was the third most important feature and
> was computed from the actual weather of the race being predicted. Every metric
> the project reported was inflated by knowing whether it rained.

---

## Article IV — Chronological Evaluation Only

**IV.1** Train/test splits SHALL be chronological. Random splits over race data
are prohibited, as they leak future information into the past.

**IV.2** The dataset SHALL span **at least eight distinct seasons**, split as:

| Partition | Minimum |
|-----------|---------|
| Total corpus | 8 seasons |
| Training | 5 seasons |
| Holdout | 3 seasons |

Evaluation on a single season, or on fewer than 200 driver-race rows, is not
evidence.

**IV.2a** IF fewer than eight seasons can be collected, THEN the shortfall SHALL
be stated in the README and in every results file, and the minima SHALL NOT be
quietly lowered to fit what was collected. A smaller corpus is a published
limitation, never a silent redefinition.

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

**V.4** Probabilities whose calibration is unmeasured or failing MAY be
displayed, and SHALL carry an unmissable `UNCALIBRATED` label wherever they
appear — CLI tables, exported CSVs, plots, and documentation alike. The label
SHALL be applied at the rendering layer, so that no call site can omit it.

**V.5** Probabilities that are unmeasured or failing calibration SHALL NOT be
used to support any performance claim, in the README or elsewhere.

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

**VII.1** The order of work is **spec → plan → tasks → code**. **Production**
code that no task requires SHALL NOT be written.

**VII.1a** The following are exempt from VII.1 and may be written freely:

- **Spikes** — throwaway code answering a factual question needed to write a
  sensible requirement ("does FastF1 expose lap-1 position?"). Spikes live in
  `spikes/`, are gitignored, and are never imported by production code.
- **Support code** — test helpers, fixtures, dev scripts, profiling harnesses,
  and diagnostics that serve a task without being shipped by it.
- **Refactoring** of code already required by a task, where behaviour is
  unchanged.

The rule exists to stop unrequested *features* accreting, not to make
investigation illegal. WHERE a spike reveals something that changes the
requirements, the spec is amended per VII.3 and the spike is deleted.

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

## Article IX — Human Ratification

**IX.1** A spec SHALL carry `Status: Draft` until explicitly ratified by the
project owner. Implementation against a Draft spec is prohibited.

**IX.2** Ratification means the owner has read the spec in full. Approving a
summary, an outline, or a set of scoping questions is **not** ratification.

**IX.3** WHERE an agent drafts a spec, the drafting agent SHALL NOT also ratify
it. Author and approver SHALL be different parties.

**IX.4** Ratification SHALL be recorded in the spec's header with a date and the
ratifying party's name.

**IX.5** A material amendment to a ratified spec SHALL return it to Draft for the
amended section, and SHALL require re-ratification of that section.

> **Rationale.** Constitution v1.0.0 mandated the order *spec → plan → tasks →
> code* but never required a human to approve the spec before implementation
> began. An agent could therefore author the requirements, approve them
> implicitly by proceeding, and implement against them — satisfying every stated
> rule while removing the human from the only decision that matters. This
> happened: Phase P0 executed against an unratified draft. A rule that permits
> the thing it exists to prevent is a defective rule.

---

## Article X — Race Randomness Must Be Modelled, Not Known

Formula 1 outcomes are irreducibly stochastic. Retirements, safety cars, first-lap
contact, and weather swings are not noise to be minimised away — they are part of
the phenomenon being predicted, and a model that ignores them is wrong even when
its point predictions look accurate.

**X.1** The system SHALL explicitly model irreducible race randomness — at
minimum: retirement, safety-car incidence, pace variability, and position-change
volatility — as **distributions over outcomes**.

**X.2** No prediction SHALL be published as a bare point estimate. Every
predicted position SHALL carry a distribution or interval expressing its
uncertainty.

**X.3** Parameters governing randomness SHALL be estimated from **prior** races
only, per III.3. Modelling that a car *might* retire is required; knowing that it
*did* is prohibited.

**X.4** Evaluation SHALL include at least one metric that rewards correctly
stated uncertainty (Brier or log score), not rank accuracy alone. A system that
nails the modal outcome while misrepresenting variance is not a good race model.

**X.5** The system SHALL estimate and report the **irreducible noise floor** —
the share of finishing-order variance that no pre-race information could
explain. Model error SHALL be interpreted against that floor rather than against
zero.

> **Rationale.** Raised by the project owner: F1's inherent unpredictability
> affects results, and the constitution addressed it only by prohibition. It said
> what may not be known and nothing about what must be modelled. X.5 matters
> most — without a noise floor, a model that is doing genuinely well against an
> unpredictable sport is indistinguishable from one that is doing badly, and the
> temptation to close that gap with leaked features becomes overwhelming. That is
> precisely how v0 went wrong.

---

## Amendments

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| 1.0.0 | 2026-08-15 | Initial ratified text, Articles I–VIII | — |
| **1.1.0** | **2026-08-15** | **Added Article IX — Human Ratification** | Owner observed that the agent had authored a spec and begun implementing against it with no approval gate. v1.0.0 permitted this. IX.3 makes author-and-approver separation structural. |
| **1.2.0** | **2026-08-15** | **V.4 softened**: uncalibrated probabilities may be displayed if labelled at the render layer. New V.5 bars them from supporting claims. | v1.1.0's V.4 barred display outright while FR-705 required labelled display — a direct contradiction in which FR-705 cited V.4 as its authority. Owner chose display-with-label. |
| **1.2.0** | **2026-08-15** | **VII.1 narrowed to production code**; new VII.1a exempts spikes, support code, and behaviour-preserving refactors. | The rule banned investigation, not just unrequested features. Owner: "don't forbid writing supportive code if required." |
| **1.2.0** | **2026-08-15** | **IV.2 raised**: 8-season corpus, 5 training, 3 holdout. New IV.2a requires publishing any shortfall rather than lowering the bar. | Owner judged the previous floor too small to learn from. Note: the old three-season figure was the *holdout* minimum, not a cap on training. |
| **1.2.0** | **2026-08-15** | **New III.6** distinguishing leakage from stochasticity; **new Article X — Race Randomness Must Be Modelled, Not Known** | Owner observed Article III addressed F1's unpredictability only by prohibition. Article III was **not** loosened: supplying actual outcomes deletes uncertainty rather than modelling it. Instead the missing affirmative obligation was added, including X.5's irreducible noise floor. |

### Outstanding compliance debt

**Phase P0 (commit `920b7c8`) was executed against an unratified Draft spec**,
which Article IX.1 now prohibits. This is recorded rather than quietly
retro-approved. The work is deletion-only and fully recoverable at tag
`v0-vibe-coded`; it is not re-litigated, but it does not count as precedent.
Spec 001 requires ratification before Phase P1 begins.
