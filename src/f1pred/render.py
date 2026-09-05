"""The rendering layer: precision rules (FR-804) and the calibration label (FR-705).

Two rules, both enforceable, both here rather than at the call sites:

1. A probability renders to at most two decimal places. v0 printed win
   probabilities to four, and there was never any evidence the third and fourth
   digits meant anything at all.

2. A metric shown with a confidence interval renders to no more decimal places
   than its half-width supports. A score of 0.4123 whose interval is +/- 0.03
   knows its own value to about a hundredth, so it renders `0.41`. Printing
   `0.4123` claims three digits of resolution the measurement does not have.

The label lives here for the reason plan D-7 gives: a display gate enforced at
the render layer cannot be forgotten by a caller. Any probability leaving this
module while calibration is unmeasured or failing carries `UNCALIBRATED` with
it (FR-705, Constitution V.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Final

__all__ = [
    "CALIBRATED",
    "PrecisionError",
    "UNCALIBRATED",
    "decimals_supported_by",
    "format_measurement",
    "format_probability",
    "label_probability",
    "probability_table",
    "render_calibration_note",
]

#: FR-804 rule 1. Not configurable: a rule with a knob is a default.
PROBABILITY_DECIMALS: Final[int] = 2

#: Ceiling on rule 2. An interval can always be narrow enough to justify more
#: digits than a float carries honestly; this is where that stops.
MAX_DECIMALS: Final[int] = 6

CALIBRATED: Final[str] = "CALIBRATED"
UNCALIBRATED: Final[str] = "UNCALIBRATED"


class PrecisionError(ValueError):
    """Raised when a number cannot be rendered under the FR-804 rules."""


def format_probability(value: float) -> str:
    """FR-804 rule 1: a probability, to two decimal places, never more.

    Rejects values outside [0, 1] rather than clipping them. A probability of
    1.4 is a bug upstream, and rendering it as `1.00` hides the bug behind a
    plausible-looking number.
    """
    if not math.isfinite(value):
        raise PrecisionError(f"probability is not finite: {value!r}")
    if not (0.0 <= value <= 1.0):
        raise PrecisionError(
            f"probability {value!r} lies outside [0, 1]; this is an upstream "
            f"defect, not a rendering one"
        )
    return f"{value:.{PROBABILITY_DECIMALS}f}"


def decimals_supported_by(half_width: float) -> int:
    """How many decimals a measurement of this uncertainty can honestly show.

    The answer is the decimal place of the half-width's first significant
    digit: a half-width of 0.03 resolves the hundredths, so two decimals; a
    half-width of 0.003 resolves the thousandths, so three. FR-804's own worked
    example (0.412 +/- 0.03 renders `0.41`) is this rule.

    A half-width of zero is not "infinite precision" — it means the interval
    was never computed, or was computed from a single observation. It raises,
    because silently returning `MAX_DECIMALS` would turn a missing measurement
    into the most confident number on the page.
    """
    if not math.isfinite(half_width):
        raise PrecisionError(f"half-width is not finite: {half_width!r}")
    if half_width < 0.0:
        raise PrecisionError(f"half-width is negative: {half_width!r}")
    if half_width == 0.0:
        raise PrecisionError(
            "half-width is zero: an interval of no width is an unmeasured "
            "interval, not an exact one"
        )
    if half_width >= 1.0:
        # The uncertainty exceeds the unit the metric is measured in. Nothing
        # after the decimal point is supported; zero decimals is the honest read.
        return 0
    return min(MAX_DECIMALS, -math.floor(math.log10(half_width)))


def format_measurement(
    point: float,
    *,
    low: float | None = None,
    high: float | None = None,
    half_width: float | None = None,
    with_interval: bool = False,
) -> str:
    """FR-804 rule 2: a metric rendered at the precision its interval supports.

    Give either the interval bounds or the half-width directly. Asymmetric
    bounds take the larger arm, because the wider side is what the number is
    actually uncertain by — averaging the two would report a resolution the
    measurement does not have on one side.
    """
    if half_width is None:
        if low is None or high is None:
            raise PrecisionError(
                "a metric needs its interval to be rendered: pass low and high, "
                "or half_width. A metric with no interval is not covered by "
                "FR-804 rule 2 and should not be routed through it."
            )
        if high < low:
            raise PrecisionError(f"interval is inverted: [{low!r}, {high!r}]")
        half_width = max(point - low, high - point)
    if not math.isfinite(point):
        raise PrecisionError(f"metric is not finite: {point!r}")

    places = decimals_supported_by(half_width)
    rendered = f"{point:.{places}f}"
    if not with_interval:
        return rendered
    if low is None or high is None:
        return f"{rendered} (+/-{_round_outward(half_width, places, up=True)})"
    return (
        f"{rendered} ({_round_outward(low, places, up=False)} to "
        f"{_round_outward(high, places, up=True)})"
    )


def _round_outward(value: float, places: int, *, up: bool) -> str:
    """Round an interval bound away from the point estimate, never toward it.

    Nearest-rounding a bound can narrow a measured interval: 0.6050 displayed
    to two places becomes 0.61, and the reader is shown a tighter result than
    the bootstrap found. Widening by less than half a displayed digit is the
    honest direction, so the lower bound floors and the upper bound ceils.
    """
    quantum = Decimal(1).scaleb(-places)
    rounding = ROUND_CEILING if up else ROUND_FLOOR
    return str(Decimal(repr(value)).quantize(quantum, rounding=rounding))


def label_probability(value: float, *, status: str) -> str:
    """A probability and its calibration status, inseparable (FR-705).

    There is deliberately no way to get the number out of this module without
    the status travelling with it. That is the whole design: v0's probabilities
    were unlabelled because labelling them was a call-site's job and no call
    site did it.
    """
    if status not in (CALIBRATED, UNCALIBRATED):
        raise PrecisionError(
            f"calibration status must be {CALIBRATED!r} or {UNCALIBRATED!r}, "
            f"not {status!r}; an unmeasured status is {UNCALIBRATED!r}"
        )
    rendered = format_probability(value)
    return rendered if status == CALIBRATED else f"{rendered} [{UNCALIBRATED}]"


@dataclass(frozen=True, slots=True)
class ProbabilityColumn:
    """One probability column of a rendered table, with its own status."""

    heading: str
    key: str
    status: str


def probability_table(
    rows: list[dict[str, Any]],
    columns: list[ProbabilityColumn],
    *,
    label_columns: list[tuple[str, str]],
) -> tuple[list[str], list[list[str]]]:
    """Render rows of probabilities under both FR-804 and FR-705.

    Returns headings and cells rather than a formatted string so the caller
    chooses the table style, but never the precision or the label. Statuses
    are hoisted into the column heading rather than repeated on every cell,
    which keeps a twenty-row grid readable while still making the label
    impossible to miss.
    """
    headings = [heading for heading, _ in label_columns]
    for column in columns:
        if column.status not in (CALIBRATED, UNCALIBRATED):
            raise PrecisionError(
                f"column {column.heading!r} has status {column.status!r}; "
                f"an unmeasured status is {UNCALIBRATED!r}"
            )
        suffix = "" if column.status == CALIBRATED else f" [{UNCALIBRATED}]"
        headings.append(f"{column.heading}{suffix}")

    cells: list[list[str]] = []
    for row in rows:
        line = [str(row[key]) for _, key in label_columns]
        line.extend(format_probability(float(row[column.key])) for column in columns)
        cells.append(line)
    return headings, cells


def render_calibration_note(statuses: dict[str, str]) -> str:
    """One paragraph saying which outputs are trustworthy and which are not.

    FR-706 forbids an uncalibrated probability supporting a claim. The note is
    what stops a reader making the claim on the project's behalf.
    """
    failing = sorted(k for k, v in statuses.items() if v != CALIBRATED)
    if not failing:
        return (
            "All probabilities shown have been checked against held-out races "
            "and land within the documented calibration threshold."
        )
    listed = ", ".join(failing)
    return (
        f"Calibration: {listed} {'is' if len(failing) == 1 else 'are'} marked "
        f"{UNCALIBRATED}. The number is the model's honest belief, but how "
        f"often events at that stated probability actually happen has been "
        f"measured and misses the documented threshold, so these figures must "
        f"not be used to support a performance claim (FR-706)."
    )
