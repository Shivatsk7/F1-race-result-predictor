"""T-604: the two FR-804 precision rules, and FR-705's label.

Both rules are stated as enforceable, so both are tested against their own
worked example from the requirement text rather than against whatever the
implementation happens to do.
"""

from __future__ import annotations

import pytest

from f1pred.render import (
    CALIBRATED,
    PrecisionError,
    ProbabilityColumn,
    UNCALIBRATED,
    decimals_supported_by,
    format_measurement,
    format_probability,
    label_probability,
    probability_table,
    render_calibration_note,
)


class TestProbabilityPrecision:
    """FR-804 rule 1: a probability, two decimals, never more."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "0.00"),
            (1.0, "1.00"),
            (0.5, "0.50"),
            (0.126, "0.13"),
            (0.124, "0.12"),
            # v0's actual output shape: four decimals, none of them meaningful.
            (0.3847, "0.38"),
        ],
    )
    def test_two_decimals_always(self, value: float, expected: str) -> None:
        assert format_probability(value) == expected

    @pytest.mark.parametrize("value", [-0.01, 1.01, 1.5, float("nan"), float("inf")])
    def test_a_value_outside_zero_to_one_raises_rather_than_clipping(self, value: float) -> None:
        # Clipping would render an upstream bug as a plausible-looking number.
        with pytest.raises(PrecisionError):
            format_probability(value)


class TestMeasurementPrecision:
    """FR-804 rule 2: no more decimals than the CI half-width supports."""

    def test_the_requirements_own_worked_example(self) -> None:
        # "a metric of 0.412 with a half-width of 0.03 renders as 0.41".
        assert format_measurement(0.412, half_width=0.03) == "0.41"

    @pytest.mark.parametrize(
        ("half_width", "places"),
        [(0.3, 1), (0.03, 2), (0.003, 3), (0.0003, 4), (0.05, 2), (0.0499, 2)],
    )
    def test_decimals_track_the_half_widths_leading_digit(
        self, half_width: float, places: int
    ) -> None:
        assert decimals_supported_by(half_width) == places

    def test_an_uncertainty_wider_than_the_unit_supports_no_decimals(self) -> None:
        assert decimals_supported_by(1.5) == 0
        assert format_measurement(0.412, half_width=1.5) == "0"

    def test_the_projects_own_headline_rounds_to_two_places(self) -> None:
        # The real numbers: grid 0.6555 (0.6050 to 0.7047), model 0.6599
        # (0.6169 to 0.7016). Both land on 0.66, which is the finding.
        assert format_measurement(0.6555, low=0.6050, high=0.7047) == "0.66"
        assert format_measurement(0.6599, low=0.6169, high=0.7016) == "0.66"

    def test_an_asymmetric_interval_takes_its_wider_arm(self) -> None:
        # Half-widths are 0.003 and 0.04. Averaging them, or taking the narrow
        # one, would claim a resolution the wide side does not have; the wide
        # arm gives two places, not three.
        assert format_measurement(0.5, low=0.497, high=0.54) == "0.50"

    def test_a_zero_width_interval_raises_rather_than_claiming_exactness(self) -> None:
        with pytest.raises(PrecisionError, match="unmeasured"):
            decimals_supported_by(0.0)

    def test_a_metric_with_no_interval_is_refused(self) -> None:
        with pytest.raises(PrecisionError, match="needs its interval"):
            format_measurement(0.412)

    def test_an_inverted_interval_is_refused(self) -> None:
        with pytest.raises(PrecisionError, match="inverted"):
            format_measurement(0.5, low=0.6, high=0.4)

    def test_the_interval_renders_at_the_same_precision_as_the_point(self) -> None:
        rendered = format_measurement(0.6555, low=0.6050, high=0.7047, with_interval=True)
        assert rendered == "0.66 (0.60 to 0.71)"

    def test_bounds_round_outward_so_rendering_never_narrows_an_interval(self) -> None:
        # Nearest-rounding both bounds of (0.6050, 0.7047) would display
        # (0.61, 0.70) — a tighter interval than the bootstrap measured.
        rendered = format_measurement(0.66, low=0.6050, high=0.7047, with_interval=True)
        assert "0.60 to 0.71" in rendered
        # An already-exact bound is left alone rather than nudged outward.
        assert "0.60 to 0.70" in format_measurement(
            0.65, low=0.60, high=0.70, with_interval=True
        )


class TestCalibrationLabel:
    """FR-705: the label is applied here, so no call site can omit it."""

    def test_an_uncalibrated_probability_carries_the_label(self) -> None:
        assert label_probability(0.42, status=UNCALIBRATED) == "0.42 [UNCALIBRATED]"

    def test_a_calibrated_probability_does_not(self) -> None:
        assert label_probability(0.42, status=CALIBRATED) == "0.42"

    def test_an_unrecognised_status_raises_rather_than_defaulting_to_calibrated(self) -> None:
        # A typo'd status silently treated as calibrated is the failure this
        # whole mechanism exists to prevent.
        with pytest.raises(PrecisionError):
            label_probability(0.42, status="probably fine")

    def test_a_column_of_unmeasured_probabilities_is_labelled_in_its_heading(self) -> None:
        headings, cells = probability_table(
            [{"driver_id": "norris", "win_probability": 0.2512}],
            [ProbabilityColumn("win", "win_probability", UNCALIBRATED)],
            label_columns=[("driver", "driver_id")],
        )
        assert headings == ["driver", "win [UNCALIBRATED]"]
        assert cells == [["norris", "0.25"]]

    def test_the_table_still_obeys_rule_one(self) -> None:
        _, cells = probability_table(
            [{"d": "x", "p": 0.123456}],
            [ProbabilityColumn("win", "p", CALIBRATED)],
            label_columns=[("driver", "d")],
        )
        assert cells == [["x", "0.12"]]


class TestCalibrationNote:
    """FR-706: the reader is told not to make a claim the numbers cannot carry."""

    def test_failing_events_are_named_and_the_prohibition_stated(self) -> None:
        note = render_calibration_note(
            {"win probability": UNCALIBRATED, "position intervals": CALIBRATED}
        )
        assert "win probability" in note
        assert "position intervals" not in note
        assert "FR-706" in note

    def test_an_all_clear_says_so_without_hedging(self) -> None:
        note = render_calibration_note({"win probability": CALIBRATED})
        assert UNCALIBRATED not in note
