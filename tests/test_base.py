"""T-514 — no published prediction may be a bare point estimate (FR-752).

`rank_to_positions` and the `Predictor` protocol have their behaviour
exercised indirectly through the baselines and the ranker; this file is for
`assert_carries_uncertainty`, the FR-752 check itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.models.base import BarePointEstimateError, assert_carries_uncertainty
from tests.test_ranker import fitted, one_race  # noqa: F401 -- reused fixtures


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestAssertCarriesUncertainty:
    def test_passes_when_both_bounds_are_present(self):
        frame = _frame(
            predicted_position=[1.0, 2.0],
            predicted_position_lower=[0.5, 1.5],
            predicted_position_upper=[1.5, 2.5],
        )
        assert_carries_uncertainty(frame)  # must not raise

    def test_passes_trivially_when_there_is_no_point_column_at_all(self):
        # Nothing published here for FR-752 to police.
        assert_carries_uncertainty(_frame(some_other_column=[1, 2]))

    def test_raises_when_both_bounds_are_missing(self):
        frame = _frame(predicted_position=[1.0, 2.0])
        with pytest.raises(BarePointEstimateError, match="predicted_position_lower"):
            assert_carries_uncertainty(frame)

    def test_raises_when_only_the_upper_bound_is_missing(self):
        frame = _frame(
            predicted_position=[1.0], predicted_position_lower=[0.5]
        )
        with pytest.raises(BarePointEstimateError, match="predicted_position_upper"):
            assert_carries_uncertainty(frame)

    def test_raises_when_a_bound_is_null_on_any_row(self):
        frame = _frame(
            predicted_position=[1.0, 2.0],
            predicted_position_lower=[0.5, None],
            predicted_position_upper=[1.5, 2.5],
        )
        with pytest.raises(BarePointEstimateError, match="null interval bound"):
            assert_carries_uncertainty(frame)

    def test_honours_custom_column_names(self):
        frame = _frame(point=[1.0], lo=[0.5], hi=[1.5])
        assert_carries_uncertainty(
            frame, point_column="point", lower_column="lo", upper_column="hi"
        )  # must not raise

        with pytest.raises(BarePointEstimateError):
            assert_carries_uncertainty(
                _frame(point=[1.0]), point_column="point", lower_column="lo", upper_column="hi"
            )


class TestPredictWithIntervalNeverPublishesABarePoint:
    """The one live call site this project has today (T-514)."""

    def test_predict_with_interval_output_passes_the_fr752_check(self, fitted, one_race):
        # fitted.predict_with_interval already calls assert_carries_uncertainty
        # internally (models/ranker.py); re-checking here pins that the
        # published shape stays checkable from outside the class too.
        result = fitted.predict_with_interval(one_race)
        assert_carries_uncertainty(result)
