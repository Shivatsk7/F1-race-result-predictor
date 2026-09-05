"""Predicting an unraced grid (T-601, US-4, AC-13).

The interesting tests here are the negative ones. Predicting a future race
works by appending it to the corpus, and that is only safe while the appended
race is chronologically last — so the guard that enforces it gets more
attention than the happy path, which the CLI's end-to-end run already covers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.predict import PredictionError, append_unraced
from tests.fixtures.races import synthetic_corpus


def _grid(
    *, season: int = 2016, round_number: int = 1, race_date: str = "2016-03-20", n: int = 4
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": pd.Series([season] * n, dtype="int64"),
            "round": pd.Series([round_number] * n, dtype="int64"),
            "race_date": pd.Series([pd.Timestamp(race_date)] * n, dtype="datetime64[ns]"),
            "circuit_id": pd.Series(["albert_park"] * n, dtype="string"),
            "driver_id": pd.Series([f"driver_{i:02d}" for i in range(n)], dtype="string"),
            "constructor_id": pd.Series([f"team_{i // 2:02d}" for i in range(n)], dtype="string"),
            "grid_position": pd.Series(range(1, n + 1), dtype="int64"),
        }
    )


@pytest.fixture
def corpus() -> pd.DataFrame:
    return synthetic_corpus()


class TestTheLeakGuard:
    """A race appended anywhere but the end would leak its placeholder outcome.

    Every form feature is shifted to read only earlier races. That makes the
    placeholder harmless at the end of the corpus and harmful anywhere else —
    and harmful silently, which is why it is asserted rather than assumed.
    """

    def test_a_race_after_the_corpus_is_accepted(self, corpus: pd.DataFrame) -> None:
        combined = append_unraced(corpus, _grid())
        assert len(combined) == len(corpus) + 4

    def test_a_race_already_in_the_corpus_is_refused(self, corpus: pd.DataFrame) -> None:
        existing_season = int(corpus["season"].max())
        with pytest.raises(PredictionError, match="already in the corpus"):
            append_unraced(corpus, _grid(season=existing_season, round_number=1))

    def test_the_refusal_points_at_evaluate_instead(self, corpus: pd.DataFrame) -> None:
        # Predicting a race the model trained on is what `evaluate` exists to
        # do properly; the error says so rather than just refusing.
        with pytest.raises(PredictionError, match="f1pred evaluate"):
            append_unraced(corpus, _grid(season=int(corpus["season"].max()), round_number=1))

    def test_a_race_dated_before_the_corpus_ends_is_refused(
        self, corpus: pd.DataFrame
    ) -> None:
        with pytest.raises(PredictionError, match="not after the corpus"):
            append_unraced(corpus, _grid(season=2016, round_number=1, race_date="2013-01-01"))

    def test_the_refusal_explains_the_leak_rather_than_just_the_rule(
        self, corpus: pd.DataFrame
    ) -> None:
        with pytest.raises(PredictionError, match="leak its placeholder outcome"):
            append_unraced(corpus, _grid(race_date="2013-01-01"))

    def test_a_race_on_the_same_day_as_the_corpus_end_is_refused(
        self, corpus: pd.DataFrame
    ) -> None:
        # Same-day is not "after": the shift cannot separate them.
        last = pd.Timestamp(corpus["race_date"].max())
        with pytest.raises(PredictionError, match="not after the corpus"):
            append_unraced(corpus, _grid(season=2016, race_date=str(last.date())))


class TestTheAppendedRows:
    def test_the_appended_rows_carry_the_corpus_schema(self, corpus: pd.DataFrame) -> None:
        combined = append_unraced(corpus, _grid())
        assert list(combined.columns) == list(corpus.columns)

    def test_the_grid_positions_survive_the_append(self, corpus: pd.DataFrame) -> None:
        combined = append_unraced(corpus, _grid())
        appended = combined.tail(4)
        assert sorted(appended["grid_position"]) == [1, 2, 3, 4]

    def test_the_appended_race_is_the_last_race_in_the_result(
        self, corpus: pd.DataFrame
    ) -> None:
        combined = append_unraced(corpus, _grid())
        assert combined["race_date"].idxmax() >= len(corpus)

    def test_the_placeholder_outcome_is_a_real_enum_value(
        self, corpus: pd.DataFrame
    ) -> None:
        # A placeholder the schema does not recognise fails deep inside the
        # target builder rather than here, where the cause is visible.
        from f1pred.data.schema import ClassificationOutcome

        combined = append_unraced(corpus, _grid())
        appended = combined.tail(4)
        for value in appended["classification_outcome"]:
            assert ClassificationOutcome(value) is ClassificationOutcome.CLASSIFIED

    def test_the_corpus_itself_is_not_modified(self, corpus: pd.DataFrame) -> None:
        before = len(corpus)
        append_unraced(corpus, _grid())
        assert len(corpus) == before
