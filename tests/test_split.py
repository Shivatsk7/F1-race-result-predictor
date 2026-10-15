"""The one-season fold in `split.py` (change evaluation-rules-rework, task 3.1).

`SeasonFold` behaviour is pinned in `test_regime_transfer.py`, which imports it
under its old name; these tests pin the move itself and the practice-exam folds.
"""

from __future__ import annotations

import pytest

from f1pred.corpus import CorpusWindow
from f1pred.evaluate import regime_transfer
from f1pred.evaluate.split import (
    DIAGNOSTIC_LABEL,
    ChronologicalSplit,
    FoldError,
    SeasonFold,
    SplitError,
)

ARCHIVE = tuple(range(2010, 2026))


class TestTheMove:
    def test_regime_transfer_uses_the_fold_from_split(self):
        assert regime_transfer.TransferSplit is SeasonFold
        assert regime_transfer.DIAGNOSTIC_LABEL == DIAGNOSTIC_LABEL

    def test_one_error_type_for_folds_and_transfer_runs(self):
        assert regime_transfer.TransferError is FoldError
        assert issubclass(FoldError, SplitError)

    def test_the_multi_season_split_still_refuses_one_season(self):
        with pytest.raises(SplitError, match="below the Constitution IV.2 floor"):
            ChronologicalSplit(window=CorpusWindow(width=16), holdout_seasons=1)


class TestPracticeExamFolds:
    """The three folds of Constitution IV.4c, built from the real season range."""

    @pytest.mark.parametrize("scored", [2020, 2021, 2022])
    def test_each_fold_trains_on_every_earlier_season_only(self, scored):
        fold = SeasonFold.from_archive(ARCHIVE, scored)
        assert fold.training_seasons == tuple(range(2010, scored))
        assert fold.holdout == (scored,)
        assert fold.meets_constitutional_minimum
        assert fold.label is None

    def test_a_fold_never_trains_on_a_final_exam_season(self):
        for scored in (2020, 2021, 2022):
            fold = SeasonFold.from_archive(ARCHIVE, scored)
            assert max(fold.training_seasons) < 2023

    def test_seasons_are_plain_ints(self):
        """They reach JSON, which refuses numpy integers."""
        import numpy as np

        fold = SeasonFold(training_seasons=(np.int64(2010),), evaluated_season=np.int64(2011))
        assert all(type(s) is int for s in fold.seasons)

    def test_duplicates_are_refused(self):
        with pytest.raises(FoldError, match="duplicate"):
            SeasonFold(training_seasons=(2010, 2010), evaluated_season=2011)
