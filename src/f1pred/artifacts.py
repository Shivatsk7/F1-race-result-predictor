"""Missing upstream artifacts, and the command that produces each (FR-802).

The pipeline is a chain: collect -> features -> train -> evaluate/simulate/
predict. Every stage reads what an earlier stage wrote. When a file is not
there, the useful error is not "file not found" — it is which command puts it
there, spelled out well enough to paste into a terminal.

The table is the single place that mapping lives. A stage that invented its own
message would drift from the command names the CLI actually exposes, and the
first person to notice would be a user reading an instruction that no longer
works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from f1pred.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR

__all__ = [
    "ARTIFACTS",
    "Artifact",
    "MissingArtifactError",
    "MODEL_STEM",
    "require",
    "resolve",
]

#: Where `train` writes and `predict` reads. A stem, because a trained model is
#: two files: the booster (.ubj) and its FR-404 provenance record (.json).
MODEL_STEM: Final[Path] = PROCESSED_DIR / "ranker"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One thing a stage needs, and how to make it."""

    key: str
    what: str
    command: str
    detail: str = ""

    def message(self, path: Path | None = None) -> str:
        """FR-802's message. The command sits alone on its own line so it can
        be copied out without dragging the explanation with it."""
        where = f" (looked in {path})" if path is not None else ""
        lines = [f"missing {self.what}{where}.", "", "Produce it with:", f"    {self.command}"]
        if self.detail:
            lines += ["", self.detail]
        return "\n".join(lines)


ARTIFACTS: Final[dict[str, Artifact]] = {
    "corpus": Artifact(
        key="corpus",
        what="the race corpus",
        command="f1pred collect",
        detail=(
            "This is the only stage that uses the network, and it takes a "
            "while on a cold cache; it checkpoints per season and resumes."
        ),
    ),
    "weather": Artifact(
        key="weather",
        what="the weather archive",
        command="f1pred collect --weather",
        detail=(
            "Optional. Without it the weather features stay null and the run "
            "records that under 'absent_sources' rather than guessing values."
        ),
    ),
    "practice": Artifact(
        key="practice",
        what="the practice-telemetry checkpoints",
        command="f1pred collect --telemetry",
        detail=(
            "Optional, and 2018 onwards only. Its absence is a declared fact "
            "of the corpus (FR-214), not a gap to fill."
        ),
    ),
    "features": Artifact(
        key="features",
        what="the built feature matrix",
        command="f1pred features",
        detail="Reads the corpus; no network.",
    ),
    "model": Artifact(
        key="model",
        what="a trained model",
        command="f1pred train",
        detail="Reads the corpus and writes the booster with its provenance record.",
    ),
    "evaluation": Artifact(
        key="evaluation",
        what="the evaluation results file",
        command="f1pred evaluate",
        detail="Scores the model against all three baselines and writes results/.",
    ),
    "calibration": Artifact(
        key="calibration",
        what="the calibration results file",
        command="f1pred simulate",
        detail=(
            "Until this exists every probability is labelled UNCALIBRATED, "
            "because unmeasured calibration is failing calibration (FR-705)."
        ),
    ),
}


class MissingArtifactError(FileNotFoundError):
    """FR-802: an absent upstream artifact, named alongside its producer."""

    def __init__(self, artifact: Artifact, path: Path | None = None) -> None:
        self.artifact = artifact
        self.path = path
        super().__init__(artifact.message(path))


def resolve(key: str) -> Artifact:
    """Look an artifact up by key, failing loudly on an undeclared one.

    A typo'd key would otherwise produce an error message with an empty command
    in it — a missing-artifact error that itself fails to name the producer,
    which is precisely the defect FR-802 exists to prevent.
    """
    try:
        return ARTIFACTS[key]
    except KeyError:
        raise KeyError(
            f"no artifact declared under {key!r}; declare it in artifacts.py "
            f"rather than writing a bare message at the call site"
        ) from None


def require(key: str, path: Path | None = None, *, exists: bool | None = None) -> Path | None:
    """Raise `MissingArtifactError` unless the artifact is present.

    `exists` lets a caller do its own presence test — the corpus is a directory
    of season checkpoints, not one file — while still routing the message
    through the one table.
    """
    artifact = resolve(key)
    present = exists if exists is not None else (path is not None and Path(path).exists())
    if not present:
        raise MissingArtifactError(artifact, path)
    return Path(path) if path is not None else None


def corpus_is_present(root: Path = RAW_DIR) -> bool:
    """Whether any season checkpoint exists. One is enough to try."""
    from f1pred.dataset import available_seasons

    return bool(available_seasons(root))


def model_is_present(stem: Path = MODEL_STEM) -> bool:
    """A model is present only when both of its files are."""
    return stem.with_suffix(".ubj").exists() and stem.with_suffix(".json").exists()


def latest_result(pattern: str, *, root: Path = RESULTS_DIR) -> Path | None:
    """Newest results file matching `pattern`, or None."""
    matches = sorted(Path(root).glob(pattern))
    return matches[-1] if matches else None


__all__ += ["corpus_is_present", "latest_result", "model_is_present"]
