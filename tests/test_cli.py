"""T-601 and T-602: the six commands, and errors that name their producer.

Offline throughout (NFR-04, NFR-05). Nothing here runs the pipeline — the
stages have their own tests — so what is checked is the surface: that the six
commands exist, that a missing artifact comes back naming the command which
makes it, and that a broken CSV exits non-zero with every violation shown.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from f1pred import artifacts
from f1pred.artifacts import ARTIFACTS, Artifact, MissingArtifactError, require
from f1pred.cli import cli, main

FIXTURES = Path(__file__).parent / "fixtures"

#: FR-801 enumerates six. The list is the requirement, not a summary of it.
FR801_COMMANDS = frozenset(
    {"collect", "features", "train", "evaluate", "simulate", "predict"}
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCommandSurface:
    """FR-801: six commands, present and callable."""

    def test_all_six_commands_are_registered(self) -> None:
        assert set(cli.commands) == FR801_COMMANDS

    def test_no_seventh_command_has_crept_in(self) -> None:
        # The requirement names six. A seventh would be a surface change that
        # nothing in the spec asked for.
        assert len(cli.commands) == 6

    @pytest.mark.parametrize("name", sorted(FR801_COMMANDS))
    def test_every_command_has_help(self, runner: CliRunner, name: str) -> None:
        result = runner.invoke(cli, [name, "--help"])
        assert result.exit_code == 0
        assert result.output.strip()

    def test_the_group_lists_every_command(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        for name in FR801_COMMANDS:
            assert name in result.output


class TestMissingArtifactErrors:
    """FR-802: an absent upstream artifact names the command that makes it."""

    def test_every_declared_artifact_names_a_command_that_exists(self) -> None:
        for artifact in ARTIFACTS.values():
            assert artifact.command.startswith("f1pred ")
            assert artifact.command.split()[1] in FR801_COMMANDS, artifact.key

    @pytest.mark.parametrize("key", sorted(ARTIFACTS))
    def test_every_message_carries_its_producing_command(self, key: str) -> None:
        with pytest.raises(MissingArtifactError) as caught:
            require(key, Path("/nonexistent"), exists=False)
        assert ARTIFACTS[key].command in str(caught.value)

    def test_the_command_sits_alone_on_a_line_so_it_can_be_copied(self) -> None:
        message = ARTIFACTS["model"].message(Path("/tmp/x"))
        lines = [line.strip() for line in message.splitlines()]
        assert "f1pred train" in lines

    def test_the_path_looked_in_is_reported(self) -> None:
        with pytest.raises(MissingArtifactError, match="somewhere/else"):
            require("corpus", Path("somewhere/else"), exists=False)

    def test_a_present_artifact_does_not_raise(self, tmp_path: Path) -> None:
        present = tmp_path / "here.json"
        present.write_text("{}")
        assert require("evaluation", present) == present

    def test_an_undeclared_key_fails_loudly_rather_than_producing_an_empty_message(
        self,
    ) -> None:
        # A typo'd key must not yield a missing-artifact error that itself
        # fails to name a producer — the exact defect FR-802 prevents.
        with pytest.raises(KeyError, match="declare it in artifacts.py"):
            require("modle", Path("/nonexistent"), exists=False)

    def test_predict_without_a_model_names_the_train_command(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(artifacts, "MODEL_STEM", tmp_path / "absent")
        monkeypatch.setattr("f1pred.cli._require_corpus", lambda: None)
        result = runner.invoke(cli, ["predict", str(tmp_path / "grid.csv")])
        assert result.exit_code != 0
        assert "f1pred train" in str(result.exception or result.output)


class TestPredictSchema:
    """FR-803 says the schema is documented; `--schema` is that documentation."""

    def test_schema_prints_without_needing_any_artifact(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["predict", "--schema"])
        assert result.exit_code == 0
        assert "grid_position" in result.output
        assert "driver_id" in result.output

    def test_the_schema_says_which_columns_are_refused_and_why(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(cli, ["predict", "--schema"])
        assert "weather_mixed" in result.output
        assert "climatological prior" in result.output

    def test_predict_with_no_argument_and_no_schema_flag_explains_itself(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(cli, ["predict"])
        assert result.exit_code != 0
        assert "--schema" in result.output


class TestExitCodes:
    """`main` turns the two expected failures into an exit code, not a traceback."""

    def test_a_broken_grid_csv_exits_two_and_lists_every_violation(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("f1pred.cli._require_corpus", lambda: None)
        monkeypatch.setattr(artifacts, "model_is_present", lambda stem: True)
        fixture = FIXTURES / "2025_bahrain_starting_grid.csv"

        code = main(["predict", str(fixture)])

        assert code == 2
        captured = capsys.readouterr()
        assert "problems in" in captured.err
        assert "weather_mixed" in captured.err
        # Every violation, not the first: the header alone has a dozen faults.
        assert "\n  10. " in captured.err

    def test_a_missing_artifact_exits_two_without_a_traceback(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("f1pred.cli._require_corpus", lambda: None)
        monkeypatch.setattr(artifacts, "MODEL_STEM", tmp_path / "absent")

        code = main(["predict", str(tmp_path / "grid.csv")])

        assert code == 2
        captured = capsys.readouterr()
        assert "f1pred train" in captured.err
        assert "Traceback" not in captured.err

    def test_help_exits_zero(self) -> None:
        assert main(["--help"]) == 0


class TestArtifactHelpers:
    def test_a_model_needs_both_of_its_files(self, tmp_path: Path) -> None:
        stem = tmp_path / "ranker"
        assert not artifacts.model_is_present(stem)
        stem.with_suffix(".ubj").write_bytes(b"")
        # A booster with no provenance record is not a usable model (FR-404).
        assert not artifacts.model_is_present(stem)
        stem.with_suffix(".json").write_text("{}")
        assert artifacts.model_is_present(stem)

    def test_an_empty_corpus_directory_reads_as_absent(self, tmp_path: Path) -> None:
        assert not artifacts.corpus_is_present(tmp_path)

    def test_the_newest_matching_result_is_returned(self, tmp_path: Path) -> None:
        (tmp_path / "calibration_2020_2022.json").write_text("{}")
        (tmp_path / "calibration_2023_2025.json").write_text("{}")
        found = artifacts.latest_result("calibration_*.json", root=tmp_path)
        assert found is not None and found.name == "calibration_2023_2025.json"

    def test_no_match_returns_none_rather_than_raising(self, tmp_path: Path) -> None:
        assert artifacts.latest_result("nothing_*.json", root=tmp_path) is None


class TestArtifactTable:
    def test_the_message_names_what_is_missing_and_how_to_make_it(self) -> None:
        artifact = Artifact(key="x", what="the thing", command="f1pred train")
        message = artifact.message()
        assert "missing the thing" in message
        assert "f1pred train" in message
