from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.cli.app import app

runner = CliRunner()


class TestCLIApp:
    def test_app_loads(self):
        from betlab.cli.app import app as cli_app
        assert cli_app is not None

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Wippa Bet Lab" in result.output

    def test_dataset_help(self):
        result = runner.invoke(app, ["dataset", "--help"])
        assert result.exit_code == 0

    def test_strategy_help(self):
        result = runner.invoke(app, ["strategy", "--help"])
        assert result.exit_code == 0

    def test_experiment_help(self):
        result = runner.invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0

    def test_paper_help(self):
        result = runner.invoke(app, ["paper", "--help"])
        assert result.exit_code == 0


class TestDatasetCommands:
    def test_list_empty(self):
        result = runner.invoke(app, ["dataset", "list"])
        assert result.exit_code == 0

    def test_import_missing_file(self):
        result = runner.invoke(app, ["dataset", "import", "nonexistent.json", "--sport", "tennis", "--name", "test"])
        assert result.exit_code == 1

    def test_validate_missing(self):
        result = runner.invoke(app, ["dataset", "validate", "nonexistent"])
        assert result.exit_code == 1

    def test_info_missing(self):
        result = runner.invoke(app, ["dataset", "info", "nonexistent"])
        assert result.exit_code == 1


class TestStrategyCommands:
    def test_list_empty(self):
        result = runner.invoke(app, ["strategy", "list"])
        assert result.exit_code == 0

    def test_validate_missing_file(self):
        result = runner.invoke(app, ["strategy", "validate", "nonexistent.json"])
        assert result.exit_code == 1

    def test_validate_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        result = runner.invoke(app, ["strategy", "validate", str(bad_file)])
        assert result.exit_code == 1

    def test_validate_valid_strategy(self, tmp_path):
        strat_file = tmp_path / "valid.json"
        strat_file.write_text(json.dumps({"id": "test", "name": "Test", "sport": "tennis"}))
        result = runner.invoke(app, ["strategy", "validate", str(strat_file)])
        assert result.exit_code == 0

    def test_info_missing(self):
        result = runner.invoke(app, ["strategy", "info", "nonexistent"])
        assert result.exit_code == 1


class TestExperimentCommands:
    def test_list_empty(self):
        result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0

    def test_info_missing(self):
        result = runner.invoke(app, ["experiment", "info", "nonexistent"])
        assert result.exit_code == 1


class TestPaperCommands:
    def test_status_empty(self):
        result = runner.invoke(app, ["paper", "status"])
        assert result.exit_code == 0

    def test_pause_missing(self):
        result = runner.invoke(app, ["paper", "pause", "nonexistent"])
        assert result.exit_code == 1

    def test_resume_missing(self):
        result = runner.invoke(app, ["paper", "resume", "nonexistent"])
        assert result.exit_code == 1

    def test_stop_missing(self):
        result = runner.invoke(app, ["paper", "stop", "nonexistent"])
        assert result.exit_code == 1
