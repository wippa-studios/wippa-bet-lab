from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.cli.app import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean registry DB."""
    db_path = Path(".betlab/models.db")
    if db_path.exists():
        db_path.unlink()
    yield
    if db_path.exists():
        db_path.unlink()


class TestModelCLI:
    def test_model_help(self):
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "Model" in result.output or "model" in result.output.lower()

    def test_model_columns_help(self):
        result = runner.invoke(app, ["model", "columns", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower() or "Dataset" in result.output

    def test_model_train_help(self):
        result = runner.invoke(app, ["model", "train", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower() or "Dataset" in result.output

    def test_model_list_empty(self):
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0

    def test_model_columns_missing_dataset(self):
        result = runner.invoke(app, ["model", "columns", "--dataset", "nonexistent"])
        assert result.exit_code == 1

    def test_model_preview_missing_dataset(self):
        result = runner.invoke(app, ["model", "preview", "--dataset", "nonexistent"])
        assert result.exit_code == 1

    def test_model_info_missing(self):
        result = runner.invoke(app, ["model", "info", "nonexistent"])
        assert result.exit_code == 1

    def test_model_results_missing(self):
        result = runner.invoke(app, ["model", "results", "nonexistent"])
        assert result.exit_code == 1

    def test_model_importance_missing(self):
        result = runner.invoke(app, ["model", "importance", "nonexistent"])
        assert result.exit_code == 1
