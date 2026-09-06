from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.core.schemas.models import (
    BetSide,
    BetStatus,
    SimulatedBet,
)
from betlab.backtester.engine import BacktestResult
from betlab.reporting.json_report import generate_json_report
from betlab.reporting.markdown_report import generate_markdown_report
from betlab.reporting.html_report import generate_html_report
from betlab.reporting.comparator import compare_strategies


def _make_result() -> BacktestResult:
    bets = [
        SimulatedBet(
            id="b1", experiment_id="exp-1", event_id="e1", market_id="m1", runner_id="r1",
            side=BetSide.back, requested_price=2.0, matched_price=2.0, stake=10.0,
            result=BetStatus.settled, profit=10.0,
        ),
        SimulatedBet(
            id="b2", experiment_id="exp-1", event_id="e1", market_id="m1", runner_id="r1",
            side=BetSide.back, requested_price=3.0, matched_price=3.0, stake=10.0,
            result=BetStatus.settled, profit=-10.0,
        ),
    ]
    return BacktestResult(
        experiment_id="exp-test-1",
        total_bets=2,
        wins=1,
        losses=1,
        gross_profit=10.0,
        net_profit=0.0,
        roi=0.0,
        yield_pct=0.0,
        max_drawdown=0.05,
        average_drawdown=0.02,
        bankroll_history=[
            {"timestamp": "2025-01-01T00:00:00", "balance": 1000.0},
            {"timestamp": "2025-01-01T01:00:00", "balance": 1010.0},
            {"timestamp": "2025-01-01T02:00:00", "balance": 1000.0},
        ],
        bets=bets,
        clv_summary={"mean_clv": 0.05, "median_clv": 0.03, "clv_positive_rate": 0.6},
        calibration={"brier_score": 0.25, "log_loss": 0.5},
        warnings=["Test warning"],
        execution_time_ms=12.5,
    )


class TestJsonReport:
    def test_returns_dict(self):
        result = _make_result()
        report = generate_json_report("exp-1", result)
        assert isinstance(report, dict)

    def test_has_experiment_id(self):
        result = _make_result()
        report = generate_json_report("exp-1", result)
        assert report["experiment"]["id"] == "exp-1"

    def test_has_summary(self):
        result = _make_result()
        report = generate_json_report("exp-1", result)
        assert "summary" in report
        assert report["summary"]["total_bets"] == 2

    def test_has_metrics(self):
        result = _make_result()
        report = generate_json_report("exp-1", result)
        assert "metrics" in report
        assert "performance" in report["metrics"]
        assert "clv" in report["metrics"]

    def test_has_warnings(self):
        result = _make_result()
        report = generate_json_report("exp-1", result)
        assert "warnings" in report
        assert len(report["warnings"]) == 1

    def test_has_bankroll_history(self):
        result = _make_result()
        report = generate_json_report("exp-1", result)
        assert "bankroll_history" in report
        assert report["bankroll_history"]["total_snapshots"] == 3


class TestMarkdownReport:
    def test_returns_string(self):
        result = _make_result()
        report = generate_markdown_report("exp-1", result)
        assert isinstance(report, str)

    def test_has_headers(self):
        result = _make_result()
        report = generate_markdown_report("exp-1", result)
        assert "# Wippa Bet Lab" in report
        assert "Executive Summary" in report

    def test_has_experiment_id(self):
        result = _make_result()
        report = generate_markdown_report("exp-1", result)
        assert "exp-1" in report

    def test_has_metrics(self):
        result = _make_result()
        report = generate_markdown_report("exp-1", result)
        assert "Performance" in report
        assert "CLV" in report

    def test_has_warnings(self):
        result = _make_result()
        report = generate_markdown_report("exp-1", result)
        assert "Test warning" in report


class TestHtmlReport:
    def test_returns_string(self):
        result = _make_result()
        report = generate_html_report("exp-1", result)
        assert isinstance(report, str)

    def test_is_valid_html(self):
        result = _make_result()
        report = generate_html_report("exp-1", result)
        assert "<!DOCTYPE html>" in report or "<html" in report

    def test_has_experiment_id(self):
        result = _make_result()
        report = generate_html_report("exp-1", result)
        assert "exp-1" in report

    def test_has_svg_charts(self):
        result = _make_result()
        report = generate_html_report("exp-1", result)
        assert "<svg" in report

    def test_has_inline_css(self):
        result = _make_result()
        report = generate_html_report("exp-1", result)
        assert "<style>" in report or "style=" in report


class TestComparator:
    def test_compare_two_strategies(self):
        bt1 = BacktestResult(
            experiment_id="e1", total_bets=10, wins=6, losses=4,
            net_profit=20.0, roi=0.02, yield_pct=2.0, max_drawdown=0.05,
        )
        bt2 = BacktestResult(
            experiment_id="e2", total_bets=15, wins=8, losses=7,
            net_profit=10.0, roi=0.0067, yield_pct=0.67, max_drawdown=0.08,
        )
        report = compare_strategies([
            ("s1", "Strategy A", bt1, {"roi": 0.02, "max_drawdown": 0.05}),
            ("s2", "Strategy B", bt2, {"roi": 0.0067, "max_drawdown": 0.08}),
        ])
        assert len(report.strategies) == 2
        assert len(report.head_to_head_metrics) > 0
        assert report.recommendation != ""

    def test_single_strategy(self):
        bt1 = BacktestResult(experiment_id="e1", total_bets=10, wins=6, net_profit=20.0, roi=0.02)
        report = compare_strategies([
            ("s1", "Strategy A", bt1, {"roi": 0.02}),
        ])
        assert len(report.strategies) == 1
        assert report.recommendation != ""

    def test_empty_strategies(self):
        report = compare_strategies([])
        assert len(report.strategies) == 0
        assert "No strategies" in report.recommendation
