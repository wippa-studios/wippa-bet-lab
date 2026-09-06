from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.core.schemas.models import (
    StrategyDefinition,
    SignalCondition,
    PriceConfig,
    ValidationConfig,
)
from betlab.validation.walk_forward import WalkForwardValidator, WalkForwardResult, WindowResult
from betlab.validation.leakage import LeakageDetector, LeakageReport
from betlab.validation.splitter import DataSplitter


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_events(n: int = 20) -> list[dict]:
    events = []
    for i in range(n):
        events.append({
            "id": f"evt-{i}",
            "sport": "tennis",
            "start_time": f"2025-01-{i + 1:02d}T12:00:00",
            "markets": [
                {
                    "id": f"mkt-{i}",
                    "market_type": "match_odds",
                    "runners": [
                        {
                            "id": f"runner-{i}",
                            "name": f"Player {i}",
                            "prices": [
                                {
                                    "timestamp": f"2025-01-{i + 1:02d}T12:00:00",
                                    "runner_id": f"runner-{i}",
                                    "back_price": 2.0 + (i % 3) * 0.5,
                                    "lay_price": 2.1 + (i % 3) * 0.5,
                                    "back_size": 500.0,
                                    "lay_size": 500.0,
                                }
                            ],
                        }
                    ],
                }
            ],
        })
    return events


def _write_dataset(tmp_path: Path, events: list[dict]) -> Path:
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(events))
    return p


def _strategy() -> StrategyDefinition:
    return StrategyDefinition(
        id="test",
        version="1.0.0",
        name="Test",
        sport="tennis",
        market="match_odds",
        signal=SignalCondition(field="back_price", operator=">", value=0),
        price=PriceConfig(source="back_price", minimum=1.0, maximum=100.0),
    )


# ── Walk-forward tests ───────────────────────────────────────────────────────

class TestWalkForward:
    def test_creates_correct_number_of_windows(self, tmp_path):
        events = _make_events(20)
        ds = _write_dataset(tmp_path, events)
        config = ValidationConfig(
            training_window=8,
            validation_window=4,
            step=4,
            purge_gap=0,
        )
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) == 3

    def test_respects_purge_gap(self, tmp_path):
        events = _make_events(30)
        ds = _write_dataset(tmp_path, events)
        config = ValidationConfig(
            training_window=10,
            validation_window=5,
            step=5,
            purge_gap=2,
        )
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        for w in result.windows:
            assert w.test_start > w.train_end

    def test_windows_have_sequential_ids(self, tmp_path):
        events = _make_events(20)
        ds = _write_dataset(tmp_path, events)
        config = ValidationConfig(training_window=8, validation_window=4, step=4)
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        ids = [w.window_id for w in result.windows]
        assert ids == list(range(len(ids)))

    def test_stability_score_between_0_and_1(self, tmp_path):
        events = _make_events(20)
        ds = _write_dataset(tmp_path, events)
        config = ValidationConfig(training_window=8, validation_window=4, step=4)
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        assert 0.0 <= result.stability_score <= 1.0

    def test_overall_metrics_populated(self, tmp_path):
        events = _make_events(20)
        ds = _write_dataset(tmp_path, events)
        config = ValidationConfig(training_window=8, validation_window=4, step=4)
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        assert result.overall_metrics["windows_count"] == len(result.windows)

    def test_minimum_training_events_skips(self, tmp_path):
        events = _make_events(20)
        ds = _write_dataset(tmp_path, events)
        config = ValidationConfig(
            training_window=8,
            validation_window=4,
            step=4,
            minimum_training_events=100,
        )
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        assert len(result.windows) == 0
        assert len(result.warnings) > 0

    def test_empty_dataset(self, tmp_path):
        ds = _write_dataset(tmp_path, [])
        config = ValidationConfig(training_window=5, validation_window=2)
        wf = WalkForwardValidator(strategy=_strategy(), dataset_path=ds, config=config)
        result = wf.run()
        assert len(result.windows) == 0


# ── Leakage detector tests ───────────────────────────────────────────────────

class TestLeakageDetector:
    def test_clean_dataset(self, tmp_path):
        events = _make_events(10)
        ds = _write_dataset(tmp_path, events)
        det = LeakageDetector()
        report = det.detect(strategy=None, dataset_path=ds)
        assert isinstance(report, LeakageReport)

    def test_catches_duplicate_events(self, tmp_path):
        events = _make_events(5)
        events.append(events[0].copy())
        ds = _write_dataset(tmp_path, events)
        det = LeakageDetector()
        report = det.detect(strategy=None, dataset_path=ds)
        assert any("Duplicate" in w for w in report.warnings)

    def test_future_data_detection(self, tmp_path):
        events = _make_events(3)
        events[0]["markets"][0]["runners"][0]["result"] = "winner"
        prices = events[0]["markets"][0]["runners"][0]["prices"]
        events[0]["markets"][0]["runners"][0]["prices"] = [
            {**prices[0], "back_price": 1.02},
            {**prices[0], "back_price": 1.02},
        ]
        ds = _write_dataset(tmp_path, events)
        det = LeakageDetector()
        report = det.detect(strategy=None, dataset_path=ds)
        assert any("future data" in w.lower() for w in report.warnings)

    def test_severity_scaling(self, tmp_path):
        events = _make_events(10)
        events.append(events[0].copy())
        events.append(events[1].copy())
        ds = _write_dataset(tmp_path, events)
        det = LeakageDetector()
        report = det.detect(strategy=None, dataset_path=ds)
        assert report.severity in ("low", "medium", "high")

    def test_missing_dataset(self, tmp_path):
        det = LeakageDetector()
        report = det.detect(strategy=None, dataset_path=tmp_path / "nonexistent.json")
        assert not report.is_clean


# ── Splitter tests ───────────────────────────────────────────────────────────

class TestDataSplitter:
    def test_split_by_time(self):
        splitter = DataSplitter()
        events = _make_events(10)
        train, test = splitter.split_by_time(events, train_end="2025-01-05T12:00:00", test_start="2025-01-06T12:00:00")
        assert len(train) == 5
        assert len(test) == 5

    def test_split_by_ratio(self):
        splitter = DataSplitter()
        events = _make_events(10)
        train, test = splitter.split_by_ratio(events, train_ratio=0.7)
        assert len(train) == 7
        assert len(test) == 3

    def test_k_fold_split(self):
        splitter = DataSplitter()
        events = _make_events(20)
        folds = splitter.k_fold_split(events, k=5)
        assert len(folds) == 5
        for train, test in folds:
            assert len(train) + len(test) == 20

    def test_k_fold_covers_all_data(self):
        splitter = DataSplitter()
        events = _make_events(15)
        folds = splitter.k_fold_split(events, k=3)
        all_test_ids = set()
        for _, test in folds:
            for e in test:
                all_test_ids.add(e["id"])
        assert len(all_test_ids) == 15

    def test_split_ratio_sum(self):
        splitter = DataSplitter()
        events = _make_events(10)
        train, test = splitter.split_by_ratio(events, train_ratio=0.6)
        assert len(train) + len(test) == 10
