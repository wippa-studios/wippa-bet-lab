from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from betlab.core.schemas.models import StrategyDefinition, ValidationConfig
from betlab.backtester.engine import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class WindowResult(BaseModel):
    window_id: int = 0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    train_bets: int = 0
    test_bets: int = 0
    train_roi: float = 0.0
    test_roi: float = 0.0
    train_drawdown: float = 0.0
    test_drawdown: float = 0.0


class WalkForwardResult(BaseModel):
    windows: list[WindowResult] = Field(default_factory=list)
    overall_metrics: dict[str, Any] = Field(default_factory=dict)
    stability_score: float = 0.0
    parameter_stability: dict[str, Any] = Field(default_factory=dict)
    feature_drift: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Rolling walk-forward validation with training/validation windows and purge gap."""

    def __init__(
        self,
        strategy: StrategyDefinition,
        dataset_path: str | Path,
        config: ValidationConfig | None = None,
    ):
        self.strategy = strategy
        self.dataset_path = Path(dataset_path)
        self.config = config or ValidationConfig()

    def run(self) -> WalkForwardResult:
        events = self._load_events()
        if not events:
            return WalkForwardResult(warnings=["No events loaded from dataset"])

        events_sorted = sorted(events, key=lambda e: e.get("timestamp", e.get("start_time", "")))
        total = len(events_sorted)
        cfg = self.config

        train_win = cfg.training_window or max(total // 4, 1)
        val_win = cfg.validation_window or max(total // 8, 1)
        step = cfg.step or 1
        purge = cfg.purge_gap
        min_train = cfg.minimum_training_events or 0

        windows: list[WindowResult] = []
        warnings: list[str] = []
        idx = 0
        window_id = 0

        while idx + train_win + purge + val_win <= total:
            train_start_idx = idx
            train_end_idx = idx + train_win
            test_start_idx = train_end_idx + purge
            test_end_idx = test_start_idx + val_win

            train_events = events_sorted[train_start_idx:train_end_idx]
            test_events = events_sorted[test_start_idx:test_end_idx]

            if len(train_events) < min_train:
                warnings.append(
                    f"Window {window_id}: training set has {len(train_events)} events "
                    f"< minimum {min_train}, skipping"
                )
                idx += step
                continue

            train_result = self._run_backtest(train_events)
            test_result = self._run_backtest(test_events)

            wr = WindowResult(
                window_id=window_id,
                train_start=self._extract_ts(train_events[0]),
                train_end=self._extract_ts(train_events[-1]),
                test_start=self._extract_ts(test_events[0]),
                test_end=self._extract_ts(test_events[-1]),
                train_bets=train_result.total_bets,
                test_bets=test_result.total_bets,
                train_roi=train_result.roi,
                test_roi=test_result.roi,
                train_drawdown=train_result.max_drawdown,
                test_drawdown=test_result.max_drawdown,
            )
            windows.append(wr)

            if test_result.warnings:
                for w in test_result.warnings:
                    warnings.append(f"Window {window_id}: {w}")

            idx += step
            window_id += 1

        if not windows:
            warnings.append("No valid windows could be created — dataset too small for config")

        overall = self._compute_overall(windows)
        stability = self._compute_stability(windows)
        param_stab = self._compute_parameter_stability(windows)

        return WalkForwardResult(
            windows=windows,
            overall_metrics=overall,
            stability_score=stability,
            parameter_stability=param_stab,
            feature_drift={},
            warnings=warnings,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _run_backtest(self, events: list[dict[str, Any]]) -> BacktestResult:
        engine = BacktestEngine(strategy=self.strategy, starting_bankroll=1000.0)
        return engine.run(events)

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            return []
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("events", [])

    @staticmethod
    def _extract_ts(event: dict[str, Any]) -> str:
        return str(event.get("timestamp", event.get("start_time", "")))

    def _compute_overall(self, windows: list[WindowResult]) -> dict[str, Any]:
        if not windows:
            return {}
        test_rois = [w.test_roi for w in windows]
        test_bets = [w.test_bets for w in windows]
        total_test_bets = sum(test_bets)
        avg_roi = sum(test_rois) / len(test_rois) if test_rois else 0.0
        return {
            "avg_test_roi": avg_roi,
            "total_test_bets": total_test_bets,
            "windows_count": len(windows),
            "positive_windows": sum(1 for r in test_rois if r > 0),
            "negative_windows": sum(1 for r in test_rois if r <= 0),
        }

    def _compute_stability(self, windows: list[WindowResult]) -> float:
        if len(windows) < 2:
            return 1.0
        test_rois = [w.test_roi for w in windows]
        mean_roi = sum(test_rois) / len(test_rois)
        variance = sum((r - mean_roi) ** 2 for r in test_rois) / len(test_rois)
        std_dev = variance ** 0.5
        if abs(mean_roi) < 1e-9:
            return 0.0 if std_dev > 0.001 else 1.0
        cv = std_dev / abs(mean_roi)
        return max(0.0, 1.0 - min(cv, 2.0) / 2.0)

    def _compute_parameter_stability(self, windows: list[WindowResult]) -> dict[str, Any]:
        if not windows:
            return {}
        train_rois = [w.train_roi for w in windows]
        test_rois = [w.test_roi for w in windows]
        correlation = 0.0
        if len(train_rois) >= 2:
            mean_t = sum(train_rois) / len(train_rois)
            mean_v = sum(test_rois) / len(test_rois)
            cov = sum((t - mean_t) * (v - mean_v) for t, v in zip(train_rois, test_rois)) / len(train_rois)
            std_t = (sum((t - mean_t) ** 2 for t in train_rois) / len(train_rois)) ** 0.5
            std_v = (sum((v - mean_v) ** 2 for v in test_rois) / len(test_rois)) ** 0.5
            if std_t > 0 and std_v > 0:
                correlation = cov / (std_t * std_v)
        return {
            "train_test_correlation": correlation,
            "train_roi_range": (min(train_rois), max(train_rois)) if train_rois else (0, 0),
            "test_roi_range": (min(test_rois), max(test_rois)) if test_rois else (0, 0),
        }
