from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from betlab.core.schemas.models import (
    BetSide,
    BetStatus,
    CommissionConfig,
    ExecutionConfig,
    FilterCondition,
    RiskConfig,
    SimulatedBet,
    SlippageConfig,
    StakingConfig,
    StrategyDefinition,
)
from betlab.simulator.market import MarketSimulator
from betlab.simulator.bankroll import BankrollManager
from betlab.metrics.performance import compute_performance, PerformanceMetrics
from betlab.metrics.clv import compute_clv, CLVReport
from betlab.metrics.calibration import compute_calibration, CalibrationReport
from betlab.metrics.drawdown import compute_drawdown, DrawdownReport


@dataclass
class BacktestResult:
    experiment_id: str = ""
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    gross_profit: float = 0.0
    net_profit: float = 0.0
    roi: float = 0.0
    yield_pct: float = 0.0
    max_drawdown: float = 0.0
    average_drawdown: float = 0.0
    bankroll_history: list[dict[str, Any]] = field(default_factory=list)
    bets: list[SimulatedBet] = field(default_factory=list)
    clv_summary: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class BacktestEngine:
    """Event-driven backtesting engine that processes data chronologically."""

    def __init__(
        self,
        strategy: StrategyDefinition | None = None,
        dataset_path: str | Path | None = None,
        commission_config: CommissionConfig | None = None,
        slippage_config: SlippageConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        staking_config: StakingConfig | None = None,
        risk_config: RiskConfig | None = None,
        starting_bankroll: float = 1000.0,
        seed: int | None = None,
    ):
        self.strategy = strategy or StrategyDefinition(
            id="default", version="1.0.0", name="", sport="tennis", market="match_odds"
        )
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.starting_bankroll = starting_bankroll
        self.seed = seed

        self.simulator = MarketSimulator(
            commission_config=commission_config,
            slippage_config=slippage_config,
            execution_config=execution_config,
        )
        self.bankroll = BankrollManager(
            starting_balance=starting_bankroll,
            staking_config=staking_config,
            risk_config=risk_config,
        )

        self._audit_trail: list[dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, events: list[dict[str, Any]] | None = None) -> BacktestResult:
        start = time.perf_counter()
        experiment_id = str(uuid.uuid4())

        if events is None:
            events = self._load_dataset()

        events_sorted = sorted(events, key=lambda e: e.get("timestamp", e.get("start_time", "")))

        bets: list[SimulatedBet] = []
        warnings: list[str] = []

        for event in events_sorted:
            for market in event.get("markets", []):
                for runner in market.get("runners", []):
                    for price_point in runner.get("prices", []):
                        snapshot = {
                            "back_price": price_point.get("back_price", 0.0),
                            "lay_price": price_point.get("lay_price", 0.0),
                            "back_available": price_point.get("back_available", price_point.get("back_size", 0.0)),
                            "lay_available": price_point.get("lay_available", price_point.get("lay_size", 0.0)),
                            "is_void": price_point.get("is_void", False),
                        }

                        candidate = self._evaluate_strategy(event, market, runner, price_point)
                        if candidate is None:
                            continue

                        if not self.bankroll.check_risk_limits():
                            warnings.append(
                                f"Risk limit breach at {price_point.get('timestamp', '?')} — skipped bet"
                            )
                            continue

                        stake = self.bankroll.compute_stake(
                            probability=candidate.get("predicted_probability", 0.0),
                            odds=candidate.get("requested_price", 1.0),
                        )
                        if stake <= 0:
                            continue

                        candidate["stake"] = stake
                        candidate["experiment_id"] = experiment_id
                        bet = self.simulator.simulate_fill(candidate, snapshot, self.bankroll.balance)

                        if bet.stake > 0:
                            bets.append(bet)
                            self._audit_event("bet_placed", bet)

        result = self._compile_result(experiment_id, bets, warnings)
        result.execution_time_ms = (time.perf_counter() - start) * 1000
        return result

    # ── Strategy evaluation ───────────────────────────────────────────────

    def _evaluate_strategy(
        self,
        event: dict[str, Any],
        market: dict[str, Any],
        runner: dict[str, Any],
        price_point: dict[str, Any],
    ) -> dict[str, Any] | None:
        ctx = {**event, **market, **runner, **price_point}

        for filt in self.strategy.filters:
            if not self._check_filter(ctx, filt):
                return None

        signal_field = self.strategy.signal.field
        signal_val = self._resolve_field(ctx, signal_field)
        if signal_val is None:
            return None
        signal_op = self.strategy.signal.operator
        signal_threshold = float(self.strategy.signal.value)
        if not self._check_operator(float(signal_val), signal_op, signal_threshold):
            return None

        price_src = self.strategy.price.source
        price_val = float(price_point.get(price_src, 0.0))
        if price_val < self.strategy.price.minimum or price_val > self.strategy.price.maximum:
            return None

        side = self.strategy.selection.side.value if hasattr(self.strategy.selection.side, "value") else str(self.strategy.selection.side)
        runner_id = runner.get("id", runner.get("runner_id", ""))

        predicted_prob = 1.0 / price_val if price_val > 0 else 0.0

        return {
            "side": side,
            "runner_id": runner_id,
            "event_id": event.get("id", ""),
            "market_id": market.get("id", ""),
            "requested_price": price_val,
            "predicted_probability": predicted_prob,
        }

    def _check_filter(self, ctx: dict[str, Any], filt: FilterCondition) -> bool:
        val = self._resolve_field(ctx, filt.field)
        if val is None:
            return False
        return self._check_operator(float(val), filt.operator, float(filt.value))

    @staticmethod
    def _check_operator(actual: float, op: str, expected: float) -> bool:
        if op == ">":
            return actual > expected
        if op == ">=":
            return actual >= expected
        if op == "<":
            return actual < expected
        if op == "<=":
            return actual <= expected
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        return True

    @staticmethod
    def _resolve_field(ctx: dict[str, Any], field_path: str) -> Any:
        parts = field_path.split(".")
        current = ctx
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    # ── Audit ─────────────────────────────────────────────────────────────

    def _audit_event(self, action: str, bet: SimulatedBet) -> None:
        self._audit_trail.append({
            "action": action,
            "bet_id": bet.id,
            "timestamp": bet.placed_at.isoformat() if hasattr(bet.placed_at, "isoformat") else str(bet.placed_at),
            "stake": bet.stake,
            "price": bet.matched_price,
            "side": bet.side.value if hasattr(bet.side, "value") else bet.side,
        })

    # ── Result compilation ────────────────────────────────────────────────

    def _compile_result(
        self,
        experiment_id: str,
        bets: list[SimulatedBet],
        warnings: list[str],
    ) -> BacktestResult:
        valid = [b for b in bets if b.result != "void"]
        wins = sum(1 for b in valid if b.result == "matched" and b.profit > 0)
        losses = sum(1 for b in valid if b.result == "matched" and b.profit <= 0)
        voids = sum(1 for b in bets if b.result == "void")
        gross_profit = sum(b.profit for b in valid if b.profit > 0)
        net_profit = sum(b.profit for b in valid)
        turnover = sum(b.stake for b in valid)

        perf = compute_performance(bets, self.bankroll.snapshots, self.starting_bankroll)
        clv = compute_clv(bets)
        cal = compute_calibration(bets)

        return BacktestResult(
            experiment_id=experiment_id,
            total_bets=len(bets),
            wins=wins,
            losses=losses,
            voids=voids,
            gross_profit=gross_profit,
            net_profit=net_profit,
            roi=perf.roi,
            yield_pct=perf.yield_pct,
            max_drawdown=perf.max_drawdown,
            average_drawdown=perf.average_drawdown,
            bankroll_history=[
                {"timestamp": s.timestamp.isoformat(), "balance": s.balance}
                for s in self.bankroll.snapshots
            ],
            bets=bets,
            clv_summary={
                "mean_clv": clv.mean_clv,
                "median_clv": clv.median_clv,
                "clv_positive_rate": clv.clv_positive_rate,
            },
            calibration={
                "brier_score": cal.brier_score,
                "log_loss": cal.log_loss,
            },
            warnings=warnings,
        )

    # ── Dataset loading ───────────────────────────────────────────────────

    def _load_dataset(self) -> list[dict[str, Any]]:
        if self.dataset_path is None or not self.dataset_path.exists():
            return []
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("events", [])
