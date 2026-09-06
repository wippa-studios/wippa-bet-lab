from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.core.schemas.models import (
    BankrollSnapshot,
    BetSide,
    BetStatus,
    CommissionConfig,
    ExecutionConfig,
    FilterCondition,
    RiskConfig,
    SignalCondition,
    SimulatedBet,
    SlippageConfig,
    StakingConfig,
    StakingMethod,
    StrategyDefinition,
)
from betlab.backtester.engine import BacktestEngine, BacktestResult
from betlab.simulator.market import MarketSimulator
from betlab.simulator.bankroll import BankrollManager
from betlab.metrics.performance import compute_performance, PerformanceMetrics
from betlab.metrics.clv import compute_clv, compute_bet_clv, CLVReport
from betlab.metrics.calibration import compute_calibration, CalibrationReport
from betlab.metrics.drawdown import compute_drawdown, DrawdownReport
from betlab.metrics.monte_carlo import monte_carlo, MonteCarloReport


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _sample_events():
    return [
        {
            "id": "evt-1",
            "name": "Match 1",
            "sport": "tennis",
            "start_time": "2025-01-01T12:00:00",
            "markets": [
                {
                    "id": "mkt-1",
                    "market_type": "match_odds",
                    "runners": [
                        {
                            "id": "runner-1",
                            "name": "Player A",
                            "prices": [
                                {
                                    "timestamp": "2025-01-01T12:00:00",
                                    "runner_id": "runner-1",
                                    "back_price": 2.0,
                                    "lay_price": 2.1,
                                    "back_size": 500.0,
                                    "lay_size": 500.0,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "evt-2",
            "name": "Match 2",
            "sport": "tennis",
            "start_time": "2025-01-01T13:00:00",
            "markets": [
                {
                    "id": "mkt-2",
                    "market_type": "match_odds",
                    "runners": [
                        {
                            "id": "runner-2",
                            "name": "Player B",
                            "prices": [
                                {
                                    "timestamp": "2025-01-01T13:00:00",
                                    "runner_id": "runner-2",
                                    "back_price": 3.0,
                                    "lay_price": 3.1,
                                    "back_size": 200.0,
                                    "lay_size": 200.0,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    ]


def _sample_strategy():
    from betlab.core.schemas.models import PriceConfig
    return StrategyDefinition(
        id="test",
        version="1.0.0",
        name="Test",
        sport="tennis",
        market="match_odds",
        filters=[],
        signal=SignalCondition(field="back_price", operator=">", value=0),
        price=PriceConfig(source="back_price", minimum=1.0, maximum=100.0),
    )


def _make_bets():
    return [
        SimulatedBet(
            id="b1", experiment_id="exp-1", event_id="e1", market_id="m1", runner_id="r1",
            side=BetSide.back, requested_price=2.0, matched_price=2.0, stake=10.0,
            result=BetStatus.settled, profit=10.0,
            clv_odds=1.8, clv_implied_probability=0.5,
        ),
        SimulatedBet(
            id="b2", experiment_id="exp-1", event_id="e1", market_id="m1", runner_id="r1",
            side=BetSide.back, requested_price=3.0, matched_price=3.0, stake=10.0,
            result=BetStatus.settled, profit=-10.0,
            clv_odds=2.8, clv_implied_probability=0.33,
        ),
        SimulatedBet(
            id="b3", experiment_id="exp-1", event_id="e1", market_id="m1", runner_id="r1",
            side=BetSide.back, requested_price=2.5, matched_price=2.5, stake=10.0,
            result=BetStatus.settled, profit=15.0,
            clv_odds=2.2, clv_implied_probability=0.4,
        ),
        SimulatedBet(
            id="b4", experiment_id="exp-1", event_id="e1", market_id="m1", runner_id="r1",
            side=BetSide.back, requested_price=4.0, matched_price=4.0, stake=10.0,
            result=BetStatus.settled, profit=-10.0,
            clv_odds=3.5, clv_implied_probability=0.25,
        ),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBacktestEngine:
    def test_engine_runs_without_errors(self):
        engine = BacktestEngine(
            strategy=_sample_strategy(),
            starting_bankroll=1000.0,
        )
        result = engine.run(_sample_events())
        assert isinstance(result, BacktestResult)
        assert result.experiment_id
        assert result.execution_time_ms >= 0

    def test_engine_records_bets(self):
        engine = BacktestEngine(
            strategy=_sample_strategy(),
            starting_bankroll=1000.0,
        )
        result = engine.run(_sample_events())
        assert len(result.bets) == 2


class TestMarketSimulator:
    def test_commission_percent(self):
        sim = MarketSimulator(commission_config=CommissionConfig(
            type="percent_of_profit", rate=0.05
        ))
        comm = sim.compute_commission(100.0)
        assert comm == pytest.approx(5.0)

    def test_commission_none(self):
        sim = MarketSimulator()
        comm = sim.compute_commission(100.0)
        assert comm == 0.0

    def test_slippage_fixed(self):
        sim = MarketSimulator(slippage_config=SlippageConfig(
            mode="fixed", back=0.02, lay=0.02
        ))
        assert sim.apply_slippage(2.0, BetSide.back) == 0.02

    def test_slippage_percent(self):
        sim = MarketSimulator(slippage_config=SlippageConfig(
            mode="percent", back=0.01, lay=0.01
        ))
        assert sim.apply_slippage(2.0, BetSide.back) == pytest.approx(0.02)

    def test_slippage_ticks_default(self):
        sim = MarketSimulator()
        assert sim.apply_slippage(2.0, BetSide.back) == 0.0

    def test_liquidity_full_fill(self):
        sim = MarketSimulator(execution_config=ExecutionConfig(partial_fills=False))
        snap = {"back_available": 500.0, "lay_available": 500.0}
        avail, filled = sim.check_liquidity(snap, BetSide.back, 100.0)
        assert filled == 100.0

    def test_liquidity_no_fill(self):
        sim = MarketSimulator(execution_config=ExecutionConfig(partial_fills=False))
        snap = {"back_available": 50.0, "lay_available": 50.0}
        avail, filled = sim.check_liquidity(snap, BetSide.back, 100.0)
        assert filled == 0.0

    def test_liquidity_partial_fill(self):
        sim = MarketSimulator(execution_config=ExecutionConfig(partial_fills=True))
        snap = {"back_available": 50.0, "lay_available": 50.0}
        avail, filled = sim.check_liquidity(snap, BetSide.back, 100.0)
        assert filled == 50.0


class TestBankrollManager:
    def test_kelly_stake(self):
        bm = BankrollManager(starting_balance=1000.0)
        stake = bm.kelly_stake(probability=0.6, odds=2.0, fraction=1.0)
        assert stake == pytest.approx(200.0)

    def test_kelly_stake_fractional(self):
        bm = BankrollManager(starting_balance=1000.0)
        stake = bm.kelly_stake(probability=0.6, odds=2.0, fraction=0.25)
        assert stake == pytest.approx(50.0)

    def test_kelly_stake_negative_edge(self):
        bm = BankrollManager(starting_balance=1000.0)
        stake = bm.kelly_stake(probability=0.3, odds=2.0)
        assert stake == 0.0

    def test_fixed_stake(self):
        cfg = StakingConfig(method=StakingMethod.fixed_stake, maximum_stake_absolute=50.0)
        bm = BankrollManager(starting_balance=1000.0, staking_config=cfg)
        stake = bm.compute_stake(0.5, 2.0)
        assert stake == 50.0

    def test_fixed_percent(self):
        cfg = StakingConfig(method=StakingMethod.fixed_percent, fraction=0.05)
        bm = BankrollManager(starting_balance=1000.0, staking_config=cfg)
        stake = bm.compute_stake(0.5, 2.0)
        assert stake == pytest.approx(50.0)

    def test_update_balance(self):
        bm = BankrollManager(starting_balance=1000.0)
        new_bal = bm.update_balance(50.0)
        assert new_bal == 1050.0
        assert bm.bets_count == 1

    def test_risk_limits_ok(self):
        bm = BankrollManager(
            starting_balance=1000.0,
            risk_config=RiskConfig(maximum_open_exposure=0.2, maximum_daily_loss=0.1),
        )
        assert bm.check_risk_limits() is True

    def test_risk_limits_breach(self):
        bm = BankrollManager(
            starting_balance=1000.0,
            risk_config=RiskConfig(maximum_daily_loss=0.05),
        )
        assert bm.check_risk_limits(daily_loss=-60.0) is False

    def test_record_snapshot(self):
        bm = BankrollManager(starting_balance=1000.0)
        snap = bm.record_snapshot()
        assert snap.balance == 1000.0
        assert len(bm.snapshots) == 1


class TestPerformanceMetrics:
    def test_roi(self):
        bets = _make_bets()
        perf = compute_performance(bets, starting_bankroll=1000.0)
        assert perf.total_bets == 4
        assert perf.wins == 2
        assert perf.losses == 2
        assert perf.net_profit == pytest.approx(5.0)
        assert perf.roi == pytest.approx(0.5)

    def test_yield(self):
        bets = _make_bets()
        perf = compute_performance(bets, starting_bankroll=1000.0)
        assert perf.yield_pct == pytest.approx(5.0 / 40.0 * 100)

    def test_strike_rate(self):
        bets = _make_bets()
        perf = compute_performance(bets, starting_bankroll=1000.0)
        assert perf.strike_rate == pytest.approx(0.5)

    def test_profit_factor(self):
        bets = _make_bets()
        perf = compute_performance(bets, starting_bankroll=1000.0)
        assert perf.profit_factor == pytest.approx(25.0 / 20.0)

    def test_empty_bets(self):
        perf = compute_performance([])
        assert perf.total_bets == 0
        assert perf.roi == 0.0


class TestCLV:
    def test_bet_clv_positive(self):
        clv = compute_bet_clv(placed_odds=2.0, closing_odds=1.8)
        assert clv == pytest.approx(11.11, abs=0.1)

    def test_bet_clv_negative(self):
        clv = compute_bet_clv(placed_odds=2.0, closing_odds=2.5)
        assert clv < 0

    def test_compute_clv(self):
        bets = _make_bets()
        report = compute_clv(bets)
        assert isinstance(report, CLVReport)
        assert report.mean_clv != 0.0

    def test_clv_empty(self):
        report = compute_clv([])
        assert report.mean_clv == 0.0


class TestCalibration:
    def test_brier_score(self):
        bets = _make_bets()
        report = compute_calibration(bets)
        expected = (0.25 + 0.1089 + 0.36 + 0.0625) / 4
        assert report.brier_score == pytest.approx(expected, abs=0.01)

    def test_buckets(self):
        bets = _make_bets()
        report = compute_calibration(bets)
        assert len(report.calibration_by_bucket) == 10

    def test_empty(self):
        report = compute_calibration([])
        assert report.brier_score == 0.0


class TestDrawdown:
    def test_drawdown(self):
        from datetime import datetime, timezone
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        history = [
            BankrollSnapshot(id="1", experiment_id="e", timestamp=ts, balance=1000.0),
            BankrollSnapshot(id="2", experiment_id="e", timestamp=ts, balance=900.0),
            BankrollSnapshot(id="3", experiment_id="e", timestamp=ts, balance=950.0),
            BankrollSnapshot(id="4", experiment_id="e", timestamp=ts, balance=1050.0),
        ]
        report = compute_drawdown(history)
        assert report.max_drawdown == pytest.approx(0.1)

    def test_no_drawdown(self):
        from datetime import datetime, timezone
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        history = [
            BankrollSnapshot(id="1", experiment_id="e", timestamp=ts, balance=1000.0),
            BankrollSnapshot(id="2", experiment_id="e", timestamp=ts, balance=1050.0),
        ]
        report = compute_drawdown(history)
        assert report.max_drawdown == 0.0


class TestMonteCarlo:
    def test_runs(self):
        bets = _make_bets()
        report = monte_carlo(bets, n_simulations=100, starting_bankroll=1000.0, seed=42)
        assert isinstance(report, MonteCarloReport)
        assert 5 in report.percentiles
        assert 95 in report.percentiles
        assert 0.0 <= report.probability_of_ruin <= 1.0
        assert report.expected_terminal > 0

    def test_empty(self):
        report = monte_carlo([], n_simulations=100)
        assert report.expected_terminal == 0.0
