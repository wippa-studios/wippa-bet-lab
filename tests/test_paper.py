from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.core.schemas.models import (
    BetSide,
    BetStatus,
    PaperPosition,
    RiskConfig,
    StrategyDefinition,
    SignalCondition,
    PriceConfig,
)
from betlab.paper.session import PaperTrader
from betlab.paper.risk import RiskManager, RiskCheck


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _event(event_id: str = "evt-1", back_price: float = 2.5) -> dict:
    return {
        "id": event_id,
        "sport": "tennis",
        "start_time": "2025-01-01T12:00:00",
        "markets": [
            {
                "id": f"mkt-{event_id}",
                "market_type": "match_odds",
                "runners": [
                    {
                        "id": f"runner-{event_id}",
                        "name": "Player A",
                        "prices": [
                            {
                                "timestamp": "2025-01-01T12:00:00",
                                "runner_id": f"runner-{event_id}",
                                "back_price": back_price,
                                "lay_price": back_price + 0.1,
                                "back_size": 500.0,
                                "lay_size": 500.0,
                            }
                        ],
                    }
                ],
            }
        ],
    }


# ── Paper session tests ─────────────────────────────────────────────────────

class TestPaperSession:
    def test_start_creates_session(self):
        trader = PaperTrader(strategy=_strategy(), session_config={"starting_bankroll": 500.0})
        session = trader.start()
        assert session.id
        assert session.current_bankroll == 500.0
        assert session.status.value == "active"

    def test_stop(self):
        trader = PaperTrader(strategy=_strategy())
        trader.start()
        trader.stop()
        assert trader.get_status()["status"] == "stopped"

    def test_pause_and_resume(self):
        trader = PaperTrader(strategy=_strategy())
        trader.start()
        trader.pause()
        assert trader.get_status()["status"] == "paused"
        trader.resume()
        assert trader.get_status()["status"] == "active"

    def test_process_event_creates_position(self):
        trader = PaperTrader(strategy=_strategy(), session_config={"stake": 10.0})
        trader.start()
        pos = trader.process_event(_event())
        assert pos is not None
        assert pos.stake == 10.0

    def test_process_event_rejects_when_paused(self):
        trader = PaperTrader(strategy=_strategy())
        trader.start()
        trader.pause()
        pos = trader.process_event(_event())
        assert pos is None

    def test_get_positions_open(self):
        trader = PaperTrader(strategy=_strategy(), session_config={"stake": 10.0})
        trader.start()
        trader.process_event(_event("e1"))
        trader.process_event(_event("e2"))
        assert len(trader.get_positions()) == 2

    def test_settle_winning_bet(self):
        trader = PaperTrader(strategy=_strategy(), session_config={"starting_bankroll": 1000.0, "stake": 10.0})
        trader.start()
        trader.process_event(_event("e1", back_price=2.0))
        settled = trader.settle("e1", winner="runner-e1")
        assert len(settled) == 1
        assert settled[0].profit > 0
        assert trader.get_status()["pnl"] > 0

    def test_settle_losing_bet(self):
        trader = PaperTrader(strategy=_strategy(), session_config={"starting_bankroll": 1000.0, "stake": 10.0})
        trader.start()
        trader.process_event(_event("e1", back_price=2.0))
        settled = trader.settle("e1", winner="other-runner")
        assert len(settled) == 1
        assert settled[0].profit == -10.0

    def test_history_after_settle(self):
        trader = PaperTrader(strategy=_strategy(), session_config={"stake": 10.0})
        trader.start()
        trader.process_event(_event("e1"))
        trader.settle("e1", winner="runner-e1")
        assert len(trader.get_history()) == 1
        assert len(trader.get_positions()) == 0


# ── Risk manager tests ───────────────────────────────────────────────────────

class TestRiskManager:
    def test_allows_normal_bet(self):
        rm = RiskManager(config=RiskConfig(maximum_open_exposure=0.5))
        check = rm.checkLimits(
            positions=[],
            new_bet={"stake": 10},
            current_bankroll=1000.0,
            starting_bankroll=1000.0,
        )
        assert check.allowed is True

    def test_blocks_high_exposure(self):
        rm = RiskManager(config=RiskConfig(maximum_open_exposure=0.1))
        pos = PaperPosition(
            id="p1", session_id="s1", event_id="e1", market_id="m1",
            runner_id="r1", side=BetSide.back, stake=150.0, price=2.0,
            status=BetStatus.matched,
        )
        check = rm.checkLimits(
            positions=[pos],
            new_bet={"stake": 10},
            current_bankroll=1000.0,
            starting_bankroll=1000.0,
        )
        assert check.allowed is False
        assert "exposure" in check.reason.lower()

    def test_blocks_high_daily_loss(self):
        rm = RiskManager(config=RiskConfig(maximum_daily_loss=0.05))
        from datetime import datetime, timezone
        pos = PaperPosition(
            id="p1", session_id="s1", event_id="e1", market_id="m1",
            runner_id="r1", side=BetSide.back, stake=100.0, price=2.0,
            status=BetStatus.settled, profit=-60.0,
            settled_at=datetime.now(timezone.utc),
        )
        check = rm.checkLimits(
            positions=[pos],
            new_bet={"stake": 10},
            current_bankroll=1000.0,
            starting_bankroll=1000.0,
        )
        assert check.allowed is False
        assert "daily loss" in check.reason.lower()

    def test_blocks_consecutive_losses(self):
        rm = RiskManager()
        check = rm.checkLimits(
            positions=[],
            new_bet={"stake": 10},
            current_bankroll=1000.0,
            starting_bankroll=1000.0,
            consecutive_losses=5,
        )
        assert check.allowed is False

    def test_compute_open_exposure(self):
        rm = RiskManager()
        pos1 = PaperPosition(
            id="p1", session_id="s1", event_id="e1", market_id="m1",
            runner_id="r1", side=BetSide.back, stake=100.0, price=2.0,
            status=BetStatus.matched,
        )
        pos2 = PaperPosition(
            id="p2", session_id="s1", event_id="e2", market_id="m2",
            runner_id="r2", side=BetSide.back, stake=50.0, price=3.0,
            status=BetStatus.pending,
        )
        assert rm.compute_open_exposure([pos1, pos2]) == 150.0

    def test_compute_daily_loss(self):
        from datetime import datetime, timezone, date
        rm = RiskManager()
        pos = PaperPosition(
            id="p1", session_id="s1", event_id="e1", market_id="m1",
            runner_id="r1", side=BetSide.back, stake=100.0, price=2.0,
            status=BetStatus.settled, profit=-25.0,
            settled_at=datetime.now(timezone.utc),
        )
        loss = rm.compute_daily_loss([pos], date.today())
        assert loss == -25.0

    def test_risk_blocks_triggers_pause(self):
        trader = PaperTrader(
            strategy=_strategy(),
            session_config={
                "starting_bankroll": 100.0,
                "stake": 60.0,
                "risk": {"maximum_open_exposure": 0.5},
            },
        )
        trader.start()
        trader.process_event(_event("e1"))
        pos = trader.process_event(_event("e2"))
        assert pos is None
        assert trader.get_status()["status"] == "paused"
