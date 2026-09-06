from __future__ import annotations

import uuid
from datetime import datetime, timezone, date
from typing import Any

from betlab.core.schemas.models import (
    BetSide,
    BetStatus,
    PaperPosition,
    PaperSession,
    PaperSessionStatus,
    RiskConfig,
    StrategyDefinition,
)
from betlab.paper.risk import RiskManager, RiskCheck


class PaperTrader:
    """Simulated paper trading session with risk management."""

    def __init__(
        self,
        strategy: StrategyDefinition,
        session_config: dict[str, Any] | None = None,
        db_path: str | None = None,
    ):
        self.strategy = strategy
        self.config = session_config or {}
        self.db_path = db_path

        self._session: PaperSession | None = None
        self._positions: list[PaperPosition] = []
        self._risk = RiskManager(config=RiskConfig(**self.config.get("risk", {})))
        self._consecutive_losses = 0
        self._daily_pnl: dict[str, float] = {}

    def start(self) -> PaperSession:
        starting = self.config.get("starting_bankroll", 1000.0)
        self._session = PaperSession(
            id=str(uuid.uuid4()),
            strategy_id=self.strategy.id,
            strategy_version=self.strategy.version,
            starting_bankroll=starting,
            current_bankroll=starting,
            status=PaperSessionStatus.active,
            risk_limits=self.config.get("risk", {}),
        )
        return self._session

    def process_event(self, event: dict[str, Any]) -> PaperPosition | None:
        if not self._session or self._session.status != PaperSessionStatus.active:
            return None

        signal_val = self._evaluate_signal(event)
        if signal_val is None:
            return None

        stake = self.config.get("stake", 10.0)
        price = float(signal_val.get("price", 0))
        if price <= 1.0:
            return None

        bet = {
            "stake": stake,
            "price": price,
            "side": "back",
        }

        risk_check = self._risk.checkLimits(
            positions=self._positions,
            new_bet=bet,
            current_bankroll=self._session.current_bankroll,
            starting_bankroll=self._session.starting_bankroll,
            consecutive_losses=self._consecutive_losses,
        )

        if not risk_check.allowed:
            self._session.status = PaperSessionStatus.paused
            return None

        pos = PaperPosition(
            id=str(uuid.uuid4()),
            session_id=self._session.id,
            event_id=event.get("id", ""),
            market_id=signal_val.get("market_id", ""),
            runner_id=signal_val.get("runner_id", ""),
            side=BetSide.back,
            stake=stake,
            price=price,
            status=BetStatus.pending,
        )
        self._positions.append(pos)
        return pos

    def pause(self) -> None:
        if self._session:
            self._session.status = PaperSessionStatus.paused

    def resume(self) -> None:
        if self._session and self._session.status == PaperSessionStatus.paused:
            self._session.status = PaperSessionStatus.active

    def stop(self) -> None:
        if self._session:
            self._session.status = PaperSessionStatus.stopped

    def settle(self, event_id: str, winner: str) -> list[PaperPosition]:
        settled = []
        for pos in self._positions:
            if pos.event_id == event_id and pos.status in (
                BetStatus.pending, BetStatus.matched
            ):
                if pos.runner_id == winner:
                    profit = pos.stake * (pos.price - 1)
                else:
                    profit = -pos.stake

                pos.profit = profit
                pos.status = BetStatus.settled
                pos.settled_at = datetime.now(timezone.utc)
                settled.append(pos)

                if self._session:
                    self._session.current_bankroll += profit

                    day_key = pos.settled_at.date().isoformat()
                    self._daily_pnl[day_key] = self._daily_pnl.get(day_key, 0.0) + profit

                if profit > 0:
                    self._consecutive_losses = 0
                else:
                    self._consecutive_losses += 1

        return settled

    def get_status(self) -> dict[str, Any]:
        if not self._session:
            return {"status": "no_session"}
        return {
            "session_id": self._session.id,
            "status": self._session.status.value,
            "starting_bankroll": self._session.starting_bankroll,
            "current_bankroll": self._session.current_bankroll,
            "open_positions": len(self.get_positions()),
            "settled_positions": len(self.get_history()),
            "pnl": self._session.current_bankroll - self._session.starting_bankroll,
            "consecutive_losses": self._consecutive_losses,
        }

    def get_positions(self) -> list[PaperPosition]:
        return [
            p for p in self._positions
            if p.status in (BetStatus.pending, BetStatus.matched, BetStatus.partially_matched)
        ]

    def get_history(self) -> list[PaperPosition]:
        return [p for p in self._positions if p.status == BetStatus.settled]

    # ── Internal ──────────────────────────────────────────────────────────

    def _evaluate_signal(self, event: dict[str, Any]) -> dict[str, Any] | None:
        for mkt in event.get("markets", []):
            for runner in mkt.get("runners", []):
                prices = runner.get("prices", [])
                if not prices:
                    continue
                last_price = prices[-1]
                back = float(last_price.get("back_price", 0))
                if back > 1.0:
                    return {
                        "market_id": mkt.get("id", ""),
                        "runner_id": runner.get("id", ""),
                        "price": back,
                    }
        return None
