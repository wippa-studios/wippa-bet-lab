from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from betlab.core.schemas.models import (
    BetSide,
    CommissionConfig,
    ExecutionConfig,
    RiskConfig,
    SimulatedBet,
    SlippageConfig,
)


class MarketSimulator:
    """Simulates order execution against a price snapshot."""

    def __init__(
        self,
        commission_config: CommissionConfig | None = None,
        slippage_config: SlippageConfig | None = None,
        execution_config: ExecutionConfig | None = None,
    ):
        self.commission_config = commission_config or CommissionConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self.execution_config = execution_config or ExecutionConfig()

    # ── Public API ────────────────────────────────────────────────────────

    def simulate_fill(
        self,
        bet_candidate: dict[str, Any],
        price_snapshot: dict[str, Any],
        current_bankroll: float,
    ) -> SimulatedBet:
        side = BetSide(bet_candidate.get("side", "back"))
        requested_price = float(bet_candidate.get("requested_price", 0.0))
        stake = float(bet_candidate.get("stake", 0.0))
        runner_id = bet_candidate.get("runner_id", "")
        event_id = bet_candidate.get("event_id", "")
        market_id = bet_candidate.get("market_id", "")
        experiment_id = bet_candidate.get("experiment_id", "")

        is_void = price_snapshot.get("is_void", False)

        if is_void:
            return SimulatedBet(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                event_id=event_id,
                market_id=market_id,
                runner_id=runner_id,
                side=side,
                requested_price=requested_price,
                stake=stake,
                result="void",
                profit=0.0,
            )

        # Slippage
        slippage_applied = self.apply_slippage(requested_price, side, self.slippage_config)
        if side == BetSide.back:
            filled_price = requested_price - slippage_applied
        else:
            filled_price = requested_price + slippage_applied
        filled_price = max(filled_price, 1.01)

        # Liquidity check
        available, filled_stake = self.check_liquidity(
            price_snapshot, side, stake, self.execution_config
        )

        if filled_stake <= 0:
            return SimulatedBet(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                event_id=event_id,
                market_id=market_id,
                runner_id=runner_id,
                side=side,
                requested_price=requested_price,
                matched_price=filled_price,
                stake=0.0,
                result="unmatched",
                profit=0.0,
            )

        return SimulatedBet(
            id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            event_id=event_id,
            market_id=market_id,
            runner_id=runner_id,
            side=side,
            requested_price=requested_price,
            matched_price=filled_price,
            stake=filled_stake,
            result="matched",
            profit=0.0,
        )

    def compute_commission(self, profit: float, commission_config: CommissionConfig | None = None) -> float:
        cfg = commission_config or self.commission_config
        if cfg.type == "none":
            return 0.0
        if cfg.type == "percent_of_profit":
            return abs(profit) * cfg.rate
        if cfg.type == "net_market_profit":
            return abs(profit) * cfg.rate
        return 0.0

    def apply_slippage(self, price: float, side: BetSide, slippage_config: SlippageConfig | None = None) -> float:
        cfg = slippage_config or self.slippage_config
        if cfg.mode == "none" or cfg.mode == "ticks":
            ticks = cfg.back if side == BetSide.back else cfg.lay
            return ticks
        if cfg.mode == "percent":
            pct = cfg.back if side == BetSide.back else cfg.lay
            return price * pct
        if cfg.mode == "fixed":
            fixed = cfg.back if side == BetSide.back else cfg.lay
            return fixed
        return 0.0

    def check_liquidity(
        self,
        price_snapshot: dict[str, Any],
        side: BetSide,
        stake: float,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[float, float]:
        cfg = execution_config or self.execution_config
        if side == BetSide.back:
            available = float(price_snapshot.get("back_available", price_snapshot.get("back_size", 0.0)))
        else:
            available = float(price_snapshot.get("lay_available", price_snapshot.get("lay_size", 0.0)))

        if cfg.partial_fills:
            filled = min(stake, available)
        else:
            filled = stake if stake <= available else 0.0

        return available, filled
