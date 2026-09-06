from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any

from betlab.core.schemas.models import PaperPosition, RiskConfig, BetStatus


@dataclass
class RiskCheck:
    allowed: bool = True
    reason: str = ""


class RiskManager:
    """Enforces risk limits for paper trading sessions."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def checkLimits(
        self,
        positions: list[PaperPosition],
        new_bet: dict[str, Any],
        current_bankroll: float,
        starting_bankroll: float,
        consecutive_losses: int = 0,
    ) -> RiskCheck:
        # Max open exposure
        if self.config.maximum_open_exposure > 0:
            exposure = self.compute_open_exposure(positions)
            new_stake = float(new_bet.get("stake", 0))
            total = exposure + new_stake
            if current_bankroll > 0 and total / current_bankroll > self.config.maximum_open_exposure:
                return RiskCheck(
                    allowed=False,
                    reason=f"Open exposure {total:.2f} would exceed limit "
                           f"{self.config.maximum_open_exposure * 100:.1f}% of bankroll",
                )

        # Max daily loss
        if self.config.maximum_daily_loss > 0:
            daily_loss = self.compute_daily_loss(positions, date.today())
            if current_bankroll > 0 and abs(daily_loss) / current_bankroll > self.config.maximum_daily_loss:
                return RiskCheck(
                    allowed=False,
                    reason=f"Daily loss {abs(daily_loss):.2f} exceeds limit "
                           f"{self.config.maximum_daily_loss * 100:.1f}% of bankroll",
                )

        # Max bankroll percent
        if self.config.maximum_bankroll_percent > 0 and self.config.maximum_bankroll_percent < 100:
            loss_pct = (starting_bankroll - current_bankroll) / starting_bankroll * 100 if starting_bankroll > 0 else 0
            if loss_pct > self.config.maximum_bankroll_percent:
                return RiskCheck(
                    allowed=False,
                    reason=f"Total loss {loss_pct:.1f}% exceeds limit "
                           f"{self.config.maximum_bankroll_percent:.1f}%",
                )

        # Max consecutive losses
        if consecutive_losses >= 3:
            return RiskCheck(
                allowed=False,
                reason=f"Consecutive losses ({consecutive_losses}) exceeded threshold",
            )

        return RiskCheck(allowed=True)

    def compute_daily_loss(self, positions: list[PaperPosition], target_date: date) -> float:
        daily = [
            p for p in positions
            if p.settled_at and p.settled_at.date() == target_date
        ]
        return sum(p.profit for p in daily)

    def compute_open_exposure(self, positions: list[PaperPosition]) -> float:
        open_pos = [
            p for p in positions
            if p.status in (BetStatus.pending, BetStatus.matched, BetStatus.partially_matched)
        ]
        return sum(p.stake for p in open_pos)
