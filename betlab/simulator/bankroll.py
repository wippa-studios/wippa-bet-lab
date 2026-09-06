from __future__ import annotations

import uuid
from datetime import datetime, timezone

from betlab.core.schemas.models import (
    BankrollSnapshot,
    RiskConfig,
    StakingConfig,
    StakingMethod,
)


class BankrollManager:
    """Manages bankroll state and stake sizing."""

    def __init__(
        self,
        starting_balance: float = 1000.0,
        staking_config: StakingConfig | None = None,
        risk_config: RiskConfig | None = None,
    ):
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.staking_config = staking_config or StakingConfig()
        self.risk_config = risk_config or RiskConfig()
        self.bets_count = 0
        self.daily_pnl = 0.0
        self.open_exposure = 0.0
        self._snapshots: list[BankrollSnapshot] = []

    # ── Staking ───────────────────────────────────────────────────────────

    def compute_stake(self, probability: float, odds: float, current_balance: float | None = None) -> float:
        bal = current_balance if current_balance is not None else self.balance
        cfg = self.staking_config

        if cfg.method == StakingMethod.fixed_stake:
            max_stake = cfg.maximum_stake_absolute if cfg.maximum_stake_absolute > 0 else bal
            return min(bal, max_stake)

        if cfg.method == StakingMethod.fixed_percent:
            stake = bal * cfg.fraction
            max_pct = bal * cfg.maximum_stake_percent / 100.0 if cfg.maximum_stake_percent > 1 else bal * cfg.maximum_stake_percent
            return min(stake, max_pct) if max_pct > 0 else stake

        if cfg.method == StakingMethod.kelly:
            return self.kelly_stake(probability, odds, fraction=1.0)

        if cfg.method == StakingMethod.fractional_kelly:
            return self.kelly_stake(probability, odds, fraction=cfg.fraction)

        if cfg.method == StakingMethod.odds_band:
            if cfg.odds_band_min <= odds <= cfg.odds_band_max:
                return bal * cfg.fraction
            return 0.0

        return 1.0

    def kelly_stake(self, probability: float, odds: float, fraction: float = 1.0) -> float:
        if odds <= 1.0 or probability <= 0.0 or probability >= 1.0:
            return 0.0
        q = 1.0 - probability
        b = odds - 1.0
        kelly = (probability * b - q) / b
        if kelly <= 0:
            return 0.0
        stake = self.balance * kelly * fraction
        max_stake = self.balance * self.staking_config.maximum_stake_percent / 100.0 if self.staking_config.maximum_stake_percent > 1 else self.balance * self.staking_config.maximum_stake_percent
        if max_stake > 0:
            return min(max(stake, 0.0), max_stake)
        return max(stake, 0.0)

    # ── Risk ──────────────────────────────────────────────────────────────

    def check_risk_limits(
        self,
        current_balance: float | None = None,
        open_exposure: float | None = None,
        daily_loss: float | None = None,
    ) -> bool:
        bal = current_balance if current_balance is not None else self.balance
        exposure = open_exposure if open_exposure is not None else self.open_exposure
        d_loss = daily_loss if daily_loss is not None else self.daily_pnl

        if self.risk_config.maximum_bankroll_percent > 0 and self.risk_config.maximum_bankroll_percent < 100:
            if bal < self.starting_balance * (1 - self.risk_config.maximum_bankroll_percent / 100.0):
                return False

        if self.risk_config.maximum_open_exposure > 0:
            if bal > 0 and exposure / bal > self.risk_config.maximum_open_exposure:
                return False

        if self.risk_config.maximum_daily_loss > 0 and d_loss < 0:
            if bal > 0 and abs(d_loss) / bal > self.risk_config.maximum_daily_loss:
                return False

        return True

    # ── Balance updates ───────────────────────────────────────────────────

    def update_balance(self, pnl: float) -> float:
        self.balance += pnl
        self.daily_pnl += pnl
        self.bets_count += 1
        return self.balance

    # ── Snapshots ─────────────────────────────────────────────────────────

    def record_snapshot(self, timestamp: datetime | None = None, bets_count: int | None = None) -> BankrollSnapshot:
        ts = timestamp or datetime.now(timezone.utc)
        bc = bets_count if bets_count is not None else self.bets_count
        snap = BankrollSnapshot(
            id=str(uuid.uuid4()),
            experiment_id="",
            timestamp=ts,
            balance=self.balance,
            bets_count=bc,
            open_exposure=self.open_exposure,
            cumulative_pnl=self.balance - self.starting_balance,
        )
        self._snapshots.append(snap)
        return snap

    @property
    def snapshots(self) -> list[BankrollSnapshot]:
        return list(self._snapshots)
