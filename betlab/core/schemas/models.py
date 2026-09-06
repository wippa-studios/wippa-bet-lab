from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Sport(str, Enum):
    tennis = "tennis"
    greyhounds = "greyhounds"
    nba = "nba"


class MarketType(str, Enum):
    match_odds = "match_odds"
    spread = "spread"
    total = "total"
    moneyline = "moneyline"


class BetSide(str, Enum):
    back = "back"
    lay = "lay"


class BetStatus(str, Enum):
    candidate = "candidate"
    pending = "pending"
    matched = "matched"
    partially_matched = "partially_matched"
    unmatched = "unmatched"
    settled = "settled"
    void = "void"
    cancelled = "cancelled"
    expired = "expired"


class ExperimentStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class PaperSessionStatus(str, Enum):
    active = "active"
    paused = "paused"
    stopped = "stopped"


class StakingMethod(str, Enum):
    fixed_stake = "fixed_stake"
    fixed_percent = "fixed_percent"
    kelly = "kelly"
    fractional_kelly = "fractional_kelly"
    odds_band = "odds_band"
    volatility_adjusted = "volatility_adjusted"
    confidence_weighted = "confidence_weighted"
    drawdown_adjusted = "drawdown_adjusted"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Participant(BaseModel):
    id: str = Field(..., description="Unique participant identifier")
    name: str
    sport: Sport
    ranking: int | None = None
    elo_rating: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    id: str
    sport: Sport
    competition: str = ""
    start_time: datetime | None = None
    venue: str = ""
    status: str = "upcoming"
    result_status: str = ""
    home_away: bool = False
    participants: list[Participant] = Field(default_factory=list)


class Runner(BaseModel):
    id: str
    market_id: str
    participant_id: str
    name: str
    result: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Market(BaseModel):
    id: str
    event_id: str
    market_type: MarketType
    market_name: str = ""
    commission_model: str = ""
    status: str = "open"


class PriceSnapshot(BaseModel):
    id: str
    market_id: str
    runner_id: str
    timestamp: datetime
    back_price: float | None = None
    back_available: float | None = None
    lay_price: float | None = None
    lay_available: float | None = None
    traded_volume: float | None = None


class Dataset(BaseModel):
    id: str
    name: str
    version: str
    sport: Sport
    source: str = ""
    event_count: int = 0
    market_count: int = 0
    time_range: str = ""
    schema_hash: str = ""
    data_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def compute_data_hash(self) -> str:
        payload = f"{self.name}:{self.version}:{self.event_count}:{self.market_count}"
        self.data_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return self.data_hash


class Strategy(BaseModel):
    id: str
    name: str
    sport: Sport
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    versions: list[str] = Field(default_factory=list)


class StrategyVersion(BaseModel):
    id: str
    strategy_id: str
    version: str
    definition: dict[str, Any] = Field(default_factory=dict)
    hash: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Experiment(BaseModel):
    id: str
    strategy_id: str
    strategy_version: str
    dataset_id: str
    dataset_version: str
    engine_version: str = "0.1.0"
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ExperimentStatus = ExperimentStatus.pending


class SimulatedBet(BaseModel):
    id: str
    experiment_id: str
    event_id: str
    market_id: str
    runner_id: str
    side: BetSide
    requested_price: float
    matched_price: float | None = None
    stake: float
    liability: float = 0.0
    commission: float = 0.0
    result: BetStatus = BetStatus.candidate
    profit: float = 0.0
    clv_odds: float | None = None
    clv_implied_probability: float | None = None
    clv_percent: float | None = None
    placed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BankrollSnapshot(BaseModel):
    id: str
    experiment_id: str
    timestamp: datetime
    balance: float
    drawdown: float = 0.0
    open_exposure: float = 0.0
    bets_count: int = 0
    cumulative_pnl: float = 0.0


class PaperSession(BaseModel):
    id: str
    strategy_id: str
    strategy_version: str
    starting_bankroll: float
    current_bankroll: float
    status: PaperSessionStatus = PaperSessionStatus.active
    started_at: datetime = Field(default_factory=datetime.utcnow)
    risk_limits: dict[str, Any] = Field(default_factory=dict)


class PaperPosition(BaseModel):
    id: str
    session_id: str
    event_id: str
    market_id: str
    runner_id: str
    side: BetSide
    stake: float
    price: float
    status: BetStatus = BetStatus.pending
    profit: float = 0.0
    placed_at: datetime = Field(default_factory=datetime.utcnow)
    settled_at: datetime | None = None


# ---------------------------------------------------------------------------
# Strategy config sub-models
# ---------------------------------------------------------------------------

class CommissionConfig(BaseModel):
    type: str = "none"
    rate: float = 0.0

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"net_market_profit", "percent_of_profit", "none"}
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}, got {v!r}")
        return v


class SlippageConfig(BaseModel):
    mode: str = "ticks"
    back: float = 0.0
    lay: float = 0.0

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"ticks", "percent", "fixed"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v!r}")
        return v


class ExecutionConfig(BaseModel):
    delay_ms: int = 0
    fill_model: str = "available_liquidity"
    partial_fills: bool = True

    @field_validator("fill_model")
    @classmethod
    def validate_fill_model(cls, v: str) -> str:
        allowed = {"available_liquidity", "guaranteed_fill", "market_odds"}
        if v not in allowed:
            raise ValueError(f"fill_model must be one of {allowed}, got {v!r}")
        return v


class StakingConfig(BaseModel):
    method: StakingMethod = StakingMethod.fixed_stake
    fraction: float = 1.0
    maximum_stake_percent: float = 100.0
    maximum_stake_absolute: float = 0.0
    odds_band_min: float = 1.0
    odds_band_max: float = 100.0


class RiskConfig(BaseModel):
    maximum_open_exposure: float = 0.0
    maximum_daily_loss: float = 0.0
    maximum_bankroll_percent: float = 100.0


class ValidationConfig(BaseModel):
    method: str = "walk_forward"
    training_window: int = 0
    validation_window: int = 0
    step: int = 1
    purge_gap: int = 0
    minimum_training_events: int = 0

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"walk_forward", "train_test", "k_fold"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}, got {v!r}")
        return v


class FilterCondition(BaseModel):
    field: str
    operator: str
    value: Any


class SignalCondition(BaseModel):
    field: str
    operator: str
    value: Any


class SelectionConfig(BaseModel):
    side: BetSide = BetSide.back
    runner: str = "market_favourite"

    @field_validator("runner")
    @classmethod
    def validate_runner(cls, v: str) -> str:
        allowed = {"market_favourite", "market_underdog", "specific_runner"}
        if v not in allowed:
            raise ValueError(f"runner must be one of {allowed}, got {v!r}")
        return v


class PriceConfig(BaseModel):
    source: str = "market"
    minimum: float = 1.01
    maximum: float = 1000.0


class StrategyDefinition(BaseModel):
    id: str
    version: str
    name: str
    sport: Sport
    market: MarketType
    filters: list[FilterCondition] = Field(default_factory=list)
    signal: SignalCondition = Field(default_factory=lambda: SignalCondition(field="", operator="", value=0))
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    price: PriceConfig = Field(default_factory=PriceConfig)
    staking: StakingConfig = Field(default_factory=StakingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    def compute_hash(self) -> str:
        import json
        payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
