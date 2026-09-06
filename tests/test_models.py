from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from betlab.core.schemas.models import (
    BankrollSnapshot,
    BetSide,
    BetStatus,
    CommissionConfig,
    Dataset,
    Event,
    Experiment,
    ExperimentStatus,
    ExecutionConfig,
    FilterCondition,
    Market,
    MarketType,
    Participant,
    PaperPosition,
    PaperSession,
    PaperSessionStatus,
    PriceConfig,
    PriceSnapshot,
    RiskConfig,
    Runner,
    SelectionConfig,
    SignalCondition,
    SimulatedBet,
    SlippageConfig,
    Sport,
    StakingConfig,
    StakingMethod,
    Strategy,
    StrategyDefinition,
    StrategyVersion,
    ValidationConfig,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestSportEnum:
    def test_valid_values(self):
        assert Sport.tennis == "tennis"
        assert Sport.greyhounds == "greyhounds"
        assert Sport.nba == "nba"

    def test_invalid_value(self):
        with pytest.raises((ValueError, ValidationError)):
            Sport("football")


class TestMarketTypeEnum:
    def test_valid_values(self):
        assert MarketType.match_odds == "match_odds"
        assert MarketType.spread == "spread"
        assert MarketType.total == "total"
        assert MarketType.moneyline == "moneyline"

    def test_invalid_value(self):
        with pytest.raises((ValueError, ValidationError)):
            MarketType("handicap")


class TestBetSideEnum:
    def test_valid(self):
        assert BetSide.back == "back"
        assert BetSide.lay == "lay"

    def test_invalid(self):
        with pytest.raises((ValueError, ValidationError)):
            BetSide("middle")


class TestBetStatusEnum:
    def test_valid(self):
        for status in ("candidate", "pending", "matched", "partially_matched",
                       "unmatched", "settled", "void", "cancelled", "expired"):
            assert BetStatus(status) == status

    def test_invalid(self):
        with pytest.raises((ValueError, ValidationError)):
            BetStatus("unknown_status")


class TestExperimentStatusEnum:
    def test_valid(self):
        for status in ("pending", "running", "completed", "failed", "cancelled"):
            assert ExperimentStatus(status) == status

    def test_invalid(self):
        with pytest.raises((ValueError, ValidationError)):
            ExperimentStatus("done")


class TestPaperSessionStatusEnum:
    def test_valid(self):
        assert PaperSessionStatus.active == "active"
        assert PaperSessionStatus.paused == "paused"
        assert PaperSessionStatus.stopped == "stopped"

    def test_invalid(self):
        with pytest.raises((ValueError, ValidationError)):
            PaperSessionStatus("closed")


class TestStakingMethodEnum:
    def test_valid(self):
        for m in ("fixed_stake", "fixed_percent", "kelly", "fractional_kelly",
                  "odds_band", "volatility_adjusted", "confidence_weighted", "drawdown_adjusted"):
            assert StakingMethod(m) == m

    def test_invalid(self):
        with pytest.raises((ValueError, ValidationError)):
            StakingMethod("martingale")


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class TestParticipant:
    def test_create_minimal(self):
        p = Participant(id="p1", name="Player A", sport=Sport.tennis)
        assert p.id == "p1"
        assert p.name == "Player A"
        assert p.ranking is None
        assert p.elo_rating is None
        assert p.metadata == {}

    def test_create_full(self):
        p = Participant(
            id="p2", name="Player B", sport=Sport.nba,
            ranking=5, elo_rating=1800.0, metadata={"team": "Lakers"}
        )
        assert p.ranking == 5
        assert p.elo_rating == 1800.0
        assert p.metadata["team"] == "Lakers"

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            Participant(name="X", sport=Sport.tennis)


class TestEvent:
    def test_create_minimal(self):
        e = Event(id="e1", sport=Sport.tennis)
        assert e.id == "e1"
        assert e.competition == ""
        assert e.participants == []

    def test_create_full(self):
        p = Participant(id="p1", name="A", sport=Sport.tennis)
        e = Event(
            id="e2", sport=Sport.nba, competition="NBA Finals",
            start_time=datetime(2025, 6, 1, 20, 0),
            venue="Staples Center", status="live",
            home_away=True, participants=[p],
        )
        assert e.competition == "NBA Finals"
        assert len(e.participants) == 1

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            Event(sport=Sport.tennis)


class TestMarket:
    def test_create(self):
        m = Market(id="m1", event_id="e1", market_type=MarketType.match_odds)
        assert m.market_name == ""
        assert m.status == "open"


class TestRunner:
    def test_create(self):
        r = Runner(id="r1", market_id="m1", participant_id="p1", name="Runner 1")
        assert r.result == ""
        assert r.metadata == {}


class TestPriceSnapshot:
    def test_create(self):
        snap = PriceSnapshot(
            id="ps1", market_id="m1", runner_id="r1",
            timestamp=datetime(2025, 1, 1, 12, 0),
            back_price=2.1, back_available=100.0,
            lay_price=2.12, lay_available=50.0,
            traded_volume=5000.0,
        )
        assert snap.back_price == 2.1
        assert snap.traded_volume == 5000.0


class TestDataset:
    def test_create(self):
        ds = Dataset(id="ds1", name="test", version="1", sport=Sport.tennis)
        assert ds.event_count == 0
        assert ds.created_at is not None

    def test_compute_hash(self):
        ds = Dataset(id="ds1", name="test", version="1", sport=Sport.tennis,
                     event_count=100, market_count=200)
        h = ds.compute_data_hash()
        assert len(h) == 16
        assert h == ds.data_hash


class TestStrategy:
    def test_create(self):
        s = Strategy(id="s1", name="My Strat", sport=Sport.greyhounds)
        assert s.versions == []
        assert s.created_at is not None


class TestStrategyVersion:
    def test_create(self):
        sv = StrategyVersion(id="sv1", strategy_id="s1", version="1.0")
        assert sv.definition == {}


class TestExperiment:
    def test_create(self):
        exp = Experiment(
            id="x1", strategy_id="s1", strategy_version="1.0",
            dataset_id="ds1", dataset_version="1",
        )
        assert exp.status == ExperimentStatus.pending
        assert exp.seed == 0


class TestSimulatedBet:
    def test_create(self):
        bet = SimulatedBet(
            id="b1", experiment_id="x1", event_id="e1",
            market_id="m1", runner_id="r1", side=BetSide.back,
            requested_price=2.0, stake=10.0,
        )
        assert bet.result == BetStatus.candidate
        assert bet.profit == 0.0


class TestBankrollSnapshot:
    def test_create(self):
        snap = BankrollSnapshot(
            id="br1", experiment_id="x1",
            timestamp=datetime(2025, 1, 1),
            balance=1000.0,
        )
        assert snap.drawdown == 0.0
        assert snap.cumulative_pnl == 0.0


class TestPaperSession:
    def test_create(self):
        ps = PaperSession(
            id="ps1", strategy_id="s1", strategy_version="1.0",
            starting_bankroll=500.0, current_bankroll=500.0,
        )
        assert ps.status == PaperSessionStatus.active


class TestPaperPosition:
    def test_create(self):
        pp = PaperPosition(
            id="pp1", session_id="ps1", event_id="e1",
            market_id="m1", runner_id="r1", side=BetSide.lay,
            stake=20.0, price=3.0,
        )
        assert pp.status == BetStatus.pending
        assert pp.settled_at is None


# ---------------------------------------------------------------------------
# Config sub-models
# ---------------------------------------------------------------------------

class TestCommissionConfig:
    def test_default(self):
        c = CommissionConfig()
        assert c.type == "none"
        assert c.rate == 0.0

    def test_valid_type(self):
        c = CommissionConfig(type="net_market_profit", rate=0.05)
        assert c.type == "net_market_profit"

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            CommissionConfig(type="invalid")


class TestSlippageConfig:
    def test_default(self):
        s = SlippageConfig()
        assert s.mode == "ticks"
        assert s.back == 0.0

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            SlippageConfig(mode="random")


class TestExecutionConfig:
    def test_default(self):
        e = ExecutionConfig()
        assert e.delay_ms == 0
        assert e.partial_fills is True

    def test_invalid_fill_model(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(fill_model="instant")


class TestStakingConfig:
    def test_default(self):
        s = StakingConfig()
        assert s.method == StakingMethod.fixed_stake
        assert s.fraction == 1.0

    def test_kelly(self):
        s = StakingConfig(method=StakingMethod.kelly, fraction=0.25)
        assert s.method == StakingMethod.kelly


class TestRiskConfig:
    def test_default(self):
        r = RiskConfig()
        assert r.maximum_open_exposure == 0.0
        assert r.maximum_bankroll_percent == 100.0


class TestValidationConfig:
    def test_default(self):
        v = ValidationConfig()
        assert v.method == "walk_forward"

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            ValidationConfig(method="random_split")


class TestFilterCondition:
    def test_create(self):
        f = FilterCondition(field="odds", operator="lt", value=3.0)
        assert f.field == "odds"


class TestSignalCondition:
    def test_create(self):
        s = SignalCondition(field="ema_cross", operator="gt", value=0)
        assert s.operator == "gt"


class TestSelectionConfig:
    def test_default(self):
        s = SelectionConfig()
        assert s.side == BetSide.back
        assert s.runner == "market_favourite"

    def test_invalid_runner(self):
        with pytest.raises(ValidationError):
            SelectionConfig(runner="random")


class TestPriceConfig:
    def test_default(self):
        p = PriceConfig()
        assert p.source == "market"
        assert p.minimum == 1.01


class TestStrategyDefinition:
    def test_create_minimal(self):
        sd = StrategyDefinition(
            id="sd1", version="1", name="Test",
            sport=Sport.tennis, market=MarketType.match_odds,
        )
        assert len(sd.filters) == 0

    def test_compute_hash(self):
        sd = StrategyDefinition(
            id="sd1", version="1", name="Test",
            sport=Sport.tennis, market=MarketType.match_odds,
        )
        h = sd.compute_hash()
        assert len(h) == 16

    def test_with_filters(self):
        sd = StrategyDefinition(
            id="sd1", version="1", name="Test",
            sport=Sport.tennis, market=MarketType.match_odds,
            filters=[FilterCondition(field="odds", operator="lt", value=3.0)],
        )
        assert len(sd.filters) == 1


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    @pytest.mark.parametrize("model_cls,kwargs", [
        (Participant, {"id": "p1", "name": "A", "sport": "tennis"}),
        (Event, {"id": "e1", "sport": "tennis"}),
        (Market, {"id": "m1", "event_id": "e1", "market_type": "match_odds"}),
        (Runner, {"id": "r1", "market_id": "m1", "participant_id": "p1", "name": "R1"}),
        (Dataset, {"id": "ds1", "name": "D", "version": "1", "sport": "tennis"}),
        (Strategy, {"id": "s1", "name": "S", "sport": "tennis"}),
        (StrategyVersion, {"id": "sv1", "strategy_id": "s1", "version": "1"}),
        (Experiment, {"id": "x1", "strategy_id": "s1", "strategy_version": "1", "dataset_id": "d1", "dataset_version": "1"}),
        (SimulatedBet, {"id": "b1", "experiment_id": "x1", "event_id": "e1", "market_id": "m1", "runner_id": "r1", "side": "back", "requested_price": 2.0, "stake": 10.0}),
        (BankrollSnapshot, {"id": "br1", "experiment_id": "x1", "timestamp": datetime(2025, 1, 1), "balance": 1000.0}),
        (PaperSession, {"id": "ps1", "strategy_id": "s1", "strategy_version": "1", "starting_bankroll": 500.0, "current_bankroll": 500.0}),
        (PaperPosition, {"id": "pp1", "session_id": "ps1", "event_id": "e1", "market_id": "m1", "runner_id": "r1", "side": "back", "stake": 10.0, "price": 2.0}),
        (CommissionConfig, {"type": "none", "rate": 0.0}),
        (SlippageConfig, {"mode": "ticks", "back": 0.0, "lay": 0.0}),
        (ExecutionConfig, {"delay_ms": 0, "fill_model": "available_liquidity"}),
        (StakingConfig, {"method": "fixed_stake", "fraction": 1.0}),
        (RiskConfig, {"maximum_open_exposure": 0.0}),
        (ValidationConfig, {"method": "walk_forward"}),
        (FilterCondition, {"field": "odds", "operator": "lt", "value": 3.0}),
        (SignalCondition, {"field": "ema", "operator": "gt", "value": 0}),
        (SelectionConfig, {"side": "back", "runner": "market_favourite"}),
        (PriceConfig, {"source": "market", "minimum": 1.01}),
        (StrategyDefinition, {"id": "sd1", "version": "1", "name": "T", "sport": "tennis", "market": "match_odds"}),
    ])
    def test_round_trip(self, model_cls, kwargs):
        obj = model_cls(**kwargs)
        d = obj.model_dump()
        restored = model_cls.model_validate(d)
        assert obj == restored

    def test_json_round_trip(self):
        exp = Experiment(
            id="x1", strategy_id="s1", strategy_version="1",
            dataset_id="d1", dataset_version="1",
            config={"key": "value"},
        )
        json_str = exp.model_dump_json()
        restored = Experiment.model_validate_json(json_str)
        assert exp == restored
