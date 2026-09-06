from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from betlab.core.db.database import Database
from betlab.core.schemas.models import (
    BankrollSnapshot,
    BetSide,
    BetStatus,
    Dataset,
    Event,
    Experiment,
    ExperimentStatus,
    Market,
    MarketType,
    Participant,
    PaperPosition,
    PaperSession,
    PaperSessionStatus,
    PriceSnapshot,
    Runner,
    SimulatedBet,
    Sport,
    Strategy,
    StrategyVersion,
)


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


def _make_event(eid: str = "e1", sport: str = "tennis") -> Event:
    p = Participant(id=f"p_{eid}", name=f"Player {eid}", sport=sport)
    return Event(id=eid, sport=sport, competition="Test Cup", participants=[p])


def _make_dataset(did: str = "ds1") -> Dataset:
    return Dataset(id=did, name="test", version="1", sport=Sport.tennis)


def _make_strategy(sid: str = "s1") -> Strategy:
    return Strategy(id=sid, name="TestStrategy", sport=Sport.tennis)


def _make_strategy_version(vid: str = "sv1", sid: str = "s1") -> StrategyVersion:
    return StrategyVersion(id=vid, strategy_id=sid, version="1.0")


def _make_experiment(xid: str = "x1", sid: str = "s1", did: str = "ds1") -> Experiment:
    return Experiment(
        id=xid, strategy_id=sid, strategy_version="1.0",
        dataset_id=did, dataset_version="1",
    )


def _insert_full_chain(db: Database, event_id: str = "e1", market_id: str = "m1",
                        runner_id: str = "r1") -> None:
    db.insert_event(_make_event(event_id))
    m = Market(id=market_id, event_id=event_id, market_type=MarketType.match_odds)
    db.insert_market(m)
    p = Participant(id=f"p_{event_id}", name=f"Player {event_id}", sport=Sport.tennis)
    r = Runner(id=runner_id, market_id=market_id, participant_id=p.id, name="Runner1")
    db.insert_runner(r)


def _make_bet(bid: str = "b1", xid: str = "x1") -> SimulatedBet:
    return SimulatedBet(
        id=bid, experiment_id=xid, event_id="e1",
        market_id="m1", runner_id="r1", side=BetSide.back,
        requested_price=2.0, stake=10.0,
    )


def _make_bankroll(brid: str = "br1", xid: str = "x1") -> BankrollSnapshot:
    return BankrollSnapshot(
        id=brid, experiment_id=xid,
        timestamp=datetime(2025, 1, 1), balance=1000.0,
    )


def _make_paper_session(psid: str = "ps1", sid: str = "s1") -> PaperSession:
    return PaperSession(
        id=psid, strategy_id=sid, strategy_version="1.0",
        starting_bankroll=500.0, current_bankroll=500.0,
    )


def _make_paper_position(ppid: str = "pp1", psid: str = "ps1") -> PaperPosition:
    return PaperPosition(
        id=ppid, session_id=psid, event_id="e1",
        market_id="m1", runner_id="r1", side=BetSide.lay,
        stake=20.0, price=3.0,
    )


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestTableCreation:
    def test_creates_all_tables(self, db):
        cur = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cur.fetchall()}
        expected = {
            "events", "participants", "markets", "runners",
            "price_snapshots", "datasets", "strategies",
            "strategy_versions", "experiments", "simulated_bets",
            "bankroll_snapshots", "paper_sessions", "paper_positions",
        }
        assert expected.issubset(tables)

    def test_idempotent(self, db):
        db.create_tables()
        db.create_tables()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_insert_and_get(self, db):
        event = _make_event()
        db.insert_event(event)
        got = db.get_event("e1")
        assert got is not None
        assert got.id == "e1"
        assert got.sport == Sport.tennis
        assert got.competition == "Test Cup"

    def test_get_nonexistent(self, db):
        assert db.get_event("missing") is None

    def test_list_events(self, db):
        db.insert_event(_make_event("e1", "tennis"))
        db.insert_event(_make_event("e2", "tennis"))
        db.insert_event(_make_event("e3", "nba"))
        tennis = db.list_events(sport="tennis")
        assert len(tennis) == 2
        all_events = db.list_events()
        assert len(all_events) == 3

    def test_list_with_limit_offset(self, db):
        for i in range(5):
            db.insert_event(_make_event(f"e{i}"))
        page = db.list_events(limit=2, offset=1)
        assert len(page) == 2

    def test_insert_event_with_participants(self, db):
        event = _make_event()
        db.insert_event(event)
        assert len(event.participants) == 1


# ---------------------------------------------------------------------------
# Markets / Runners / Prices
# ---------------------------------------------------------------------------

class TestMarkets:
    def test_insert_market(self, db):
        db.insert_event(_make_event())
        m = Market(id="m1", event_id="e1", market_type=MarketType.match_odds)
        db.insert_market(m)
        cur = db._conn.execute("SELECT * FROM markets WHERE id = 'm1'")
        row = cur.fetchone()
        assert row is not None
        assert row["market_type"] == "match_odds"

    def test_insert_runner(self, db):
        _insert_full_chain(db)
        cur = db._conn.execute("SELECT * FROM runners WHERE id = 'r1'")
        assert cur.fetchone() is not None

    def test_insert_price_snapshot(self, db):
        _insert_full_chain(db)
        snap = PriceSnapshot(
            id="ps1", market_id="m1", runner_id="r1",
            timestamp=datetime(2025, 6, 1, 12, 0),
            back_price=2.1, lay_price=2.12, traded_volume=5000.0,
        )
        db.insert_price_snapshot(snap)
        prices = db.get_prices_for_runner("r1")
        assert len(prices) == 1
        assert prices[0].back_price == 2.1

    def test_get_prices_empty(self, db):
        assert db.get_prices_for_runner("nonexistent") == []


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_insert_and_get(self, db):
        ds = _make_dataset()
        db.insert_dataset(ds)
        got = db.get_dataset("ds1")
        assert got is not None
        assert got.name == "test"

    def test_get_nonexistent(self, db):
        assert db.get_dataset("missing") is None

    def test_list_datasets(self, db):
        db.insert_dataset(_make_dataset("ds1"))
        db.insert_dataset(_make_dataset("ds2"))
        datasets = db.list_datasets()
        assert len(datasets) == 2


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_insert_and_get(self, db):
        s = _make_strategy()
        db.insert_strategy(s)
        got = db.get_strategy("s1")
        assert got is not None
        assert got.name == "TestStrategy"
        assert got.versions == []

    def test_list_strategies(self, db):
        db.insert_strategy(_make_strategy("s1"))
        db.insert_strategy(_make_strategy("s2"))
        assert len(db.list_strategies()) == 2

    def test_insert_strategy_version(self, db):
        db.insert_strategy(_make_strategy())
        sv = _make_strategy_version()
        db.insert_strategy_version(sv)
        got = db.get_strategy_version("sv1")
        assert got is not None
        assert got.version == "1.0"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class TestExperiments:
    def test_insert_and_get(self, db):
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        exp = _make_experiment()
        db.insert_experiment(exp)
        got = db.get_experiment("x1")
        assert got is not None
        assert got.status == "pending"

    def test_update_status(self, db):
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        db.insert_experiment(_make_experiment())
        db.update_experiment_status("x1", ExperimentStatus.running)
        got = db.get_experiment("x1")
        assert got.status == ExperimentStatus.running

    def test_update_to_completed(self, db):
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        db.insert_experiment(_make_experiment())
        db.update_experiment_status("x1", ExperimentStatus.completed)
        got = db.get_experiment("x1")
        assert got.status == ExperimentStatus.completed


# ---------------------------------------------------------------------------
# Simulated bets
# ---------------------------------------------------------------------------

class TestSimulatedBets:
    def test_insert_and_get(self, db):
        _insert_full_chain(db)
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        db.insert_experiment(_make_experiment())
        bet = _make_bet()
        db.insert_simulated_bet(bet)
        bets = db.get_bets_for_experiment("x1")
        assert len(bets) == 1
        assert bets[0].side == BetSide.back

    def test_bulk_insert(self, db):
        _insert_full_chain(db)
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        db.insert_experiment(_make_experiment())
        bets = [_make_bet(f"b{i}") for i in range(10)]
        db.insert_simulated_bets_bulk(bets)
        assert len(db.get_bets_for_experiment("x1")) == 10

    def test_get_bets_empty(self, db):
        assert db.get_bets_for_experiment("nonexistent") == []


# ---------------------------------------------------------------------------
# Bankroll snapshots
# ---------------------------------------------------------------------------

class TestBankrollSnapshots:
    def test_insert_and_get(self, db):
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        db.insert_experiment(_make_experiment())
        snap = _make_bankroll()
        db.insert_bankroll_snapshot(snap)
        history = db.get_bankroll_history("x1")
        assert len(history) == 1
        assert history[0].balance == 1000.0

    def test_bulk_insert(self, db):
        db.insert_strategy(_make_strategy())
        db.insert_dataset(_make_dataset())
        db.insert_experiment(_make_experiment())
        snaps = [_make_bankroll(f"br{i}") for i in range(5)]
        db.insert_bankroll_snapshots_bulk(snaps)
        assert len(db.get_bankroll_history("x1")) == 5


# ---------------------------------------------------------------------------
# Paper sessions
# ---------------------------------------------------------------------------

class TestPaperSessions:
    def test_insert_and_get(self, db):
        db.insert_strategy(_make_strategy())
        session = _make_paper_session()
        db.insert_paper_session(session)
        got = db.get_paper_session("ps1")
        assert got is not None
        assert got.status == PaperSessionStatus.active
        assert got.starting_bankroll == 500.0

    def test_update_status(self, db):
        db.insert_strategy(_make_strategy())
        db.insert_paper_session(_make_paper_session())
        db.update_paper_session_status("ps1", PaperSessionStatus.paused)
        got = db.get_paper_session("ps1")
        assert got.status == PaperSessionStatus.paused

    def test_insert_position(self, db):
        _insert_full_chain(db)
        db.insert_strategy(_make_strategy())
        db.insert_paper_session(_make_paper_session())
        pos = _make_paper_position()
        db.insert_paper_position(pos)
        positions = db.get_positions_for_session("ps1")
        assert len(positions) == 1
        assert positions[0].side == BetSide.lay

    def test_get_positions_empty(self, db):
        assert db.get_positions_for_session("nonexistent") == []


# ---------------------------------------------------------------------------
# Bulk insert transactions
# ---------------------------------------------------------------------------

class TestBulkTransactions:
    def test_bulk_events(self, db):
        events = [_make_event(f"e{i}", "tennis") for i in range(20)]
        for e in events:
            db.insert_event(e)
        assert len(db.list_events(limit=100)) == 20

    def test_bulk_strategies(self, db):
        for i in range(10):
            db.insert_strategy(_make_strategy(f"s{i}"))
        assert len(db.list_strategies(limit=100)) == 10

    def test_bulk_datasets(self, db):
        for i in range(8):
            db.insert_dataset(_make_dataset(f"ds{i}"))
        assert len(db.list_datasets(limit=100)) == 8
