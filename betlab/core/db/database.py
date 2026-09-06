from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from betlab.core.schemas.models import (
    BankrollSnapshot,
    BetStatus,
    Dataset,
    Event,
    Experiment,
    ExperimentStatus,
    Market,
    Participant,
    PaperPosition,
    PaperSession,
    PaperSessionStatus,
    PriceSnapshot,
    Runner,
    SimulatedBet,
    Strategy,
    StrategyVersion,
)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    competition TEXT DEFAULT '',
    start_time TEXT,
    venue TEXT DEFAULT '',
    status TEXT DEFAULT 'upcoming',
    result_status TEXT DEFAULT '',
    home_away INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS participants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sport TEXT NOT NULL,
    ranking INTEGER,
    elo_rating REAL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    market_name TEXT DEFAULT '',
    commission_model TEXT DEFAULT '',
    status TEXT DEFAULT 'open',
    FOREIGN KEY (event_id) REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS runners (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    result TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (market_id) REFERENCES markets(id),
    FOREIGN KEY (participant_id) REFERENCES participants(id)
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    runner_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    back_price REAL,
    back_available REAL,
    lay_price REAL,
    lay_available REAL,
    traded_volume REAL,
    FOREIGN KEY (market_id) REFERENCES markets(id),
    FOREIGN KEY (runner_id) REFERENCES runners(id)
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    sport TEXT NOT NULL,
    source TEXT DEFAULT '',
    event_count INTEGER DEFAULT 0,
    market_count INTEGER DEFAULT 0,
    time_range TEXT DEFAULT '',
    schema_hash TEXT DEFAULT '',
    data_hash TEXT DEFAULT '',
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sport TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT,
    versions TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    version TEXT NOT NULL,
    definition TEXT DEFAULT '{}',
    hash TEXT DEFAULT '',
    created_at TEXT,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    engine_version TEXT DEFAULT '0.1.0',
    config TEXT DEFAULT '{}',
    seed INTEGER DEFAULT 0,
    created_at TEXT,
    status TEXT DEFAULT 'pending',
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS simulated_bets (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    runner_id TEXT NOT NULL,
    side TEXT NOT NULL,
    requested_price REAL NOT NULL,
    matched_price REAL,
    stake REAL NOT NULL,
    liability REAL DEFAULT 0.0,
    commission REAL DEFAULT 0.0,
    result TEXT DEFAULT 'candidate',
    profit REAL DEFAULT 0.0,
    clv_odds REAL,
    clv_implied_probability REAL,
    clv_percent REAL,
    placed_at TEXT,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (event_id) REFERENCES events(id),
    FOREIGN KEY (market_id) REFERENCES markets(id),
    FOREIGN KEY (runner_id) REFERENCES runners(id)
);

CREATE TABLE IF NOT EXISTS bankroll_snapshots (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    balance REAL NOT NULL,
    drawdown REAL DEFAULT 0.0,
    open_exposure REAL DEFAULT 0.0,
    bets_count INTEGER DEFAULT 0,
    cumulative_pnl REAL DEFAULT 0.0,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS paper_sessions (
    id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    starting_bankroll REAL NOT NULL,
    current_bankroll REAL NOT NULL,
    status TEXT DEFAULT 'active',
    started_at TEXT,
    risk_limits TEXT DEFAULT '{}',
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    runner_id TEXT NOT NULL,
    side TEXT NOT NULL,
    stake REAL NOT NULL,
    price REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    profit REAL DEFAULT 0.0,
    placed_at TEXT,
    settled_at TEXT,
    FOREIGN KEY (session_id) REFERENCES paper_sessions(id),
    FOREIGN KEY (event_id) REFERENCES events(id),
    FOREIGN KEY (market_id) REFERENCES markets(id),
    FOREIGN KEY (runner_id) REFERENCES runners(id)
);
"""


class Database:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def insert_event(self, event: Event) -> None:
        self._conn.execute(
            "INSERT INTO events (id, sport, competition, start_time, venue, status, result_status, home_away) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.sport.value,
                event.competition,
                event.start_time.isoformat() if event.start_time else None,
                event.venue,
                event.status,
                event.result_status,
                int(event.home_away),
            ),
        )
        for p in event.participants:
            self.insert_participant(p)
        self._conn.commit()

    def get_event(self, event_id: str) -> Event | None:
        cur = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        if row is None:
            return None
        participants = self._get_participants_for_event(event_id)
        return self._row_to_event(row, participants)

    def list_events(
        self, sport: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Event]:
        if sport:
            cur = self._conn.execute(
                "SELECT * FROM events WHERE sport = ? ORDER BY start_time LIMIT ? OFFSET ?",
                (sport, limit, offset),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM events ORDER BY start_time LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = cur.fetchall()
        return [
            self._row_to_event(r, self._get_participants_for_event(r["id"])) for r in rows
        ]

    def _get_participants_for_event(self, event_id: str) -> list[Participant]:
        cur = self._conn.execute("SELECT * FROM participants WHERE sport IN (SELECT sport FROM events WHERE id = ?)", (event_id,))
        rows = cur.fetchall()
        return [self._row_to_participant(r) for r in rows]

    def insert_participant(self, p: Participant) -> None:
        import json
        self._conn.execute(
            "INSERT OR IGNORE INTO participants (id, name, sport, ranking, elo_rating, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (p.id, p.name, p.sport.value, p.ranking, p.elo_rating, json.dumps(p.metadata)),
        )

    # ------------------------------------------------------------------
    # Markets / Runners / Prices
    # ------------------------------------------------------------------

    def insert_market(self, market: Market) -> None:
        self._conn.execute(
            "INSERT INTO markets (id, event_id, market_type, market_name, commission_model, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (market.id, market.event_id, market.market_type.value, market.market_name, market.commission_model, market.status),
        )
        self._conn.commit()

    def insert_runner(self, runner: Runner) -> None:
        import json
        self._conn.execute(
            "INSERT INTO runners (id, market_id, participant_id, name, result, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (runner.id, runner.market_id, runner.participant_id, runner.name, runner.result, json.dumps(runner.metadata)),
        )
        self._conn.commit()

    def insert_price_snapshot(self, snap: PriceSnapshot) -> None:
        self._conn.execute(
            "INSERT INTO price_snapshots (id, market_id, runner_id, timestamp, back_price, back_available, lay_price, lay_available, traded_volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (snap.id, snap.market_id, snap.runner_id, snap.timestamp.isoformat(), snap.back_price, snap.back_available, snap.lay_price, snap.lay_available, snap.traded_volume),
        )
        self._conn.commit()

    def get_prices_for_runner(self, runner_id: str) -> list[PriceSnapshot]:
        cur = self._conn.execute("SELECT * FROM price_snapshots WHERE runner_id = ? ORDER BY timestamp", (runner_id,))
        return [self._row_to_price_snapshot(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def insert_dataset(self, dataset: Dataset) -> None:
        self._conn.execute(
            "INSERT INTO datasets (id, name, version, sport, source, event_count, market_count, time_range, schema_hash, data_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset.id, dataset.name, dataset.version, dataset.sport.value,
                dataset.source, dataset.event_count, dataset.market_count,
                dataset.time_range, dataset.schema_hash, dataset.data_hash,
                dataset.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        cur = self._conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cur.fetchone()
        return self._row_to_dataset(row) if row else None

    def list_datasets(self, limit: int = 50, offset: int = 0) -> list[Dataset]:
        cur = self._conn.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        )
        return [self._row_to_dataset(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def insert_strategy(self, strategy: Strategy) -> None:
        import json
        self._conn.execute(
            "INSERT INTO strategies (id, name, sport, description, created_at, versions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                strategy.id, strategy.name, strategy.sport.value,
                strategy.description, strategy.created_at.isoformat(),
                json.dumps(strategy.versions),
            ),
        )
        self._conn.commit()

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        cur = self._conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        row = cur.fetchone()
        return self._row_to_strategy(row) if row else None

    def list_strategies(self, limit: int = 50, offset: int = 0) -> list[Strategy]:
        cur = self._conn.execute(
            "SELECT * FROM strategies ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        )
        return [self._row_to_strategy(r) for r in cur.fetchall()]

    def insert_strategy_version(self, version: StrategyVersion) -> None:
        import json
        self._conn.execute(
            "INSERT INTO strategy_versions (id, strategy_id, version, definition, hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                version.id, version.strategy_id, version.version,
                json.dumps(version.definition), version.hash,
                version.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_strategy_version(self, version_id: str) -> StrategyVersion | None:
        cur = self._conn.execute("SELECT * FROM strategy_versions WHERE id = ?", (version_id,))
        row = cur.fetchone()
        return self._row_to_strategy_version(row) if row else None

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def insert_experiment(self, experiment: Experiment) -> None:
        import json
        self._conn.execute(
            "INSERT INTO experiments (id, strategy_id, strategy_version, dataset_id, dataset_version, engine_version, config, seed, created_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment.id, experiment.strategy_id, experiment.strategy_version,
                experiment.dataset_id, experiment.dataset_version,
                experiment.engine_version, json.dumps(experiment.config),
                experiment.seed, experiment.created_at.isoformat(),
                experiment.status.value,
            ),
        )
        self._conn.commit()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        cur = self._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cur.fetchone()
        return self._row_to_experiment(row) if row else None

    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        self._conn.execute(
            "UPDATE experiments SET status = ? WHERE id = ?", (status.value, experiment_id)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Simulated bets
    # ------------------------------------------------------------------

    def insert_simulated_bet(self, bet: SimulatedBet) -> None:
        import json
        self._conn.execute(
            "INSERT INTO simulated_bets (id, experiment_id, event_id, market_id, runner_id, side, requested_price, matched_price, stake, liability, commission, result, profit, clv_odds, clv_implied_probability, clv_percent, placed_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bet.id, bet.experiment_id, bet.event_id, bet.market_id,
                bet.runner_id, bet.side.value, bet.requested_price,
                bet.matched_price, bet.stake, bet.liability, bet.commission,
                bet.result.value, bet.profit, bet.clv_odds,
                bet.clv_implied_probability, bet.clv_percent,
                bet.placed_at.isoformat(), json.dumps(bet.metadata),
            ),
        )
        self._conn.commit()

    def insert_simulated_bets_bulk(self, bets: list[SimulatedBet]) -> None:
        import json
        rows = [
            (
                b.id, b.experiment_id, b.event_id, b.market_id,
                b.runner_id, b.side.value, b.requested_price,
                b.matched_price, b.stake, b.liability, b.commission,
                b.result.value, b.profit, b.clv_odds,
                b.clv_implied_probability, b.clv_percent,
                b.placed_at.isoformat(), json.dumps(b.metadata),
            )
            for b in bets
        ]
        self._conn.executemany(
            "INSERT INTO simulated_bets (id, experiment_id, event_id, market_id, runner_id, side, requested_price, matched_price, stake, liability, commission, result, profit, clv_odds, clv_implied_probability, clv_percent, placed_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_bets_for_experiment(self, experiment_id: str) -> list[SimulatedBet]:
        cur = self._conn.execute(
            "SELECT * FROM simulated_bets WHERE experiment_id = ? ORDER BY placed_at", (experiment_id,)
        )
        return [self._row_to_simulated_bet(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Bankroll snapshots
    # ------------------------------------------------------------------

    def insert_bankroll_snapshot(self, snapshot: BankrollSnapshot) -> None:
        self._conn.execute(
            "INSERT INTO bankroll_snapshots (id, experiment_id, timestamp, balance, drawdown, open_exposure, bets_count, cumulative_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.id, snapshot.experiment_id,
                snapshot.timestamp.isoformat(), snapshot.balance,
                snapshot.drawdown, snapshot.open_exposure,
                snapshot.bets_count, snapshot.cumulative_pnl,
            ),
        )
        self._conn.commit()

    def insert_bankroll_snapshots_bulk(self, snapshots: list[BankrollSnapshot]) -> None:
        rows = [
            (
                s.id, s.experiment_id, s.timestamp.isoformat(),
                s.balance, s.drawdown, s.open_exposure,
                s.bets_count, s.cumulative_pnl,
            )
            for s in snapshots
        ]
        self._conn.executemany(
            "INSERT INTO bankroll_snapshots (id, experiment_id, timestamp, balance, drawdown, open_exposure, bets_count, cumulative_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_bankroll_history(self, experiment_id: str) -> list[BankrollSnapshot]:
        cur = self._conn.execute(
            "SELECT * FROM bankroll_snapshots WHERE experiment_id = ? ORDER BY timestamp", (experiment_id,)
        )
        return [self._row_to_bankroll_snapshot(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Paper sessions
    # ------------------------------------------------------------------

    def insert_paper_session(self, session: PaperSession) -> None:
        import json
        self._conn.execute(
            "INSERT INTO paper_sessions (id, strategy_id, strategy_version, starting_bankroll, current_bankroll, status, started_at, risk_limits) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.id, session.strategy_id, session.strategy_version,
                session.starting_bankroll, session.current_bankroll,
                session.status.value, session.started_at.isoformat(),
                json.dumps(session.risk_limits),
            ),
        )
        self._conn.commit()

    def get_paper_session(self, session_id: str) -> PaperSession | None:
        cur = self._conn.execute("SELECT * FROM paper_sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        return self._row_to_paper_session(row) if row else None

    def update_paper_session_status(self, session_id: str, status: PaperSessionStatus) -> None:
        self._conn.execute(
            "UPDATE paper_sessions SET status = ? WHERE id = ?", (status.value, session_id)
        )
        self._conn.commit()

    def insert_paper_position(self, pos: PaperPosition) -> None:
        self._conn.execute(
            "INSERT INTO paper_positions (id, session_id, event_id, market_id, runner_id, side, stake, price, status, profit, placed_at, settled_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pos.id, pos.session_id, pos.event_id, pos.market_id,
                pos.runner_id, pos.side.value, pos.stake, pos.price,
                pos.status.value, pos.profit,
                pos.placed_at.isoformat(),
                pos.settled_at.isoformat() if pos.settled_at else None,
            ),
        )
        self._conn.commit()

    def get_positions_for_session(self, session_id: str) -> list[PaperPosition]:
        cur = self._conn.execute(
            "SELECT * FROM paper_positions WHERE session_id = ? ORDER BY placed_at", (session_id,)
        )
        return [self._row_to_paper_position(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Row → model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_participant(row: sqlite3.Row) -> Participant:
        import json
        return Participant(
            id=row["id"],
            name=row["name"],
            sport=row["sport"],
            ranking=row["ranking"],
            elo_rating=row["elo_rating"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row, participants: list[Participant] | None = None) -> Event:
        from datetime import datetime
        st = row["start_time"]
        return Event(
            id=row["id"],
            sport=row["sport"],
            competition=row["competition"],
            start_time=datetime.fromisoformat(st) if st else None,
            venue=row["venue"],
            status=row["status"],
            result_status=row["result_status"],
            home_away=bool(row["home_away"]),
            participants=participants or [],
        )

    @staticmethod
    def _row_to_market(row: sqlite3.Row) -> Market:
        return Market(
            id=row["id"],
            event_id=row["event_id"],
            market_type=row["market_type"],
            market_name=row["market_name"],
            commission_model=row["commission_model"],
            status=row["status"],
        )

    @staticmethod
    def _row_to_runner(row: sqlite3.Row) -> Runner:
        import json
        return Runner(
            id=row["id"],
            market_id=row["market_id"],
            participant_id=row["participant_id"],
            name=row["name"],
            result=row["result"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_price_snapshot(row: sqlite3.Row) -> PriceSnapshot:
        from datetime import datetime
        return PriceSnapshot(
            id=row["id"],
            market_id=row["market_id"],
            runner_id=row["runner_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            back_price=row["back_price"],
            back_available=row["back_available"],
            lay_price=row["lay_price"],
            lay_available=row["lay_available"],
            traded_volume=row["traded_volume"],
        )

    @staticmethod
    def _row_to_dataset(row: sqlite3.Row) -> Dataset:
        from datetime import datetime
        return Dataset(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            sport=row["sport"],
            source=row["source"],
            event_count=row["event_count"],
            market_count=row["market_count"],
            time_range=row["time_range"],
            schema_hash=row["schema_hash"],
            data_hash=row["data_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    @staticmethod
    def _row_to_strategy(row: sqlite3.Row) -> Strategy:
        import json
        from datetime import datetime
        return Strategy(
            id=row["id"],
            name=row["name"],
            sport=row["sport"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            versions=json.loads(row["versions"] or "[]"),
        )

    @staticmethod
    def _row_to_strategy_version(row: sqlite3.Row) -> StrategyVersion:
        import json
        from datetime import datetime
        return StrategyVersion(
            id=row["id"],
            strategy_id=row["strategy_id"],
            version=row["version"],
            definition=json.loads(row["definition"] or "{}"),
            hash=row["hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        import json
        from datetime import datetime
        return Experiment(
            id=row["id"],
            strategy_id=row["strategy_id"],
            strategy_version=row["strategy_version"],
            dataset_id=row["dataset_id"],
            dataset_version=row["dataset_version"],
            engine_version=row["engine_version"],
            config=json.loads(row["config"] or "{}"),
            seed=row["seed"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            status=row["status"],
        )

    @staticmethod
    def _row_to_simulated_bet(row: sqlite3.Row) -> SimulatedBet:
        import json
        from datetime import datetime
        return SimulatedBet(
            id=row["id"],
            experiment_id=row["experiment_id"],
            event_id=row["event_id"],
            market_id=row["market_id"],
            runner_id=row["runner_id"],
            side=row["side"],
            requested_price=row["requested_price"],
            matched_price=row["matched_price"],
            stake=row["stake"],
            liability=row["liability"],
            commission=row["commission"],
            result=row["result"],
            profit=row["profit"],
            clv_odds=row["clv_odds"],
            clv_implied_probability=row["clv_implied_probability"],
            clv_percent=row["clv_percent"],
            placed_at=datetime.fromisoformat(row["placed_at"]) if row["placed_at"] else None,
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_bankroll_snapshot(row: sqlite3.Row) -> BankrollSnapshot:
        from datetime import datetime
        return BankrollSnapshot(
            id=row["id"],
            experiment_id=row["experiment_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            balance=row["balance"],
            drawdown=row["drawdown"],
            open_exposure=row["open_exposure"],
            bets_count=row["bets_count"],
            cumulative_pnl=row["cumulative_pnl"],
        )

    @staticmethod
    def _row_to_paper_session(row: sqlite3.Row) -> PaperSession:
        import json
        from datetime import datetime
        return PaperSession(
            id=row["id"],
            strategy_id=row["strategy_id"],
            strategy_version=row["strategy_version"],
            starting_bankroll=row["starting_bankroll"],
            current_bankroll=row["current_bankroll"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            risk_limits=json.loads(row["risk_limits"] or "{}"),
        )

    @staticmethod
    def _row_to_paper_position(row: sqlite3.Row) -> PaperPosition:
        from datetime import datetime
        return PaperPosition(
            id=row["id"],
            session_id=row["session_id"],
            event_id=row["event_id"],
            market_id=row["market_id"],
            runner_id=row["runner_id"],
            side=row["side"],
            stake=row["stake"],
            price=row["price"],
            status=row["status"],
            profit=row["profit"],
            placed_at=datetime.fromisoformat(row["placed_at"]) if row["placed_at"] else None,
            settled_at=datetime.fromisoformat(row["settled_at"]) if row["settled_at"] else None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
