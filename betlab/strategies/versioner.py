from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from betlab.strategies.builder import StrategyDefinition


@dataclass
class StrategyVersion:
    version_id: str
    strategy_id: str
    version_number: int
    strategy_hash: str
    created_at: str
    definition: dict = field(default_factory=dict)


class StrategyVersioner:
    def __init__(self, db_path: str | Path = ".betlab/strategy_versions.db"):
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            from pathlib import Path as _P
            _P(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_versions (
                version_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                strategy_hash TEXT,
                created_at TEXT NOT NULL,
                definition TEXT,
                UNIQUE(strategy_id, version_number)
            )
        """)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def compute_hash(self, strategy_def: StrategyDefinition) -> str:
        data = json.dumps(
            {
                "name": strategy_def.name,
                "sport": strategy_def.sport,
                "market": strategy_def.market,
                "filters": [
                    {"field": f.field, "operator": f.operator, "value": f.value}
                    for f in strategy_def.filters
                ],
                "signal": (
                    {
                        "field": strategy_def.signal.field,
                        "operator": strategy_def.signal.operator,
                        "value": strategy_def.signal.value,
                    }
                    if strategy_def.signal
                    else None
                ),
                "selection": (
                    {"side": strategy_def.selection.side, "runner": strategy_def.selection.runner}
                    if strategy_def.selection
                    else None
                ),
                "price": (
                    {
                        "source": strategy_def.price.source,
                        "minimum": strategy_def.price.minimum,
                        "maximum": strategy_def.price.maximum,
                    }
                    if strategy_def.price
                    else None
                ),
                "staking": {
                    "method": strategy_def.staking.method,
                    "kwargs": strategy_def.staking.kwargs,
                },
                "risk": {
                    "max_open_exposure": strategy_def.risk.max_open_exposure,
                    "max_daily_loss": strategy_def.risk.max_daily_loss,
                },
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def create_version(self, strategy_def: StrategyDefinition) -> StrategyVersion:
        version_id = uuid.uuid4().hex[:12]
        strategy_hash = self.compute_hash(strategy_def)

        row = self._conn.execute(
            "SELECT MAX(version_number) FROM strategy_versions WHERE strategy_id = ?",
            (strategy_def.id,),
        ).fetchone()
        next_version = (row[0] or 0) + 1

        definition_dict = {
            "name": strategy_def.name,
            "sport": strategy_def.sport,
            "market": strategy_def.market,
            "filters": [
                {"field": f.field, "operator": f.operator, "value": f.value}
                for f in strategy_def.filters
            ],
        }

        version = StrategyVersion(
            version_id=version_id,
            strategy_id=strategy_def.id,
            version_number=next_version,
            strategy_hash=strategy_hash,
            created_at=datetime.utcnow().isoformat(),
            definition=definition_dict,
        )

        self._conn.execute(
            """INSERT INTO strategy_versions
               (version_id, strategy_id, version_number, strategy_hash, created_at, definition)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                version.version_id,
                version.strategy_id,
                version.version_number,
                version.strategy_hash,
                version.created_at,
                json.dumps(version.definition),
            ),
        )
        self._conn.commit()

        return version

    def list_versions(self, strategy_id: str) -> list[StrategyVersion]:
        rows = self._conn.execute(
            "SELECT * FROM strategy_versions WHERE strategy_id = ? ORDER BY version_number DESC",
            (strategy_id,),
        ).fetchall()

        return [
            StrategyVersion(
                version_id=r[0],
                strategy_id=r[1],
                version_number=r[2],
                strategy_hash=r[3] or "",
                created_at=r[4],
                definition=json.loads(r[5]) if r[5] else {},
            )
            for r in rows
        ]
