from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class DatasetVersion:
    version_id: str
    dataset_id: str
    version_number: int
    description: str
    data_hash: str
    row_count: int
    created_at: str
    metadata: dict = field(default_factory=dict)


class DatasetVersioner:
    def __init__(self, db_path: str | Path = ".betlab/versions.db"):
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            from pathlib import Path as _P
            _P(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                description TEXT,
                data_hash TEXT,
                row_count INTEGER,
                created_at TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(dataset_id, version_number)
            )
        """)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def create_version(self, dataset, description: str = "") -> DatasetVersion:
        version_id = uuid.uuid4().hex[:12]
        data_hash = dataset.data_hash

        row = self._conn.execute(
            "SELECT MAX(version_number) FROM dataset_versions WHERE dataset_id = ?",
            (dataset.id,),
        ).fetchone()
        next_version = (row[0] or 0) + 1

        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset.id,
            version_number=next_version,
            description=description,
            data_hash=data_hash,
            row_count=dataset.row_count,
            created_at=datetime.utcnow().isoformat(),
        )

        self._conn.execute(
            """INSERT INTO dataset_versions
               (version_id, dataset_id, version_number, description, data_hash, row_count, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version.version_id,
                version.dataset_id,
                version.version_number,
                version.description,
                version.data_hash,
                version.row_count,
                version.created_at,
                json.dumps(version.metadata),
            ),
        )
        self._conn.commit()

        return version

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        rows = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version_number DESC",
            (dataset_id,),
        ).fetchall()

        return [
            DatasetVersion(
                version_id=r[0],
                dataset_id=r[1],
                version_number=r[2],
                description=r[3] or "",
                data_hash=r[4] or "",
                row_count=r[5] or 0,
                created_at=r[6],
                metadata=json.loads(r[7]) if r[7] else {},
            )
            for r in rows
        ]
