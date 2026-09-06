from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb


@dataclass
class Dataset:
    id: str
    name: str
    sport: str
    table_name: str
    row_count: int
    market_count: int
    columns: list[str]
    data_hash: str
    imported_at: str
    metadata: dict = field(default_factory=dict)


class DatasetImporter:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self._db_path)
        return self._conn

    def import_csv(self, path: str | Path, sport: str, name: str) -> Dataset:
        path = Path(path)
        data_hash = self._compute_hash(path)
        ds_id = self._generate_id()

        conn = self._get_conn()
        table_name = f"dataset_{ds_id}"

        result = conn.execute(
            f"SELECT * FROM read_csv_auto('{path}')"
        )
        columns = [desc[0] for desc in result.description]
        conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path}')"
        )

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        market_count = self._count_markets(conn, table_name)

        return Dataset(
            id=ds_id,
            name=name,
            sport=sport,
            table_name=table_name,
            row_count=row_count,
            market_count=market_count,
            columns=columns,
            data_hash=data_hash,
            imported_at=datetime.utcnow().isoformat(),
            metadata={"source_path": str(path), "format": "csv"},
        )

    def import_parquet(self, path: str | Path, sport: str, name: str) -> Dataset:
        path = Path(path)
        data_hash = self._compute_hash(path)
        ds_id = self._generate_id()

        conn = self._get_conn()
        table_name = f"dataset_{ds_id}"

        result = conn.execute(
            f"SELECT * FROM read_parquet('{path}')"
        )
        columns = [desc[0] for desc in result.description]
        conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{path}')"
        )

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        market_count = self._count_markets(conn, table_name)

        return Dataset(
            id=ds_id,
            name=name,
            sport=sport,
            table_name=table_name,
            row_count=row_count,
            market_count=market_count,
            columns=columns,
            data_hash=data_hash,
            imported_at=datetime.utcnow().isoformat(),
            metadata={"source_path": str(path), "format": "parquet"},
        )

    def import_json(self, path: str | Path, sport: str, name: str) -> Dataset:
        path = Path(path)
        data_hash = self._compute_hash(path)
        ds_id = self._generate_id()

        conn = self._get_conn()
        table_name = f"dataset_{ds_id}"
        with open(path) as f:
            raw = json.load(f)

        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = raw.get("data", raw.get("records", [raw]))
        else:
            records = [raw]

        if not records:
            raise ValueError(f"No records found in {path}")

        columns = list(records[0].keys())
        conn.execute(
            f"CREATE TABLE {table_name} ({', '.join(f'{c} VARCHAR' for c in columns)})"
        )
        for rec in records:
            vals = [rec.get(c) for c in columns]
            placeholders = ", ".join(["?" for _ in columns])
            conn.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", vals)

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        market_count = self._count_markets(conn, table_name)

        return Dataset(
            id=ds_id,
            name=name,
            sport=sport,
            table_name=table_name,
            row_count=row_count,
            market_count=market_count,
            columns=columns,
            data_hash=data_hash,
            imported_at=datetime.utcnow().isoformat(),
            metadata={"source_path": str(path), "format": "json"},
        )

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _compute_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _generate_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _count_markets(conn: duckdb.DuckDBPyConnection, table_name: str) -> int:
        try:
            result = conn.execute(
                f"SELECT COUNT(DISTINCT event_id) FROM {table_name}"
            ).fetchone()
            return result[0] if result else 0
        except Exception:
            try:
                result = conn.execute(
                    f"SELECT COUNT(DISTINCT market_id) FROM {table_name}"
                ).fetchone()
                return result[0] if result else 0
            except Exception:
                return 0
