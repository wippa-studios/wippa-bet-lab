from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import duckdb

from betlab.datasets.importer import Dataset


@dataclass
class SnapshotMeta:
    snapshot_id: str
    source_dataset_id: str
    table_name: str
    row_count: int
    time_range: tuple[str, str] | None
    created_at: str


class DatasetSnapshot:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._snapshots: dict[str, SnapshotMeta] = {}

    def create_snapshot(
        self,
        dataset: Dataset,
        time_range: tuple[str, str] | None = None,
    ) -> SnapshotMeta:
        snap_id = uuid.uuid4().hex[:12]
        snap_table = f"snapshot_{snap_id}"

        if time_range and "start_time" in dataset.columns:
            start, end = time_range
            query = f"""
                CREATE TABLE {snap_table} AS
                SELECT * FROM {dataset.table_name}
                WHERE start_time >= '{start}' AND start_time <= '{end}'
            """
        else:
            query = f"CREATE TABLE {snap_table} AS SELECT * FROM {dataset.table_name}"

        self._conn.execute(query)
        row_count = self._conn.execute(f"SELECT COUNT(*) FROM {snap_table}").fetchone()[0]

        meta = SnapshotMeta(
            snapshot_id=snap_id,
            source_dataset_id=dataset.id,
            table_name=snap_table,
            row_count=row_count,
            time_range=time_range,
            created_at=datetime.utcnow().isoformat(),
        )
        self._snapshots[snap_id] = meta
        return meta

    def get_snapshot(self, snapshot_id: str) -> tuple[SnapshotMeta, list[dict]]:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot {snapshot_id} not found")

        meta = self._snapshots[snapshot_id]
        rows = self._conn.execute(f"SELECT * FROM {meta.table_name}").fetchall()
        columns = [
            desc[0]
            for desc in self._conn.execute(f"SELECT * FROM {meta.table_name} LIMIT 0").description
        ]
        records = [dict(zip(columns, row)) for row in rows]
        return meta, records
