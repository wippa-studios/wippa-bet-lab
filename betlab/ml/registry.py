from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from betlab.ml.schemas import ModelManifest


_REGISTRY_SQL = """\
CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    target_config TEXT NOT NULL,
    feature_columns TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_parameters TEXT NOT NULL,
    preprocessing_config TEXT NOT NULL,
    validation_config TEXT NOT NULL,
    dataset_id TEXT DEFAULT '',
    dataset_version TEXT DEFAULT '',
    metrics_hash TEXT DEFAULT '',
    status TEXT DEFAULT 'draft',
    created_at TEXT NOT NULL,
    metrics TEXT DEFAULT '{}',
    strategy TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    model_id TEXT PRIMARY KEY,
    artifact_path TEXT NOT NULL,
    file_size INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES model_registry(id)
);
"""


class ModelRegistry:
    def __init__(self, db_path: str = ".betlab/models.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self) -> None:
        self._conn.executescript(_REGISTRY_SQL)
        self._conn.commit()

    def save_model(
        self,
        manifest: ModelManifest,
        model_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        artifacts_dir = self._path.parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        src = Path(model_path)
        dest = artifacts_dir / f"{manifest.id}.joblib"
        if src.exists():
            shutil.copy2(str(src), str(dest))

        self._conn.execute(
            "INSERT OR REPLACE INTO model_registry "
            "(id, name, version, target_config, feature_columns, model_type, "
            "model_parameters, preprocessing_config, validation_config, "
            "dataset_id, dataset_version, metrics_hash, status, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                manifest.id,
                manifest.name,
                manifest.version,
                manifest.target_config.model_dump_json(),
                json.dumps(manifest.feature_columns),
                manifest.model_type.value,
                json.dumps(manifest.model_parameters),
                manifest.preprocessing_config.model_dump_json(),
                manifest.validation_config.model_dump_json(),
                manifest.dataset_id,
                manifest.dataset_version,
                manifest.metrics_hash,
                "draft",
                manifest.created_at.isoformat(),
                json.dumps(metadata or {}),
            ),
        )

        self._conn.execute(
            "INSERT OR REPLACE INTO model_artifacts (model_id, artifact_path, file_size, created_at) "
            "VALUES (?, ?, ?, ?)",
            (manifest.id, str(dest), src.stat().st_size if src.exists() else 0, datetime.utcnow().isoformat()),
        )

        self._conn.commit()
        return manifest.id

    def get_model(self, model_id: str) -> ModelManifest | None:
        cur = self._conn.execute("SELECT * FROM model_registry WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_manifest(row)

    def list_models(self, limit: int = 50, offset: int = 0) -> list[ModelManifest]:
        cur = self._conn.execute(
            "SELECT * FROM model_registry ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_manifest(r) for r in cur.fetchall()]

    def get_model_artifact(self, model_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT artifact_path FROM model_artifacts WHERE model_id = ?", (model_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return row["artifact_path"]

    def delete_model(self, model_id: str) -> None:
        self._conn.execute("DELETE FROM model_artifacts WHERE model_id = ?", (model_id,))
        self._conn.execute("DELETE FROM model_registry WHERE id = ?", (model_id,))
        self._conn.commit()

    def update_status(self, model_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE model_registry SET status = ? WHERE id = ?", (status, model_id)
        )
        self._conn.commit()

    def get_model_metrics(self, model_id: str) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT metrics FROM model_registry WHERE id = ?", (model_id,)
        )
        row = cur.fetchone()
        if row is None:
            return {}
        return json.loads(row["metrics"] or "{}")

    def save_model_metrics(self, model_id: str, metrics: dict[str, Any]) -> None:
        self._conn.execute(
            "UPDATE model_registry SET metrics = ? WHERE id = ?",
            (json.dumps(metrics), model_id),
        )
        self._conn.commit()

    def save_model_strategy(self, model_id: str, strategy: dict[str, Any]) -> None:
        self._conn.execute(
            "UPDATE model_registry SET strategy = ? WHERE id = ?",
            (json.dumps(strategy), model_id),
        )
        self._conn.commit()

    def get_model_strategy(self, model_id: str) -> dict[str, Any]:
        cur = self._conn.execute(
            "SELECT strategy FROM model_registry WHERE id = ?", (model_id,)
        )
        row = cur.fetchone()
        if row is None:
            return {}
        return json.loads(row["strategy"] or "{}")

    @staticmethod
    def compute_metrics_hash(metrics: dict[str, Any]) -> str:
        canonical = json.dumps(metrics, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def _row_to_manifest(row: sqlite3.Row) -> ModelManifest:
        from betlab.ml.schemas import ModelType
        return ModelManifest(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            target_config=json.loads(row["target_config"]),
            feature_columns=json.loads(row["feature_columns"]),
            model_type=ModelType(row["model_type"]),
            model_parameters=json.loads(row["model_parameters"] or "{}"),
            preprocessing_config=json.loads(row["preprocessing_config"]),
            validation_config=json.loads(row["validation_config"]),
            dataset_id=row["dataset_id"],
            dataset_version=row["dataset_version"],
            metrics_hash=row["metrics_hash"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )

    def close(self) -> None:
        self._conn.close()
