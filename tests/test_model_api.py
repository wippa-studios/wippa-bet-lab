from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from betlab.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean registry DB."""
    db_path = Path(".betlab/models.db")
    if db_path.exists():
        db_path.unlink()
    yield
    if db_path.exists():
        db_path.unlink()


class TestModelColumns:
    def test_columns_not_found(self):
        resp = client.get("/api/models/datasets/nonexistent/columns")
        assert resp.status_code == 404

    def test_columns_with_data(self, tmp_path):
        ds_file = Path("datasets") / "test-ds.json"
        ds_file.parent.mkdir(parents=True, exist_ok=True)
        ds_file.write_text(json.dumps([
            {"id": "e1", "home_score": 2, "away_score": 1, "result": "home"},
            {"id": "e2", "home_score": 0, "away_score": 0, "result": "draw"},
        ]))
        try:
            resp = client.get("/api/models/datasets/test-ds/columns")
            assert resp.status_code == 200
            body = resp.json()
            assert body["dataset_id"] == "test-ds"
            assert "columns" in body
            assert len(body["columns"]) >= 1
        finally:
            ds_file.unlink(missing_ok=True)


class TestModelPreview:
    def test_preview_not_found(self):
        resp = client.get("/api/models/datasets/nonexistent/preview")
        assert resp.status_code == 404

    def test_preview_with_data(self, tmp_path):
        ds_file = Path("datasets") / "test-ds2.json"
        ds_file.parent.mkdir(parents=True, exist_ok=True)
        ds_file.write_text(json.dumps([
            {"id": "e1", "score": 10},
            {"id": "e2", "score": 20},
        ]))
        try:
            resp = client.get("/api/models/datasets/test-ds2/preview")
            assert resp.status_code == 200
            body = resp.json()
            assert body["row_count"] == 2
            assert len(body["preview_rows"]) >= 1
        finally:
            ds_file.unlink(missing_ok=True)


class TestModelLeakage:
    def test_leakage_check(self, tmp_path):
        ds_file = Path("datasets") / "test-leak.json"
        ds_file.parent.mkdir(parents=True, exist_ok=True)
        ds_file.write_text(json.dumps([
            {"id": "e1", "result": "win", "profit": 5.0},
        ]))
        try:
            resp = client.post("/api/models/datasets/test-leak/leakage-check")
            assert resp.status_code == 200
            body = resp.json()
            assert "leakage_columns" in body
            assert body["count"] >= 1
        finally:
            ds_file.unlink(missing_ok=True)


class TestModelList:
    def test_list_empty(self):
        resp = client.get("/api/models/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestModelNotFound:
    def test_get_model_not_found(self):
        resp = client.get("/api/models/nonexistent")
        assert resp.status_code == 404

    def test_get_metrics_not_found(self):
        resp = client.get("/api/models/nonexistent/metrics")
        assert resp.status_code == 404

    def test_get_feature_importance_not_found(self):
        resp = client.get("/api/models/nonexistent/feature-importance")
        assert resp.status_code == 404
