from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from betlab.api import app

client = TestClient(app)


class TestRootEndpoints:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Wippa Bet Lab API"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestDatasetAPI:
    def test_list_datasets(self):
        resp = client.get("/api/datasets/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_dataset_not_found(self):
        resp = client.get("/api/datasets/nonexistent")
        assert resp.status_code == 404

    def test_import_dataset(self, tmp_path):
        data_file = tmp_path / "test_events.json"
        events = [{"id": "e1", "start_time": "2025-01-01T00:00:00", "markets": []}]
        data_file.write_text(json.dumps(events))

        with open(data_file, "rb") as f:
            resp = client.post(
                "/api/datasets/import",
                files={"file": ("test.json", f, "application/json")},
                params={"sport": "tennis", "name": "test-dataset"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "imported"
        assert body["event_count"] == 1

    def test_validate_dataset_not_found(self):
        resp = client.post("/api/datasets/nonexistent/validate")
        assert resp.status_code == 404

    def test_snapshot_dataset_not_found(self):
        resp = client.post("/api/datasets/nonexistent/snapshot")
        assert resp.status_code == 404


class TestStrategyAPI:
    def test_list_strategies(self):
        resp = client.get("/api/strategies/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_strategy_not_found(self):
        resp = client.get("/api/strategies/nonexistent")
        assert resp.status_code == 404

    def test_create_strategy(self):
        resp = client.post(
            "/api/strategies/",
            json={"name": "Test Strategy", "sport": "tennis", "market": "match_odds"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Test Strategy"
        assert "id" in body

    def test_validate_strategy_not_found(self):
        resp = client.post("/api/strategies/nonexistent/validate")
        assert resp.status_code == 404

    def test_clone_strategy_not_found(self):
        resp = client.post("/api/strategies/nonexistent/clone")
        assert resp.status_code == 404


class TestExperimentAPI:
    def test_list_experiments(self):
        resp = client.get("/api/experiments/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_experiment_not_found(self):
        resp = client.get("/api/experiments/nonexistent")
        assert resp.status_code == 404

    def test_create_experiment_no_dataset(self):
        resp = client.post(
            "/api/experiments/",
            json={"strategy_id": "s1", "dataset_id": "nonexistent", "seed": 42},
        )
        assert resp.status_code == 404

    def test_progress_not_found(self):
        resp = client.get("/api/experiments/nonexistent/progress")
        assert resp.status_code == 404

    def test_report(self):
        resp = client.get("/api/experiments/exp-test/report")
        assert resp.status_code == 200
        assert "summary" in resp.json()


class TestPaperAPI:
    def test_list_sessions(self):
        resp = client.get("/api/paper/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_session(self):
        resp = client.post(
            "/api/paper/sessions",
            json={"strategy_id": "nonexistent", "bankroll": 1000.0},
        )
        assert resp.status_code == 404

    def test_pause_session_not_found(self):
        resp = client.post("/api/paper/sessions/nonexistent/pause")
        assert resp.status_code == 404

    def test_stop_session_not_found(self):
        resp = client.post("/api/paper/sessions/nonexistent/stop")
        assert resp.status_code == 404

    def test_positions_not_found(self):
        resp = client.get("/api/paper/sessions/nonexistent/positions")
        assert resp.status_code == 404

    def test_metrics_not_found(self):
        resp = client.get("/api/paper/sessions/nonexistent/metrics")
        assert resp.status_code == 404
