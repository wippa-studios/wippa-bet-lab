from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ExperimentCreateRequest(BaseModel):
    strategy_id: str
    dataset_id: str
    seed: int = 42


class ExperimentResponse(BaseModel):
    id: str
    strategy_id: str
    dataset_id: str
    status: str
    seed: int
    created_at: str


@router.get("/")
def list_experiments() -> list[ExperimentResponse]:
    from betlab.core.db.database import Database
    db = Database()
    cur = db._conn.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50")
    rows = cur.fetchall()
    db.close()
    return [
        ExperimentResponse(
            id=r["id"], strategy_id=r["strategy_id"], dataset_id=r["dataset_id"],
            status=r["status"], seed=r["seed"], created_at=r["created_at"][:19] if r["created_at"] else "",
        )
        for r in rows
    ]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str) -> ExperimentResponse:
    from betlab.core.db.database import Database
    db = Database()
    cur = db._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = cur.fetchone()
    db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse(
        id=row["id"], strategy_id=row["strategy_id"], dataset_id=row["dataset_id"],
        status=row["status"], seed=row["seed"],
        created_at=row["created_at"][:19] if row["created_at"] else "",
    )


@router.post("/")
def create_experiment(req: ExperimentCreateRequest) -> ExperimentResponse:
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import Experiment, ExperimentStatus, StrategyDefinition

    db = Database()
    ds = db.get_dataset(req.dataset_id)
    if ds is None:
        db.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
    experiment = Experiment(
        id=experiment_id,
        strategy_id=req.strategy_id,
        strategy_version="1.0.0",
        dataset_id=req.dataset_id,
        dataset_version=ds.version,
        config={"seed": req.seed},
        seed=req.seed,
    )
    db.insert_experiment(experiment)
    db.update_experiment_status(experiment_id, ExperimentStatus.running)

    strategy_def = StrategyDefinition(
        id=req.strategy_id, version="1.0.0", name="api-created",
        sport=ds.sport, market="match_odds",
    )

    from betlab.backtester.engine import BacktestEngine
    engine = BacktestEngine(strategy=strategy_def, seed=req.seed)
    engine.run()

    db.update_experiment_status(experiment_id, ExperimentStatus.completed)
    db.close()

    return ExperimentResponse(
        id=experiment_id, strategy_id=req.strategy_id, dataset_id=req.dataset_id,
        status="completed", seed=req.seed, created_at=str(experiment.created_at),
    )


@router.get("/{experiment_id}/progress")
def get_progress(experiment_id: str) -> dict:
    from betlab.core.db.database import Database
    db = Database()
    cur = db._conn.execute("SELECT status FROM experiments WHERE id = ?", (experiment_id,))
    row = cur.fetchone()
    db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"experiment_id": experiment_id, "status": row["status"], "progress": 100 if row["status"] == "completed" else 0}


@router.get("/{experiment_id}/report")
def get_report(experiment_id: str) -> dict:
    from betlab.backtester.engine import BacktestResult
    from betlab.reporting.json_report import generate_json_report
    bt = BacktestResult(experiment_id=experiment_id)
    return generate_json_report(experiment_id, bt)
