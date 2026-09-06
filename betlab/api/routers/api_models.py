from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from betlab.ml.column_analyzer import ColumnAnalyzer
from betlab.ml.schemas import (
    ModelBuilderState,
    ModelManifest,
    ModelType,
    TargetConfig,
    TargetType,
)
from betlab.ml.trainer import ModelTrainer
from betlab.ml.strategy_converter import StrategyConverter

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_dataframe(dataset_id: str) -> pd.DataFrame:
    dataset_path = Path("datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("events", [])
    if not events:
        raise HTTPException(status_code=400, detail="Dataset contains no events")
    return pd.DataFrame(events)


def _get_registry():
    from betlab.ml.registry import ModelRegistry
    return ModelRegistry()


# ── Column / preview endpoints ─────────────────────────────────────────────────

@router.get("/datasets/{dataset_id}/columns")
def get_columns(dataset_id: str) -> dict[str, Any]:
    df = _load_dataframe(dataset_id)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)
    return {
        "dataset_id": dataset_id,
        "row_count": preview.row_count,
        "columns": [c.model_dump() for c in preview.columns],
    }


@router.get("/datasets/{dataset_id}/preview")
def get_preview(dataset_id: str, limit: int = 10) -> dict[str, Any]:
    df = _load_dataframe(dataset_id)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)
    return {
        "dataset_id": dataset_id,
        "row_count": preview.row_count,
        "columns": [c.model_dump() for c in preview.columns],
        "preview_rows": preview.preview_rows[:limit],
        "warnings": preview.warnings,
    }


# ── Leakage check ─────────────────────────────────────────────────────────────

@router.post("/datasets/{dataset_id}/leakage-check")
def leakage_check(dataset_id: str) -> dict[str, Any]:
    df = _load_dataframe(dataset_id)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)
    leakage_cols = analyzer.detect_leakage_columns(preview.columns)
    return {
        "dataset_id": dataset_id,
        "leakage_columns": leakage_cols,
        "count": len(leakage_cols),
    }


# ── Train ──────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    state: ModelBuilderState


@router.post("/train")
def train_model(req: TrainRequest) -> dict[str, Any]:
    state = req.state

    if not state.dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    if not state.features:
        raise HTTPException(status_code=400, detail="At least one feature is required")

    df = _load_dataframe(state.dataset_id)
    feature_columns = [f.column for f in state.features]

    trainer = ModelTrainer(
        model_config=state.model,
        preprocessing_config=state.preprocessing,
        validation_config=state.validation,
    )

    result = trainer.train(df, state.target, feature_columns)

    registry = _get_registry()
    metrics_hash = registry.compute_metrics_hash(result.metrics)

    model_id = result.model_id
    model_path = Path(f".betlab/artifacts/{model_id}.joblib")

    manifest = ModelManifest(
        id=model_id,
        name=f"model-{model_id[:8]}",
        version="1.0.0",
        target_config=state.target,
        feature_columns=feature_columns,
        model_type=state.model.type,
        model_parameters=state.model.parameters,
        preprocessing_config=state.preprocessing,
        validation_config=state.validation,
        dataset_id=state.dataset_id,
        dataset_version=state.dataset_version,
        metrics_hash=metrics_hash,
    )

    registry.save_model(manifest, str(model_path))
    registry.save_model_metrics(model_id, result.metrics)
    registry.update_status(model_id, "trained")

    return {
        "model_id": model_id,
        "status": "trained",
        "metrics": result.metrics,
        "feature_importance": result.feature_importance,
        "calibration": result.calibration,
        "warnings": result.warnings,
        "execution_time_ms": result.execution_time_ms,
    }


# ── Model CRUD ─────────────────────────────────────────────────────────────────

@router.get("/{model_id}")
def get_model(model_id: str) -> dict[str, Any]:
    registry = _get_registry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return manifest.model_dump()


@router.get("/{model_id}/metrics")
def get_metrics(model_id: str) -> dict[str, Any]:
    registry = _get_registry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "metrics": registry.get_model_metrics(model_id)}


@router.get("/{model_id}/feature-importance")
def get_feature_importance(model_id: str) -> dict[str, Any]:
    registry = _get_registry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Model not found")
    metrics = registry.get_model_metrics(model_id)
    importance = metrics.get("feature_importance", [])
    return {"model_id": model_id, "feature_importance": importance}


@router.get("/{model_id}/calibration")
def get_calibration(model_id: str) -> dict[str, Any]:
    registry = _get_registry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Model not found")
    metrics = registry.get_model_metrics(model_id)
    calibration = metrics.get("calibration", {})
    return {"model_id": model_id, "calibration": calibration}


class ExportStrategyRequest(BaseModel):
    probability_threshold: float = 0.55
    minimum_edge: float = 0.03
    minimum_odds: float = 1.2
    maximum_odds: float = 10.0
    market_side: str = "back"
    staking_method: str = "kelly"
    kelly_fraction: float = 0.25
    maximum_stake_percent: float = 5.0


# ── Export strategy ────────────────────────────────────────────────────────────

@router.post("/{model_id}/export-strategy")
def export_strategy(model_id: str, betting_config: ExportStrategyRequest | None = None) -> dict[str, Any]:
    registry = _get_registry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Model not found")

    from betlab.ml.schemas import BettingConversionConfig

    if betting_config is not None:
        config = BettingConversionConfig(**betting_config.model_dump())
    else:
        config = BettingConversionConfig()

    converter = StrategyConverter()
    strategy = converter.convert_to_strategy(
        model=None,
        feature_columns=manifest.feature_columns,
        betting_config=config,
        model_metrics=registry.get_model_metrics(model_id),
    )

    registry.save_model_strategy(model_id, strategy)
    return {"model_id": model_id, "strategy": strategy}


# ── Backtest ───────────────────────────────────────────────────────────────────

@router.post("/{model_id}/backtest")
def run_backtest(model_id: str) -> dict[str, Any]:
    registry = _get_registry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Model not found")

    strategy = registry.get_model_strategy(model_id)
    if not strategy:
        return {
            "model_id": model_id,
            "status": "error",
            "message": "No strategy exported yet. Call export-strategy first.",
        }

    return {
        "model_id": model_id,
        "status": "completed",
        "strategy": strategy,
        "message": "Backtest placeholder — full integration pending backtester refactor",
    }


# ── Model list ─────────────────────────────────────────────────────────────────

@router.get("/")
def list_models(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    registry = _get_registry()
    models = registry.list_models(limit=limit, offset=offset)
    return [m.model_dump() for m in models]
