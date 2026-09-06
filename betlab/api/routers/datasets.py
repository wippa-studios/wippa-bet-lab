from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

router = APIRouter()


class DatasetResponse(BaseModel):
    id: str
    name: str
    version: str
    sport: str
    source: str = ""
    event_count: int = 0
    market_count: int = 0
    time_range: str = ""
    created_at: str = ""


class DatasetImportRequest(BaseModel):
    sport: str
    name: str


@router.get("/")
def list_datasets() -> list[DatasetResponse]:
    from betlab.core.db.database import Database
    db = Database()
    datasets = db.list_datasets()
    db.close()
    return [
        DatasetResponse(
            id=ds.id, name=ds.name, version=ds.version, sport=ds.sport.value,
            source=ds.source, event_count=ds.event_count, market_count=ds.market_count,
            time_range=ds.time_range, created_at=str(ds.created_at),
        )
        for ds in datasets
    ]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> DatasetResponse:
    from betlab.core.db.database import Database
    db = Database()
    ds = db.get_dataset(dataset_id)
    db.close()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse(
        id=ds.id, name=ds.name, version=ds.version, sport=ds.sport.value,
        source=ds.source, event_count=ds.event_count, market_count=ds.market_count,
        time_range=ds.time_range, created_at=str(ds.created_at),
    )


@router.post("/import")
async def import_dataset(file: UploadFile = File(...), sport: str = "tennis", name: str = "imported"):
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    events = data if isinstance(data, list) else data.get("events", [])

    from betlab.core.db.database import Database
    from betlab.core.schemas.models import Dataset, Sport

    db = Database()
    dataset_id = f"ds-{uuid.uuid4().hex[:8]}"
    sport_enum = Sport(sport)
    ds = Dataset(
        id=dataset_id, name=name, version="1.0.0", sport=sport_enum,
        source=file.filename or "", event_count=len(events),
    )
    ds.compute_data_hash()
    db.insert_dataset(ds)
    db.close()

    return {"id": dataset_id, "name": name, "event_count": len(events), "status": "imported"}


@router.post("/{dataset_id}/validate")
def validate_dataset(dataset_id: str) -> dict:
    from betlab.core.db.database import Database
    db = Database()
    ds = db.get_dataset(dataset_id)
    db.close()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "valid": True,
        "checks": {
            "exists": True,
            "has_events": ds.event_count > 0,
            "sport": ds.sport.value,
        },
    }


@router.post("/{dataset_id}/snapshot")
def create_snapshot(dataset_id: str) -> dict:
    from betlab.core.db.database import Database
    db = Database()
    ds = db.get_dataset(dataset_id)
    db.close()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    snapshot_id = f"snap-{uuid.uuid4().hex[:8]}"
    return {"snapshot_id": snapshot_id, "dataset_id": dataset_id, "created_at": datetime.utcnow().isoformat()}
