from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class StrategyCreateRequest(BaseModel):
    name: str
    sport: str
    market: str = "match_odds"
    filters: list[dict] | None = None
    signal: dict | None = None


class StrategyResponse(BaseModel):
    id: str
    name: str
    sport: str
    market: str
    created_at: str


@router.get("/")
def list_strategies() -> list[StrategyResponse]:
    from betlab.core.db.database import Database
    db = Database()
    strategies = db.list_strategies()
    db.close()
    return [
        StrategyResponse(id=s.id, name=s.name, sport=s.sport.value, market="match_odds", created_at=str(s.created_at))
        for s in strategies
    ]


@router.get("/{strategy_id}")
def get_strategy(strategy_id: str) -> StrategyResponse:
    from betlab.core.db.database import Database
    db = Database()
    strategy = db.get_strategy(strategy_id)
    db.close()
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return StrategyResponse(
        id=strategy.id, name=strategy.name, sport=strategy.sport.value,
        market="match_odds", created_at=str(strategy.created_at),
    )


@router.post("/")
def create_strategy(req: StrategyCreateRequest) -> StrategyResponse:
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import Strategy, Sport

    db = Database()
    strategy_id = f"strat-{uuid.uuid4().hex[:8]}"
    sport_enum = Sport(req.sport)
    strategy = Strategy(
        id=strategy_id,
        name=req.name,
        sport=sport_enum,
        description=f"Strategy: {req.name}",
    )
    db.insert_strategy(strategy)

    version_id = f"sv-{uuid.uuid4().hex[:8]}"
    from betlab.core.schemas.models import StrategyVersion
    sv = StrategyVersion(
        id=version_id,
        strategy_id=strategy_id,
        version="1.0.0",
        definition=req.model_dump(),
    )
    db.insert_strategy_version(sv)
    db.close()

    return StrategyResponse(
        id=strategy_id, name=req.name, sport=sport_enum.value,
        market=req.market, created_at=str(strategy.created_at),
    )


@router.post("/{strategy_id}/validate")
def validate_strategy(strategy_id: str) -> dict:
    from betlab.core.db.database import Database
    db = Database()
    strategy = db.get_strategy(strategy_id)
    db.close()
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"valid": True, "strategy_id": strategy_id, "checks": {"exists": True, "has_name": bool(strategy.name)}}


@router.post("/{strategy_id}/clone")
def clone_strategy(strategy_id: str) -> StrategyResponse:
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import Strategy, StrategyVersion

    db = Database()
    strategy = db.get_strategy(strategy_id)
    if strategy is None:
        db.close()
        raise HTTPException(status_code=404, detail="Strategy not found")

    new_id = f"strat-{uuid.uuid4().hex[:8]}"
    new_strategy = Strategy(
        id=new_id,
        name=f"{strategy.name} (clone)",
        sport=strategy.sport,
        description=strategy.description,
    )
    db.insert_strategy(new_strategy)

    cur = db._conn.execute("SELECT * FROM strategy_versions WHERE strategy_id = ? LIMIT 1", (strategy_id,))
    row = cur.fetchone()
    if row:
        sv = StrategyVersion(
            id=f"sv-{uuid.uuid4().hex[:8]}",
            strategy_id=new_id,
            version="1.0.0",
            definition=json.loads(row["definition"] or "{}"),
            hash=row["hash"],
        )
        db.insert_strategy_version(sv)

    db.close()
    return StrategyResponse(
        id=new_id, name=new_strategy.name, sport=new_strategy.sport.value,
        market="match_odds", created_at=str(new_strategy.created_at),
    )
