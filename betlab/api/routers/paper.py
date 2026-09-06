from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SessionCreateRequest(BaseModel):
    strategy_id: str
    bankroll: float = 1000.0


class SessionResponse(BaseModel):
    id: str
    strategy_id: str
    starting_bankroll: float
    current_bankroll: float
    status: str
    started_at: str


@router.get("/sessions")
def list_sessions() -> list[SessionResponse]:
    from betlab.core.db.database import Database
    db = Database()
    cur = db._conn.execute("SELECT * FROM paper_sessions ORDER BY started_at DESC LIMIT 20")
    rows = cur.fetchall()
    db.close()
    return [
        SessionResponse(
            id=r["id"], strategy_id=r["strategy_id"],
            starting_bankroll=r["starting_bankroll"], current_bankroll=r["current_bankroll"],
            status=r["status"], started_at=r["started_at"][:19] if r["started_at"] else "",
        )
        for r in rows
    ]


@router.post("/sessions")
def create_session(req: SessionCreateRequest) -> SessionResponse:
    import sqlite3
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSession, PaperSessionStatus

    db = Database()
    session_id = f"paper-{uuid.uuid4().hex[:8]}"
    session = PaperSession(
        id=session_id,
        strategy_id=req.strategy_id,
        strategy_version="1.0.0",
        starting_bankroll=req.bankroll,
        current_bankroll=req.bankroll,
        status=PaperSessionStatus.active,
    )
    try:
        db.insert_paper_session(session)
    except sqlite3.IntegrityError:
        db.close()
        raise HTTPException(status_code=404, detail="Strategy not found")
    db.close()

    return SessionResponse(
        id=session_id, strategy_id=req.strategy_id,
        starting_bankroll=req.bankroll, current_bankroll=req.bankroll,
        status="active", started_at=str(session.started_at),
    )


@router.post("/sessions/{session_id}/start")
def start_session(session_id: str) -> dict:
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSessionStatus

    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")
    db.update_paper_session_status(session_id, PaperSessionStatus.active)
    db.close()
    return {"session_id": session_id, "status": "active"}


@router.post("/sessions/{session_id}/pause")
def pause_session(session_id: str) -> dict:
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSessionStatus

    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")
    db.update_paper_session_status(session_id, PaperSessionStatus.paused)
    db.close()
    return {"session_id": session_id, "status": "paused"}


@router.post("/sessions/{session_id}/stop")
def stop_session(session_id: str) -> dict:
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSessionStatus

    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")
    db.update_paper_session_status(session_id, PaperSessionStatus.stopped)
    db.close()
    return {"session_id": session_id, "status": "stopped"}


@router.get("/sessions/{session_id}/positions")
def get_positions(session_id: str) -> list[dict]:
    from betlab.core.db.database import Database
    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")
    positions = db.get_positions_for_session(session_id)
    db.close()
    return [
        {
            "id": p.id, "session_id": p.session_id, "event_id": p.event_id,
            "market_id": p.market_id, "runner_id": p.runner_id, "side": p.side.value,
            "stake": p.stake, "price": p.price, "status": p.status.value,
            "profit": p.profit, "placed_at": str(p.placed_at),
        }
        for p in positions
    ]


@router.get("/sessions/{session_id}/metrics")
def get_metrics(session_id: str) -> dict:
    from betlab.core.db.database import Database
    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        db.close()
        raise HTTPException(status_code=404, detail="Session not found")
    positions = db.get_positions_for_session(session_id)
    db.close()

    total = len(positions)
    wins = sum(1 for p in positions if p.profit > 0)
    losses = sum(1 for p in positions if p.profit < 0)
    net_profit = sum(p.profit for p in positions)

    return {
        "session_id": session_id,
        "total_positions": total,
        "wins": wins,
        "losses": losses,
        "net_profit": round(net_profit, 2),
        "current_bankroll": session.current_bankroll,
        "roi": round(net_profit / session.starting_bankroll * 100, 2) if session.starting_bankroll else 0,
    }
