from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("start")
def start_session(
    strategy_file: str = typer.Option(..., "--strategy", help="Strategy JSON file"),
    bankroll: float = typer.Option(1000.0, "--bankroll", help="Starting bankroll"),
) -> None:
    """Start a paper trading session."""
    strategy_path = Path(strategy_file)
    if not strategy_path.exists():
        console.print(f"[red]Strategy file not found: {strategy_file}[/]")
        raise typer.Exit(code=1)

    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSession, PaperSessionStatus

    db = Database()
    session_id = f"paper-{uuid.uuid4().hex[:8]}"

    with open(strategy_path) as f:
        strat_data = json.load(f)

    session = PaperSession(
        id=session_id,
        strategy_id=strat_data.get("id", "unknown"),
        strategy_version=strat_data.get("version", "1.0.0"),
        starting_bankroll=bankroll,
        current_bankroll=bankroll,
        status=PaperSessionStatus.active,
    )
    db.insert_paper_session(session)
    db.close()

    table = Table(title="Paper Trading Session Started")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Session ID", session_id)
    table.add_row("Strategy", strat_data.get("name", "unknown"))
    table.add_row("Bankroll", f"${bankroll:,.2f}")
    table.add_row("Status", "active")
    console.print(table)


@app.command("status")
def session_status() -> None:
    """Show active paper trading sessions."""
    from betlab.core.db.database import Database

    db = Database()
    cur = db._conn.execute("SELECT * FROM paper_sessions ORDER BY started_at DESC LIMIT 10")
    rows = cur.fetchall()
    db.close()

    if not rows:
        console.print("[yellow]No paper trading sessions found.[/]")
        return

    table = Table(title="Paper Trading Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Strategy")
    table.add_column("Bankroll", justify="right")
    table.add_column("Status")
    table.add_column("Started")
    for r in rows:
        status_color = "green" if r["status"] == "active" else "yellow" if r["status"] == "paused" else "red"
        table.add_row(
            r["id"], r["strategy_id"],
            f"${r['current_bankroll']:,.2f}",
            f"[{status_color}]{r['status']}[/]",
            r["started_at"][:19] if r["started_at"] else "N/A",
        )
    console.print(table)


@app.command("pause")
def pause_session(
    session_id: str = typer.Argument(..., help="Session ID to pause"),
) -> None:
    """Pause a paper trading session."""
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSessionStatus

    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        console.print(f"[red]Session not found: {session_id}[/]")
        db.close()
        raise typer.Exit(code=1)

    db.update_paper_session_status(session_id, PaperSessionStatus.paused)
    db.close()
    console.print(f"[green]Session {session_id} paused.[/]")


@app.command("resume")
def resume_session(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
) -> None:
    """Resume a paused paper trading session."""
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSessionStatus

    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        console.print(f"[red]Session not found: {session_id}[/]")
        db.close()
        raise typer.Exit(code=1)

    db.update_paper_session_status(session_id, PaperSessionStatus.active)
    db.close()
    console.print(f"[green]Session {session_id} resumed.[/]")


@app.command("stop")
def stop_session(
    session_id: str = typer.Argument(..., help="Session ID to stop"),
) -> None:
    """Stop a paper trading session."""
    from betlab.core.db.database import Database
    from betlab.core.schemas.models import PaperSessionStatus

    db = Database()
    session = db.get_paper_session(session_id)
    if session is None:
        console.print(f"[red]Session not found: {session_id}[/]")
        db.close()
        raise typer.Exit(code=1)

    db.update_paper_session_status(session_id, PaperSessionStatus.stopped)
    db.close()
    console.print(f"[green]Session {session_id} stopped.[/]")
