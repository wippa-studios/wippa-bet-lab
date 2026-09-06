from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("create")
def create_strategy(
    name: str = typer.Option(..., "--name", "-n", help="Strategy name"),
    sport: str = typer.Option(..., "--sport", "-s", help="Sport type"),
    market: str = typer.Option("match_odds", "--market", "-m", help="Market type"),
    output: str = typer.Option("strategy.json", "--output", "-o", help="Output file path"),
) -> None:
    """Create a new strategy interactively."""
    console.print("[bold]Creating strategy...[/]")

    from betlab.strategies.builder import StrategyBuilder

    builder = StrategyBuilder()
    builder.set_name(name).set_sport(sport).set_market(market)

    builder.set_selection(side="back")
    builder.set_price(source="back_price", minimum=1.5, maximum=10.0)

    console.print("[yellow]Adding default filter: back_price > 1.5[/]")
    builder.add_filter("back_price", ">", 1.5)

    strategy = builder.build()

    with open(output, "w") as f:
        f.write(builder.to_json())

    table = Table(title="Strategy Created")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ID", strategy.id)
    table.add_row("Name", name)
    table.add_row("Sport", sport)
    table.add_row("Market", market)
    table.add_row("Output", output)
    console.print(table)


@app.command("validate")
def validate_strategy(
    strategy_file: str = typer.Argument(..., help="Strategy JSON file path"),
) -> None:
    """Validate a strategy definition."""
    path = Path(strategy_file)
    if not path.exists():
        console.print(f"[red]File not found: {strategy_file}[/]")
        raise typer.Exit(code=1)

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/]")
        raise typer.Exit(code=1)

    required_fields = ["id", "name", "sport"]
    missing = [f for f in required_fields if f not in data]

    table = Table(title=f"Validation: {path.name}")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_row("JSON Valid", "[green]PASS[/]")
    table.add_row("Has ID", "[green]PASS[/]" if "id" in data else "[red]MISSING[/]")
    table.add_row("Has Name", "[green]PASS[/]" if "name" in data else "[red]MISSING[/]")
    table.add_row("Has Sport", "[green]PASS[/]" if "sport" in data else "[red]MISSING[/]")
    table.add_row("Has Filters", "[green]YES[/]" if data.get("filters") else "[yellow]NO[/]")
    table.add_row("Has Signal", "[green]YES[/]" if data.get("signal") else "[yellow]NO[/]")
    table.add_row("Has Selection", "[green]YES[/]" if data.get("selection") else "[yellow]NO[/]")
    table.add_row("Has Price", "[green]YES[/]" if data.get("price") else "[yellow]NO[/]")
    console.print(table)

    if missing:
        console.print(f"[red]Missing required fields: {', '.join(missing)}[/]")
        raise typer.Exit(code=1)

    console.print("[green]Strategy is valid.[/]")


@app.command("list")
def list_strategies() -> None:
    """List all strategies."""
    from betlab.core.db.database import Database

    db = Database()
    strategies = db.list_strategies()
    db.close()

    if not strategies:
        console.print("[yellow]No strategies found.[/]")
        return

    table = Table(title="Strategies")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Sport")
    table.add_column("Versions")
    table.add_column("Created")
    for s in strategies:
        table.add_row(s.id, s.name, s.sport.value, str(len(s.versions)), str(s.created_at)[:19])
    console.print(table)


@app.command("info")
def strategy_info(
    strategy_id: str = typer.Argument(..., help="Strategy ID"),
) -> None:
    """Show strategy details."""
    from betlab.core.db.database import Database

    db = Database()
    strategy = db.get_strategy(strategy_id)
    db.close()

    if strategy is None:
        console.print(f"[red]Strategy not found: {strategy_id}[/]")
        raise typer.Exit(code=1)

    table = Table(title=f"Strategy: {strategy.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ID", strategy.id)
    table.add_row("Name", strategy.name)
    table.add_row("Sport", strategy.sport.value)
    table.add_row("Description", strategy.description or "N/A")
    table.add_row("Versions", str(len(strategy.versions)))
    table.add_row("Created", str(strategy.created_at)[:19])
    console.print(table)
