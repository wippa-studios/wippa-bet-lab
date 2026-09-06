from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("import")
def import_dataset(
    path: str = typer.Argument(..., help="Path to dataset file (JSON)"),
    sport: str = typer.Option(..., "--sport", "-s", help="Sport type: tennis, greyhounds, nba"),
    name: str = typer.Option(..., "--name", "-n", help="Dataset name"),
) -> None:
    """Import a dataset from a JSON file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        console.print(f"[red]File not found: {path}[/]")
        raise typer.Exit(code=1)

    try:
        with open(dataset_path) as f:
            data = json.load(f)
        events = data if isinstance(data, list) else data.get("events", [])
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/]")
        raise typer.Exit(code=1)

    dataset_id = f"ds-{uuid.uuid4().hex[:8]}"
    from betlab.core.schemas.models import Dataset, Sport
    from betlab.core.db.database import Database

    db = Database()
    sport_enum = Sport(sport)
    ds = Dataset(
        id=dataset_id,
        name=name,
        version="1.0.0",
        sport=sport_enum,
        source=str(dataset_path),
        event_count=len(events),
    )
    ds.compute_data_hash()
    db.insert_dataset(ds)
    db.close()

    table = Table(title="Dataset Imported")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ID", dataset_id)
    table.add_row("Name", name)
    table.add_row("Sport", sport)
    table.add_row("Events", str(len(events)))
    console.print(table)


@app.command("list")
def list_datasets() -> None:
    """List all datasets."""
    from betlab.core.db.database import Database

    db = Database()
    datasets = db.list_datasets()
    db.close()

    if not datasets:
        console.print("[yellow]No datasets found.[/]")
        return

    table = Table(title="Datasets")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Sport")
    table.add_column("Events", justify="right")
    table.add_column("Created")
    for ds in datasets:
        table.add_row(ds.id, ds.name, ds.sport.value, str(ds.event_count), str(ds.created_at)[:19])
    console.print(table)


@app.command("validate")
def validate_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID to validate"),
) -> None:
    """Validate a dataset."""
    from betlab.core.db.database import Database

    db = Database()
    ds = db.get_dataset(dataset_id)
    db.close()

    if ds is None:
        console.print(f"[red]Dataset not found: {dataset_id}[/]")
        raise typer.Exit(code=1)

    table = Table(title=f"Validation: {ds.name}")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Exists", "[green]PASS[/]")
    table.add_row("Event Count", f"[green]{ds.event_count}[/]" if ds.event_count > 0 else "[red]0[/]")
    table.add_row("Sport", ds.sport.value)
    console.print(table)


@app.command("info")
def dataset_info(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
) -> None:
    """Show dataset details."""
    from betlab.core.db.database import Database

    db = Database()
    ds = db.get_dataset(dataset_id)
    db.close()

    if ds is None:
        console.print(f"[red]Dataset not found: {dataset_id}[/]")
        raise typer.Exit(code=1)

    table = Table(title=f"Dataset: {ds.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ID", ds.id)
    table.add_row("Name", ds.name)
    table.add_row("Version", ds.version)
    table.add_row("Sport", ds.sport.value)
    table.add_row("Source", ds.source or "N/A")
    table.add_row("Events", str(ds.event_count))
    table.add_row("Markets", str(ds.market_count))
    table.add_row("Time Range", ds.time_range or "N/A")
    table.add_row("Data Hash", ds.data_hash or "N/A")
    table.add_row("Created", str(ds.created_at)[:19])
    console.print(table)
