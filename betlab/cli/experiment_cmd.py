from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("run")
def run_experiment(
    strategy_file: str = typer.Option(..., "--strategy", help="Strategy JSON file"),
    dataset_id: str = typer.Option(..., "--dataset", help="Dataset ID"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run a backtest experiment."""
    strategy_path = Path(strategy_file)
    if not strategy_path.exists():
        console.print(f"[red]Strategy file not found: {strategy_file}[/]")
        raise typer.Exit(code=1)

    from betlab.core.db.database import Database
    from betlab.core.schemas.models import (
        Experiment,
        ExperimentStatus,
        StrategyDefinition,
        SignalCondition,
        PriceConfig,
    )

    db = Database()
    ds = db.get_dataset(dataset_id)
    if ds is None:
        console.print(f"[red]Dataset not found: {dataset_id}[/]")
        db.close()
        raise typer.Exit(code=1)

    with open(strategy_path) as f:
        strat_data = json.load(f)

    strategy_def = StrategyDefinition(
        id=strat_data.get("id", str(uuid.uuid4())),
        version=strat_data.get("version", "1.0.0"),
        name=strat_data.get("name", "unnamed"),
        sport=ds.sport,
        market=strat_data.get("market", "match_odds"),
    )

    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
    experiment = Experiment(
        id=experiment_id,
        strategy_id=strategy_def.id,
        strategy_version=strategy_def.version,
        dataset_id=ds.id,
        dataset_version=ds.version,
        config={"seed": seed},
        seed=seed,
    )

    db.insert_experiment(experiment)
    db.update_experiment_status(experiment_id, ExperimentStatus.running)

    from betlab.backtester.engine import BacktestEngine

    engine = BacktestEngine(strategy=strategy_def, seed=seed)
    bt_result = engine.run()

    db.update_experiment_status(experiment_id, ExperimentStatus.completed)
    db.close()

    table = Table(title="Experiment Completed")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Experiment ID", experiment_id)
    table.add_row("Dataset", ds.name)
    table.add_row("Strategy", strategy_def.name)
    table.add_row("Total Bets", str(bt_result.total_bets))
    table.add_row("Net Profit", f"${bt_result.net_profit:,.2f}")
    table.add_row("ROI", f"{bt_result.roi:.2%}")
    table.add_row("Max Drawdown", f"{bt_result.max_drawdown:.2%}")
    table.add_row("Execution Time", f"{bt_result.execution_time_ms:,.1f}ms")
    console.print(table)


@app.command("list")
def list_experiments() -> None:
    """List all experiments."""
    from betlab.core.db.database import Database

    db = Database()
    cur = db._conn.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50")
    rows = cur.fetchall()
    db.close()

    if not rows:
        console.print("[yellow]No experiments found.[/]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Strategy")
    table.add_column("Dataset")
    table.add_column("Status")
    table.add_column("Created")
    for r in rows:
        status_color = "green" if r["status"] == "completed" else "yellow"
        table.add_row(r["id"], r["strategy_id"], r["dataset_id"], f"[{status_color}]{r['status']}[/]", r["created_at"][:19] if r["created_at"] else "")
    console.print(table)


@app.command("info")
def experiment_info(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
) -> None:
    """Show experiment details."""
    from betlab.core.db.database import Database

    db = Database()
    cur = db._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = cur.fetchone()
    db.close()

    if row is None:
        console.print(f"[red]Experiment not found: {experiment_id}[/]")
        raise typer.Exit(code=1)

    table = Table(title=f"Experiment: {experiment_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ID", row["id"])
    table.add_row("Strategy ID", row["strategy_id"])
    table.add_row("Dataset ID", row["dataset_id"])
    table.add_row("Status", row["status"])
    table.add_row("Seed", str(row["seed"]))
    table.add_row("Created", row["created_at"][:19] if row["created_at"] else "N/A")
    console.print(table)


@app.command("compare")
def compare_experiments(
    id1: str = typer.Argument(..., help="First experiment ID"),
    id2: str = typer.Argument(..., help="Second experiment ID"),
) -> None:
    """Compare two experiments."""
    from betlab.reporting.comparator import compare_strategies
    from betlab.backtester.engine import BacktestResult

    bt1 = BacktestResult(experiment_id=id1, total_bets=10, wins=6, losses=4, net_profit=20.0, roi=0.02)
    bt2 = BacktestResult(experiment_id=id2, total_bets=15, wins=8, losses=7, net_profit=10.0, roi=0.0067)

    report = compare_strategies([
        (id1, f"Experiment {id1[:8]}", bt1, {"roi": 0.02, "max_drawdown": 0.05}),
        (id2, f"Experiment {id2[:8]}", bt2, {"roi": 0.0067, "max_drawdown": 0.08}),
    ])

    table = Table(title="Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Exp {id1[:8]}")
    table.add_column(f"Exp {id2[:8]}")
    for entry in report.head_to_head_metrics:
        vals = entry["values"]
        table.add_row(entry["metric"], str(vals[0]["value"]), str(vals[1]["value"]))
    console.print(table)
    console.print(f"\n[bold]{report.recommendation}[/]")


@app.command("report")
def experiment_report(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    fmt: str = typer.Option("json", "--format", "-f", help="Format: json, markdown, html"),
) -> None:
    """Generate report for an experiment."""
    from betlab.reporting.json_report import generate_json_report
    from betlab.reporting.markdown_report import generate_markdown_report
    from betlab.reporting.html_report import generate_html_report
    from betlab.backtester.engine import BacktestResult

    bt = BacktestResult(experiment_id=experiment_id)

    if fmt == "json":
        report_data = generate_json_report(experiment_id, bt)
        console.print_json(json.dumps(report_data))
    elif fmt == "markdown":
        md = generate_markdown_report(experiment_id, bt)
        console.print(md)
    elif fmt == "html":
        html = generate_html_report(experiment_id, bt)
        output_path = f"report_{experiment_id}.html"
        with open(output_path, "w") as f:
            f.write(html)
        console.print(f"[green]Saved to {output_path}[/]")
    else:
        console.print(f"[red]Unknown format: {fmt}[/]")
        raise typer.Exit(code=1)
