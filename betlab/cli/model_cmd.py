from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("columns")
def show_columns(
    dataset_id: str = typer.Option(..., "--dataset", "-d", help="Dataset ID"),
) -> None:
    """Show column analysis for a dataset."""
    dataset_path = Path("datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        console.print(f"[red]Dataset file not found: {dataset_path}[/]")
        raise typer.Exit(code=1)

    with open(dataset_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("events", [])
    if not events:
        console.print("[yellow]Dataset contains no events.[/]")
        return

    import pandas as pd
    from betlab.ml.column_analyzer import ColumnAnalyzer

    df = pd.DataFrame(events)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)

    table = Table(title=f"Columns — {dataset_id} ({preview.row_count} rows)")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Missing %", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Mean", justify="right")

    for col in preview.columns:
        table.add_row(
            col.name,
            col.dtype,
            f"{col.missing_pct:.1f}",
            str(col.unique_count),
            str(col.min) if col.min is not None else "",
            str(col.max) if col.max is not None else "",
            f"{col.mean:.2f}" if col.mean is not None else "",
        )

    console.print(table)

    if preview.warnings:
        for w in preview.warnings:
            console.print(f"[yellow]Warning: {w}[/]")


@app.command("preview")
def show_preview(
    dataset_id: str = typer.Option(..., "--dataset", "-d", help="Dataset ID"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of rows"),
) -> None:
    """Show a preview of dataset rows."""
    dataset_path = Path("datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        console.print(f"[red]Dataset file not found: {dataset_path}[/]")
        raise typer.Exit(code=1)

    with open(dataset_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("events", [])
    if not events:
        console.print("[yellow]Dataset contains no events.[/]")
        return

    import pandas as pd
    from betlab.ml.column_analyzer import ColumnAnalyzer

    df = pd.DataFrame(events)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)

    console.print(f"[bold]Preview — {dataset_id}[/] ({preview.row_count} total rows)")
    for i, row in enumerate(preview.preview_rows[:limit]):
        console.print(f"\n[bold cyan]Row {i}:[/]")
        for k, v in row.items():
            console.print(f"  {k}: {v}")

    if preview.warnings:
        for w in preview.warnings:
            console.print(f"\n[yellow]Warning: {w}[/]")


@app.command("train")
def train_model(
    dataset_id: str = typer.Option(..., "--dataset", "-d", help="Dataset ID"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    features: str = typer.Option(..., "--features", "-f", help="Comma-separated feature columns"),
    model_type: str = typer.Option("random_forest", "--model", "-m", help="Model type"),
) -> None:
    """Train a model on a dataset."""
    import pandas as pd
    from betlab.ml.schemas import (
        FeatureConfig,
        ModelConfig,
        ModelManifest,
        ModelType,
        PreprocessingConfig,
        TargetConfig,
        TargetType,
        ValidationConfig,
    )
    from betlab.ml.trainer import ModelTrainer
    from betlab.ml.registry import ModelRegistry

    dataset_path = Path("datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        console.print(f"[red]Dataset file not found: {dataset_path}[/]")
        raise typer.Exit(code=1)

    with open(dataset_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("events", [])
    if not events:
        console.print("[yellow]Dataset contains no events.[/]")
        raise typer.Exit(code=1)

    feature_list = [f.strip() for f in features.split(",") if f.strip()]

    target_config = TargetConfig(column=target, type=TargetType.binary_classification)
    model_config = ModelConfig(type=ModelType(model_type))
    preprocessing_config = PreprocessingConfig()
    validation_config = ValidationConfig()

    trainer = ModelTrainer(
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        validation_config=validation_config,
    )

    df = pd.DataFrame(events)
    result = trainer.train(df, target_config, feature_list)

    registry = ModelRegistry()
    metrics_hash = registry.compute_metrics_hash(result.metrics)

    manifest = ModelManifest(
        id=result.model_id,
        name=f"model-{result.model_id[:8]}",
        version="1.0.0",
        target_config=target_config,
        feature_columns=feature_list,
        model_type=model_config.type,
        model_parameters=model_config.parameters,
        preprocessing_config=preprocessing_config,
        validation_config=validation_config,
        dataset_id=dataset_id,
        metrics_hash=metrics_hash,
    )

    artifact_path = f".betlab/artifacts/{result.model_id}.joblib"
    registry.save_model(manifest, artifact_path)
    registry.save_model_metrics(result.model_id, result.metrics)
    registry.update_status(result.model_id, "trained")
    registry.close()

    table = Table(title="Training Complete")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model ID", result.model_id)
    table.add_row("Status", "trained")
    table.add_row("Execution Time", f"{result.execution_time_ms:.0f}ms")

    for k, v in result.metrics.items():
        if isinstance(v, float):
            table.add_row(f"  {k}", f"{v:.4f}")
        else:
            table.add_row(f"  {k}", str(v))

    console.print(table)

    if result.warnings:
        for w in result.warnings:
            console.print(f"[yellow]Warning: {w}[/]")


@app.command("results")
def show_results(
    model_id: str = typer.Argument(..., help="Model ID"),
) -> None:
    """Show training results for a model."""
    from betlab.ml.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        console.print(f"[red]Model not found: {model_id}[/]")
        registry.close()
        raise typer.Exit(code=1)

    metrics = registry.get_model_metrics(model_id)
    registry.close()

    table = Table(title=f"Results — {model_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))

    console.print(table)


@app.command("importance")
def show_importance(
    model_id: str = typer.Argument(..., help="Model ID"),
) -> None:
    """Show feature importance for a model."""
    from betlab.ml.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        console.print(f"[red]Model not found: {model_id}[/]")
        registry.close()
        raise typer.Exit(code=1)

    metrics = registry.get_model_metrics(model_id)
    importance = metrics.get("feature_importance", [])
    registry.close()

    if not importance:
        console.print("[yellow]No feature importance data available.[/]")
        return

    table = Table(title=f"Feature Importance — {model_id}")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", style="green", justify="right")

    for item in importance:
        table.add_row(item.get("feature", "?"), f"{item.get('importance', 0):.4f}")

    console.print(table)


@app.command("export-strategy")
def export_strategy(
    model_id: str = typer.Argument(..., help="Model ID"),
    output: str = typer.Option("strategy.json", "--output", "-o", help="Output file"),
) -> None:
    """Export a trained model as a strategy JSON."""
    from betlab.ml.registry import ModelRegistry
    from betlab.ml.strategy_converter import StrategyConverter

    registry = ModelRegistry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        console.print(f"[red]Model not found: {model_id}[/]")
        registry.close()
        raise typer.Exit(code=1)

    converter = StrategyConverter()
    strategy = converter.convert_to_strategy(
        model=None,
        feature_columns=manifest.feature_columns,
        betting_config=None,
        model_metrics=registry.get_model_metrics(model_id),
    )

    registry.save_model_strategy(model_id, strategy)
    registry.close()

    out_path = Path(output)
    out_path.write_text(json.dumps(strategy, indent=2))
    console.print(f"[green]Strategy exported to {out_path}[/]")


@app.command("list")
def list_models(
    limit: int = typer.Option(20, "--limit", "-n", help="Max models to show"),
) -> None:
    """List all registered models."""
    from betlab.ml.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_models(limit=limit)
    registry.close()

    if not models:
        console.print("[yellow]No models found.[/]")
        return

    table = Table(title="Models")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Dataset")
    table.add_column("Created")

    for m in models:
        table.add_row(
            m.id,
            m.name,
            m.model_type.value if hasattr(m.model_type, "value") else str(m.model_type),
            "draft",
            m.dataset_id or "-",
            str(m.created_at)[:19],
        )

    console.print(table)


@app.command("info")
def model_info(
    model_id: str = typer.Argument(..., help="Model ID"),
) -> None:
    """Show detailed model information."""
    from betlab.ml.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get_model(model_id)
    if manifest is None:
        console.print(f"[red]Model not found: {model_id}[/]")
        registry.close()
        raise typer.Exit(code=1)

    metrics = registry.get_model_metrics(model_id)
    registry.close()

    table = Table(title=f"Model — {model_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("ID", manifest.id)
    table.add_row("Name", manifest.name)
    table.add_row("Version", manifest.version)
    table.add_row("Type", manifest.model_type.value)
    table.add_row("Dataset", manifest.dataset_id or "-")
    table.add_row("Features", ", ".join(manifest.feature_columns))
    table.add_row("Target", manifest.target_config.column)
    table.add_row("Metrics Hash", manifest.metrics_hash)
    table.add_row("Created", str(manifest.created_at)[:19])

    if metrics:
        console.print("\n[bold]Metrics:[/]")
        for k, v in metrics.items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            elif not isinstance(v, (dict, list)):
                console.print(f"  {k}: {v}")

    console.print(table)
