from __future__ import annotations

import typer

from betlab.cli.dataset_cmd import app as dataset_app
from betlab.cli.strategy_cmd import app as strategy_app
from betlab.cli.experiment_cmd import app as experiment_app
from betlab.cli.paper_cmd import app as paper_app
from betlab.cli.model_cmd import app as model_app

app = typer.Typer(
    name="betlab",
    help="Wippa Bet Lab — Sports betting research platform",
    no_args_is_help=True,
)

app.add_typer(dataset_app, name="dataset", help="Dataset management commands")
app.add_typer(strategy_app, name="strategy", help="Strategy management commands")
app.add_typer(experiment_app, name="experiment", help="Experiment management commands")
app.add_typer(paper_app, name="paper", help="Paper trading commands")
app.add_typer(model_app, name="model", help="Model Builder commands")


@app.command()
def report(
    experiment_id: str = typer.Argument(..., help="Experiment ID to generate report for"),
    fmt: str = typer.Option("json", "--format", "-f", help="Report format: json, markdown, html"),
) -> None:
    """Generate a report for an experiment."""
    from rich.console import Console
    from betlab.reporting.json_report import generate_json_report
    from betlab.reporting.markdown_report import generate_markdown_report
    from betlab.reporting.html_report import generate_html_report
    from betlab.backtester.engine import BacktestResult

    console = Console()
    console.print(f"[bold]Generating {fmt} report for experiment {experiment_id}...[/]")

    bt = BacktestResult(experiment_id=experiment_id)
    if fmt == "json":
        report_data = generate_json_report(experiment_id, bt)
        import json
        console.print_json(json.dumps(report_data))
    elif fmt == "markdown":
        md = generate_markdown_report(experiment_id, bt)
        console.print(md)
    elif fmt == "html":
        html = generate_html_report(experiment_id, bt)
        output_path = f"report_{experiment_id}.html"
        with open(output_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {output_path}[/]")
    else:
        console.print(f"[red]Unknown format: {fmt}. Use json, markdown, or html.[/]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
