from __future__ import annotations

from datetime import datetime
from typing import Any

from betlab.backtester.engine import BacktestResult


def generate_markdown_report(
    experiment_id: str,
    backtest_result: BacktestResult,
    metrics: dict[str, Any] | None = None,
) -> str:
    metrics = metrics or {}
    lines: list[str] = []

    def _h(level: int, title: str) -> None:
        lines.append(f"{'#' * level} {title}\n")

    def _table(headers: list[str], rows: list[list[str]]) -> None:
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(header_line)
        lines.append(sep_line)
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append(f"# Wippa Bet Lab — Backtest Report\n")
    lines.append(f"**Experiment:** `{experiment_id}`  ")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()}Z\n")

    # ── Executive Summary ──
    _h(2, "Executive Summary")
    summary_rows = [
        ("Total Bets", str(backtest_result.total_bets)),
        ("Wins / Losses / Voids", f"{backtest_result.wins} / {backtest_result.losses} / {backtest_result.voids}"),
        ("Net Profit", f"${backtest_result.net_profit:,.2f}"),
        ("ROI", f"{backtest_result.roi:.2%}"),
        ("Yield", f"{backtest_result.yield_pct:.2f}%"),
        ("Max Drawdown", f"{backtest_result.max_drawdown:.2%}"),
        ("Execution Time", f"{backtest_result.execution_time_ms:,.1f}ms"),
    ]
    _table(["Metric", "Value"], [[m, v] for m, v in summary_rows])

    # ── Strategy ──
    _h(2, "Strategy")
    strategy_name = getattr(backtest_result, "strategy_name", "N/A")
    sport = getattr(backtest_result, "sport", "N/A")
    market = getattr(backtest_result, "market", "N/A")
    lines.append(f"- **Name:** {strategy_name}")
    lines.append(f"- **Sport:** {sport}")
    lines.append(f"- **Market:** {market}")
    lines.append("")

    filters = getattr(backtest_result, "filters", [])
    if filters:
        _h(3, "Filters")
        for f in filters:
            if isinstance(f, dict):
                lines.append(f"- `{f.get('field', '')}` {f.get('operator', '')} `{f.get('value', '')}`")
            else:
                lines.append(f"- {f}")
        lines.append("")

    signal = getattr(backtest_result, "signal", {})
    if signal:
        _h(3, "Signal")
        if isinstance(signal, dict):
            lines.append(f"- Field: `{signal.get('field', '')}`")
            lines.append(f"- Operator: `{signal.get('operator', '')}`")
            lines.append(f"- Threshold: `{signal.get('value', '')}`")
        lines.append("")

    # ── Configuration ──
    _h(2, "Configuration")
    starting_bankroll = getattr(backtest_result, "starting_bankroll", 1000.0)
    seed = getattr(backtest_result, "seed", None)
    _table(
        ["Parameter", "Value"],
        [
            ["Starting Bankroll", f"${starting_bankroll:,.2f}"],
            ["Seed", str(seed) if seed is not None else "None"],
        ],
    )

    # ── Performance ──
    _h(2, "Performance")
    perf = metrics.get("performance", {})
    if perf:
        perf_rows = []
        for k, v in perf.items():
            if isinstance(v, float):
                perf_rows.append([k, f"{v:.4f}"])
            else:
                perf_rows.append([k, str(v)])
        _table(["Metric", "Value"], perf_rows)
    else:
        _table(
            ["Metric", "Value"],
            [
                ["Total Bets", str(backtest_result.total_bets)],
                ["Wins", str(backtest_result.wins)],
                ["Losses", str(backtest_result.losses)],
                ["Gross Profit", f"${backtest_result.gross_profit:,.2f}"],
                ["Net Profit", f"${backtest_result.net_profit:,.2f}"],
                ["ROI", f"{backtest_result.roi:.2%}"],
                ["Yield", f"{backtest_result.yield_pct:.2f}%"],
            ],
        )

    # ── Risk ──
    _h(2, "Risk Analysis")
    _table(
        ["Risk Metric", "Value"],
        [
            ["Max Drawdown", f"{backtest_result.max_drawdown:.2%}"],
            ["Average Drawdown", f"{backtest_result.average_drawdown:.2%}"],
        ],
    )

    drawdown_data = metrics.get("drawdown", {})
    if drawdown_data:
        lines.append("\n**Drawdown Details:**\n")
        for k, v in drawdown_data.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            elif isinstance(v, list):
                lines.append(f"- {k}: {len(v)} entries")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    # ── CLV ──
    _h(2, "Closing Line Value (CLV)")
    clv_summary = backtest_result.clv_summary
    if clv_summary:
        clv_rows = []
        for k, v in clv_summary.items():
            if isinstance(v, float):
                clv_rows.append([k, f"{v:.4f}"])
            else:
                clv_rows.append([k, str(v)])
        _table(["CLV Metric", "Value"], clv_rows)
    else:
        lines.append("No CLV data available.\n")

    # ── Calibration ──
    _h(2, "Calibration")
    calibration = backtest_result.calibration
    if calibration:
        cal_rows = []
        for k, v in calibration.items():
            if isinstance(v, float):
                cal_rows.append([k, f"{v:.4f}"])
            else:
                cal_rows.append([k, str(v)])
        _table(["Calibration Metric", "Value"], cal_rows)
    else:
        lines.append("No calibration data available.\n")

    # ── Warnings ──
    if backtest_result.warnings:
        _h(2, "Warnings")
        for w in backtest_result.warnings:
            lines.append(f"- **Warning:** {w}")
        lines.append("")

    return "\n".join(lines)
