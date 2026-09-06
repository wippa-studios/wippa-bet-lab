from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from betlab.backtester.engine import BacktestResult


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_html_report(
    experiment_id: str,
    backtest_result: BacktestResult,
    metrics: dict[str, Any] | None = None,
) -> str:
    metrics = metrics or {}
    perf = metrics.get("performance", {})
    clv_summary = backtest_result.clv_summary
    calibration = backtest_result.calibration

    bankroll_history = backtest_result.bankroll_history or []
    pnl_series = [s["balance"] for s in bankroll_history]
    timestamps = [s.get("timestamp", "") for s in bankroll_history]

    max_dd = backtest_result.max_drawdown
    avg_dd = backtest_result.average_drawdown

    cal_brier = calibration.get("brier_score", 0.0) if isinstance(calibration, dict) else getattr(calibration, "brier_score", 0.0)
    cal_log_loss = calibration.get("log_loss", 0.0) if isinstance(calibration, dict) else getattr(calibration, "log_loss", 0.0)

    pnl_svg = _build_pnl_svg(pnl_series)
    dd_svg = _build_drawdown_svg(pnl_series)
    cal_svg = _build_calibration_svg()

    warning_html = ""
    if backtest_result.warnings:
        warning_html = "<ul>" + "".join(f"<li>{w}</li>" for w in backtest_result.warnings) + "</ul>"

    perf_rows = ""
    if perf:
        for k, v in perf.items():
            val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            perf_rows += f"<tr><td>{k}</td><td>{val}</td></tr>"
    else:
        perf_rows = (
            f"<tr><td>Total Bets</td><td>{backtest_result.total_bets}</td></tr>"
            f"<tr><td>Wins</td><td>{backtest_result.wins}</td></tr>"
            f"<tr><td>Losses</td><td>{backtest_result.losses}</td></tr>"
            f"<tr><td>Net Profit</td><td>${backtest_result.net_profit:,.2f}</td></tr>"
            f"<tr><td>ROI</td><td>{backtest_result.roi:.2%}</td></tr>"
        )

    clv_rows = ""
    if clv_summary:
        for k, v in clv_summary.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            clv_rows += f"<tr><td>{k}</td><td>{val}</td></tr>"

    cal_rows = ""
    if calibration:
        for k, v in calibration.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            cal_rows += f"<tr><td>{k}</td><td>{val}</td></tr>"

    strategy_name = getattr(backtest_result, "strategy_name", "N/A")
    sport = getattr(backtest_result, "sport", "N/A")
    market = getattr(backtest_result, "market", "N/A")
    starting_bankroll = getattr(backtest_result, "starting_bankroll", 1000.0)

    try:
        env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
        template = env.get_template("report.html")
        return template.render(
            experiment_id=experiment_id,
            generated_at=datetime.utcnow().isoformat(),
            strategy_name=strategy_name,
            sport=sport,
            market=market,
            starting_bankroll=starting_bankroll,
            total_bets=backtest_result.total_bets,
            wins=backtest_result.wins,
            losses=backtest_result.losses,
            voids=backtest_result.voids,
            net_profit=backtest_result.net_profit,
            roi=backtest_result.roi,
            yield_pct=backtest_result.yield_pct,
            max_drawdown=backtest_result.max_drawdown,
            avg_drawdown=backtest_result.average_drawdown,
            gross_profit=backtest_result.gross_profit,
            execution_time_ms=backtest_result.execution_time_ms,
            perf_rows=perf_rows,
            clv_rows=clv_rows,
            cal_rows=cal_rows,
            cal_brier=cal_brier,
            cal_log_loss=cal_log_loss,
            pnl_svg=pnl_svg,
            dd_svg=dd_svg,
            cal_svg=cal_svg,
            warning_html=warning_html,
            bankroll_count=len(bankroll_history),
        )
    except Exception:
        return _build_inline_html(
            experiment_id=experiment_id,
            strategy_name=strategy_name,
            sport=sport,
            market=market,
            starting_bankroll=starting_bankroll,
            perf_rows=perf_rows,
            clv_rows=clv_rows,
            cal_rows=cal_rows,
            pnl_svg=pnl_svg,
            dd_svg=dd_svg,
            cal_svg=cal_svg,
            warning_html=warning_html,
            backtest_result=backtest_result,
        )


def _build_pnl_svg(series: list[float]) -> str:
    if not series:
        return "<p>No P&L data</p>"
    w, h = 600, 200
    mn, mx = min(series), max(series)
    rng = mx - mn if mx != mn else 1
    points = []
    for i, v in enumerate(series):
        x = (i / max(len(series) - 1, 1)) * w
        y = h - ((v - mn) / rng) * (h - 20) - 10
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="4"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#00d4aa" stroke-width="2"/>'
        f'<text x="10" y="15" fill="#aaa" font-size="10">Cumulative P&L</text>'
        f"</svg>"
    )


def _build_drawdown_svg(series: list[float]) -> str:
    if not series:
        return "<p>No drawdown data</p>"
    w, h = 600, 200
    peak = series[0]
    dd_vals = []
    for v in series:
        if v > peak:
            peak = v
        dd_vals.append((v - peak) / peak if peak else 0)
    mn, mx = min(dd_vals), 0
    rng = mx - mn if mx != mn else 1
    points = []
    for i, v in enumerate(dd_vals):
        x = (i / max(len(dd_vals) - 1, 1)) * w
        y = ((v - mn) / rng) * (h - 20) + 10
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="4"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#ff6b6b" stroke-width="2"/>'
        f'<text x="10" y="15" fill="#aaa" font-size="10">Drawdown Curve</text>'
        f"</svg>"
    )


def _build_calibration_svg() -> str:
    w, h = 300, 300
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="4"/>'
        f'<line x1="10" y1="{h - 10}" x2="{w - 10}" y2="{h - 10}" stroke="#444" stroke-width="1"/>'
        f'<line x1="10" y1="10" x2="10" y2="{h - 10}" stroke="#444" stroke-width="1"/>'
        f'<line x1="10" y1="{h - 10}" x2="{w - 10}" y2="10" stroke="#666" stroke-width="1" stroke-dasharray="4"/>'
        f'<text x="10" y="15" fill="#aaa" font-size="10">Calibration Plot</text>'
        f'<text x="{w // 2}" y="{h - 2}" fill="#aaa" font-size="9" text-anchor="middle">Predicted</text>'
        f"</svg>"
    )


def _build_inline_html(
    experiment_id: str,
    strategy_name: str,
    sport: str,
    market: str,
    starting_bankroll: float,
    perf_rows: str,
    clv_rows: str,
    cal_rows: str,
    pnl_svg: str,
    dd_svg: str,
    cal_svg: str,
    warning_html: str,
    backtest_result: BacktestResult,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BetLab Report — {experiment_id}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0d1117; color: #e6edf3; line-height: 1.6; }}
.container {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}
h1 {{ color: #00d4aa; margin-bottom: 8px; font-size: 1.6em; }}
h2 {{ color: #58a6ff; margin: 24px 0 12px; font-size: 1.2em; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
.meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 24px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #8b949e; font-weight: 600; }}
td {{ color: #e6edf3; }}
.chart {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; overflow-x: auto; }}
.warning {{ background: #2d1b00; border: 1px solid #9e6a03; border-radius: 8px; padding: 12px; margin-bottom: 16px; color: #f0c000; }}
.warning h2 {{ color: #f0c000; border-bottom-color: #9e6a03; }}
.kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.kpi-item {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }}
.kpi-value {{ font-size: 1.4em; font-weight: 700; color: #00d4aa; }}
.kpi-label {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
<h1>Wippa Bet Lab — Backtest Report</h1>
<p class="meta">Experiment: <code>{experiment_id}</code> | Generated: {datetime.utcnow().isoformat()}Z</p>

<h2>Executive Summary</h2>
<div class="kpi">
  <div class="kpi-item"><div class="kpi-value">{backtest_result.total_bets}</div><div class="kpi-label">Total Bets</div></div>
  <div class="kpi-item"><div class="kpi-value">{backtest_result.wins}W / {backtest_result.losses}L</div><div class="kpi-label">Wins / Losses</div></div>
  <div class="kpi-item"><div class="kpi-value">${backtest_result.net_profit:,.2f}</div><div class="kpi-label">Net Profit</div></div>
  <div class="kpi-item"><div class="kpi-value">{backtest_result.roi:.2%}</div><div class="kpi-label">ROI</div></div>
  <div class="kpi-item"><div class="kpi-value">{backtest_result.yield_pct:.2f}%</div><div class="kpi-label">Yield</div></div>
  <div class="kpi-item"><div class="kpi-value">{backtest_result.max_drawdown:.2%}</div><div class="kpi-label">Max Drawdown</div></div>
</div>

<h2>Strategy</h2>
<div class="card">
<p><strong>Name:</strong> {strategy_name} | <strong>Sport:</strong> {sport} | <strong>Market:</strong> {market}</p>
</div>

<h2>Configuration</h2>
<div class="card">
<p><strong>Starting Bankroll:</strong> ${starting_bankroll:,.2f} | <strong>Execution Time:</strong> {backtest_result.execution_time_ms:,.1f}ms</p>
</div>

<h2>Performance Metrics</h2>
<div class="card">
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{perf_rows}</tbody></table>
</div>

<div class="chart">
<h2>Cumulative P&L</h2>
{pnl_svg}
</div>

<div class="chart">
<h2>Drawdown Curve</h2>
{dd_svg}
</div>

<h2>Risk Analysis</h2>
<div class="card">
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Max Drawdown</td><td>{backtest_result.max_drawdown:.2%}</td></tr>
<tr><td>Average Drawdown</td><td>{backtest_result.average_drawdown:.2%}</td></tr>
</tbody></table>
</div>

<h2>CLV Summary</h2>
<div class="card">
<table><thead><tr><th>CLV Metric</th><th>Value</th></tr></thead>
<tbody>{clv_rows}</tbody></table>
</div>

<h2>Calibration</h2>
<div class="chart">
{cal_svg}
</div>
<div class="card">
<table><thead><tr><th>Calibration Metric</th><th>Value</th></tr></thead>
<tbody>{cal_rows}</tbody></table>
</div>

{f'<div class="warning"><h2>Warnings</h2>{warning_html}</div>' if warning_html else ''}
</div>
</body>
</html>"""
