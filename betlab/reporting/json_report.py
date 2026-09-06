from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from betlab.backtester.engine import BacktestResult


def generate_json_report(
    experiment_id: str,
    backtest_result: BacktestResult,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}

    perf = metrics.get("performance", {})
    clv = metrics.get("clv", backtest_result.clv_summary)
    calibration = metrics.get("calibration", backtest_result.calibration)
    drawdown = metrics.get("drawdown", {})
    monte_carlo = metrics.get("monte_carlo", {})

    return {
        "report_version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat(),
        "experiment": {
            "id": experiment_id,
            "status": "completed",
            "execution_time_ms": backtest_result.execution_time_ms,
        },
        "strategy": {
            "id": getattr(backtest_result, "strategy_id", ""),
            "name": getattr(backtest_result, "strategy_name", ""),
            "sport": getattr(backtest_result, "sport", ""),
            "market": getattr(backtest_result, "market", ""),
            "filters": getattr(backtest_result, "filters", []),
            "signal": getattr(backtest_result, "signal", {}),
            "selection": getattr(backtest_result, "selection", {}),
            "price": getattr(backtest_result, "price", {}),
            "staking": getattr(backtest_result, "staking", {}),
            "risk": getattr(backtest_result, "risk", {}),
        },
        "dataset": {
            "id": getattr(backtest_result, "dataset_id", ""),
            "name": getattr(backtest_result, "dataset_name", ""),
            "event_count": getattr(backtest_result, "event_count", 0),
            "market_count": getattr(backtest_result, "market_count", 0),
        },
        "simulation": {
            "starting_bankroll": getattr(backtest_result, "starting_bankroll", 1000.0),
            "seed": getattr(backtest_result, "seed", None),
        },
        "metrics": {
            "performance": perf,
            "clv": {
                "mean_clv": clv.get("mean_clv", 0.0) if isinstance(clv, dict) else getattr(clv, "mean_clv", 0.0),
                "median_clv": clv.get("median_clv", 0.0) if isinstance(clv, dict) else getattr(clv, "median_clv", 0.0),
                "positive_rate": clv.get("clv_positive_rate", 0.0) if isinstance(clv, dict) else getattr(clv, "clv_positive_rate", 0.0),
            },
            "calibration": {
                "brier_score": calibration.get("brier_score", 0.0) if isinstance(calibration, dict) else getattr(calibration, "brier_score", 0.0),
                "log_loss": calibration.get("log_loss", 0.0) if isinstance(calibration, dict) else getattr(calibration, "log_loss", 0.0),
            },
            "drawdown": {
                "max_drawdown": backtest_result.max_drawdown,
                "average_drawdown": backtest_result.average_drawdown,
            },
            "monte_carlo": monte_carlo,
        },
        "summary": {
            "total_bets": backtest_result.total_bets,
            "wins": backtest_result.wins,
            "losses": backtest_result.losses,
            "voids": backtest_result.voids,
            "gross_profit": round(backtest_result.gross_profit, 2),
            "net_profit": round(backtest_result.net_profit, 2),
            "roi": round(backtest_result.roi, 4),
            "yield_pct": round(backtest_result.yield_pct, 4),
            "max_drawdown": round(backtest_result.max_drawdown, 4),
        },
        "bankroll_history": {
            "total_snapshots": len(backtest_result.bankroll_history),
            "start_balance": backtest_result.bankroll_history[0]["balance"] if backtest_result.bankroll_history else 0.0,
            "end_balance": backtest_result.bankroll_history[-1]["balance"] if backtest_result.bankroll_history else 0.0,
            "snapshots": backtest_result.bankroll_history[:10],
        },
        "warnings": backtest_result.warnings,
    }
