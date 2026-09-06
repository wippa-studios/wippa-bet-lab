from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from betlab.backtester.engine import BacktestResult


class StrategyResult(BaseModel):
    id: str
    name: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    backtest_result: dict[str, Any] = Field(default_factory=dict)


class ComparisonReport(BaseModel):
    strategies: list[StrategyResult] = Field(default_factory=list)
    head_to_head_metrics: list[dict[str, Any]] = Field(default_factory=list)
    correlation_matrix: dict[str, Any] = Field(default_factory=dict)
    best_strategy_per_metric: dict[str, str] = Field(default_factory=dict)
    recommendation: str = ""


def compare_strategies(results_list: list[tuple[str, str, BacktestResult, dict[str, Any]]]) -> ComparisonReport:
    strategies = []
    for sid, name, bt, metrics in results_list:
        strategies.append(
            StrategyResult(
                id=sid,
                name=name,
                metrics=metrics,
                backtest_result={
                    "total_bets": bt.total_bets,
                    "wins": bt.wins,
                    "losses": bt.losses,
                    "net_profit": bt.net_profit,
                    "roi": bt.roi,
                    "yield_pct": bt.yield_pct,
                    "max_drawdown": bt.max_drawdown,
                    "average_drawdown": bt.average_drawdown,
                    "gross_profit": bt.gross_profit,
                },
            )
        )

    comparison_keys = ["net_profit", "roi", "yield_pct", "max_drawdown", "average_drawdown", "total_bets"]
    head_to_head = []
    for key in comparison_keys:
        vals = []
        for s in strategies:
            vals.append({"strategy": s.name, "value": s.backtest_result.get(key, 0.0)})
        head_to_head.append({"metric": key, "values": vals})

    best_per_metric: dict[str, str] = {}
    higher_is_better = {"net_profit", "roi", "yield_pct", "total_bets"}
    lower_is_better = {"max_drawdown", "average_drawdown"}

    for key in comparison_keys:
        if not strategies:
            continue
        best_name = strategies[0].name
        best_val = strategies[0].backtest_result.get(key, 0.0)
        for s in strategies[1:]:
            val = s.backtest_result.get(key, 0.0)
            if key in higher_is_better and val > best_val:
                best_val = val
                best_name = s.name
            elif key in lower_is_better and val < best_val:
                best_val = val
                best_name = s.name
        best_per_metric[key] = best_name

    correlation = _compute_correlation(strategies, comparison_keys)
    recommendation = _build_recommendation(strategies, best_per_metric)

    return ComparisonReport(
        strategies=strategies,
        head_to_head_metrics=head_to_head,
        correlation_matrix=correlation,
        best_strategy_per_metric=best_per_metric,
        recommendation=recommendation,
    )


def _compute_correlation(
    strategies: list[StrategyResult], keys: list[str]
) -> dict[str, Any]:
    if len(strategies) < 2:
        return {}
    result: dict[str, Any] = {}
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            if j <= i:
                continue
            pair_key = f"{s1.name} vs {s2.name}"
            common_metrics = set(s1.metrics.keys()) & set(s2.metrics.keys())
            if not common_metrics:
                result[pair_key] = 0.0
                continue
            v1 = [s1.metrics[k] for k in common_metrics if isinstance(s1.metrics[k], (int, float))]
            v2 = [s2.metrics[k] for k in common_metrics if isinstance(s2.metrics[k], (int, float))]
            if len(v1) < 2:
                result[pair_key] = 0.0
                continue
            mean1 = sum(v1) / len(v1)
            mean2 = sum(v2) / len(v2)
            cov = sum((a - mean1) * (b - mean2) for a, b in zip(v1, v2)) / len(v1)
            std1 = (sum((a - mean1) ** 2 for a in v1) / len(v1)) ** 0.5
            std2 = (sum((b - mean2) ** 2 for b in v2) / len(v2)) ** 0.5
            corr = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0.0
            result[pair_key] = round(corr, 4)
    return result


def _build_recommendation(
    strategies: list[StrategyResult], best_per_metric: dict[str, str]
) -> str:
    if not strategies:
        return "No strategies to compare."
    wins: dict[str, int] = {}
    for metric, name in best_per_metric.items():
        wins[name] = wins.get(name, 0) + 1
    if not wins:
        return "Insufficient data for recommendation."
    top_name = max(wins, key=wins.get)
    top_strategy = next(s for s in strategies if s.name == top_name)
    dd = top_strategy.backtest_result.get("max_drawdown", 0)
    roi = top_strategy.backtest_result.get("roi", 0)
    risk_note = "moderate risk" if dd < 0.15 else "elevated risk — consider reducing position sizes"
    return (
        f"Recommended: {top_name} — "
        f"ROI {roi:.2%}, max drawdown {dd:.2%} ({risk_note}). "
        f"Lead in {wins[top_name]}/{len(best_per_metric)} metrics."
    )
