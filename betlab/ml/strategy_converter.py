from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from betlab.ml.schemas import BettingConversionConfig


class StrategyConverter:
    def convert_to_strategy(
        self,
        model: Any,
        feature_columns: list[str],
        betting_config: BettingConversionConfig | None = None,
        model_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if betting_config is None:
            betting_config = BettingConversionConfig()
        if model_metrics is None:
            model_metrics = {}

        strategy = {
            "name": "model_based_strategy",
            "version": "1.0",
            "type": "model_based",
            "model": {
                "model_type": type(model).__name__ if model is not None else "unknown",
                "feature_columns": feature_columns,
                "metrics": {
                    k: v
                    for k, v in model_metrics.items()
                    if isinstance(v, (int, float))
                },
            },
            "filters": [
                {
                    "type": "probability_threshold",
                    "field": "model_probability",
                    "operator": ">=",
                    "value": betting_config.probability_threshold,
                },
                {
                    "type": "edge_check",
                    "field": "edge",
                    "operator": ">=",
                    "value": betting_config.minimum_edge,
                },
                {
                    "type": "odds_range",
                    "field": "odds",
                    "operator": "between",
                    "min": betting_config.minimum_odds,
                    "max": betting_config.maximum_odds,
                },
            ],
            "signals": [
                {
                    "type": "model_probability",
                    "field": "model_probability",
                    "description": "Model predicted probability of the positive outcome",
                }
            ],
            "selection": {
                "market_side": betting_config.market_side,
                "description": f"Select if model probability >= {betting_config.probability_threshold} and edge >= {betting_config.minimum_edge}",
            },
            "staking": {
                "method": betting_config.staking_method,
                "kelly_fraction": betting_config.kelly_fraction,
                "maximum_stake_percent": betting_config.maximum_stake_percent,
                "description": f"{betting_config.staking_method} staking with {betting_config.kelly_fraction} Kelly fraction, max {betting_config.maximum_stake_percent}% stake",
            },
            "edge_calculation": {
                "formula": "model_probability - (1 / odds)",
                "minimum_edge": betting_config.minimum_edge,
                "commission_adjusted": True,
            },
        }
        return strategy

    def estimate_edge(self, model: Any, X_row: pd.DataFrame, odds: float) -> float:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_row)
            if proba.shape[1] == 2:
                model_prob = proba[0, 1]
            else:
                model_prob = float(np.max(proba[0]))
        else:
            pred = model.predict(X_row)
            model_prob = float(pred[0])

        implied_prob = 1.0 / odds if odds > 0 else 0.0
        return float(model_prob - implied_prob)

    def estimate_expected_return(
        self,
        model_probability: float,
        odds: float,
        commission_rate: float = 0.05,
    ) -> float:
        implied_prob = 1.0 / odds if odds > 0 else 0.0
        p_win = model_probability
        p_lose = 1.0 - p_win
        profit_if_win = (odds - 1.0) * (1.0 - commission_rate)
        ev = (p_win * profit_if_win) - (p_lose * 1.0)
        return float(ev)
