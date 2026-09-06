from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from betlab.core.schemas.models import SimulatedBet


@dataclass
class CLVReport:
    mean_clv: float = 0.0
    median_clv: float = 0.0
    clv_positive_rate: float = 0.0
    clv_by_odds_band: dict[str, float] = field(default_factory=dict)
    clv_by_sport: dict[str, float] = field(default_factory=dict)


def compute_bet_clv(placed_odds: float, closing_odds: float) -> float:
    if placed_odds <= 1.0 or closing_odds <= 1.0:
        return 0.0
    placed_implied = 1.0 / placed_odds
    closing_implied = 1.0 / closing_odds
    return (closing_implied - placed_implied) / placed_implied * 100


def _odds_band(odds: float) -> str:
    if odds < 1.5:
        return "1.0-1.5"
    if odds < 2.0:
        return "1.5-2.0"
    if odds < 3.0:
        return "2.0-3.0"
    if odds < 5.0:
        return "3.0-5.0"
    if odds < 10.0:
        return "5.0-10.0"
    return "10.0+"


def compute_clv(bets: list[SimulatedBet]) -> CLVReport:
    valid = [b for b in bets if b.clv_odds and b.clv_odds > 1.0 and b.result != "void"]
    if not valid:
        return CLVReport()

    clv_values = [compute_bet_clv(b.matched_price or 0, b.clv_odds) for b in valid if b.matched_price]
    if not clv_values:
        return CLVReport()

    positive_count = sum(1 for v in clv_values if v > 0)

    band_values: dict[str, list[float]] = {}
    for b, v in zip(valid, clv_values):
        if b.matched_price:
            band = _odds_band(b.matched_price)
            band_values.setdefault(band, []).append(v)
    clv_by_band = {k: sum(v) / len(v) for k, v in band_values.items()}

    return CLVReport(
        mean_clv=sum(clv_values) / len(clv_values),
        median_clv=median(clv_values),
        clv_positive_rate=positive_count / len(clv_values),
        clv_by_odds_band=clv_by_band,
    )
