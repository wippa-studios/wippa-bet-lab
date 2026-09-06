from __future__ import annotations

import random
from dataclasses import dataclass, field

from betlab.core.schemas.models import SimulatedBet


@dataclass
class MonteCarloReport:
    percentiles: dict[int, float] = field(default_factory=dict)
    probability_of_ruin: float = 0.0
    probability_of_doubling: float = 0.0
    expected_terminal: float = 0.0
    max_drawdown_distribution: list[float] = field(default_factory=list)


def monte_carlo(
    bets: list[SimulatedBet],
    n_simulations: int = 1000,
    starting_bankroll: float = 1000.0,
    seed: int | None = None,
) -> MonteCarloReport:
    if not bets:
        return MonteCarloReport()

    rng = random.Random(seed)
    profit_samples = [b.profit for b in bets if b.result != "void"]
    if not profit_samples:
        return MonteCarloReport()

    terminal_balances: list[float] = []
    max_dds: list[float] = []

    for _ in range(n_simulations):
        sampled = [rng.choice(profit_samples) for _ in range(len(profit_samples))]
        rng.shuffle(sampled)

        balance = starting_bankroll
        peak = balance
        max_dd = 0.0

        for pnl in sampled:
            balance += pnl
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        terminal_balances.append(balance)
        max_dds.append(max_dd)

    terminal_balances.sort()
    ruin_count = sum(1 for b in terminal_balances if b <= 0)
    double_count = sum(1 for b in terminal_balances if b >= starting_bankroll * 2)

    pcts = {}
    for p in [5, 25, 50, 75, 95]:
        idx = int(len(terminal_balances) * p / 100)
        idx = min(idx, len(terminal_balances) - 1)
        pcts[p] = terminal_balances[idx]

    return MonteCarloReport(
        percentiles=pcts,
        probability_of_ruin=ruin_count / n_simulations,
        probability_of_doubling=double_count / n_simulations,
        expected_terminal=sum(terminal_balances) / len(terminal_balances),
        max_drawdown_distribution=sorted(max_dds),
    )
