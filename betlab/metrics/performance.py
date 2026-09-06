from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from betlab.core.schemas.models import BankrollSnapshot, SimulatedBet, BetStatus


@dataclass
class PerformanceMetrics:
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    strike_rate: float = 0.0
    gross_profit: float = 0.0
    net_profit: float = 0.0
    roi: float = 0.0
    yield_pct: float = 0.0
    average_odds: float = 0.0
    average_stake: float = 0.0
    turnover: float = 0.0
    commission_paid: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_consecutive_losses: int = 0
    max_drawdown: float = 0.0
    average_drawdown: float = 0.0
    drawdown_duration: int = 0
    recovery_time: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0


def _is_win(bet: SimulatedBet) -> bool:
    return bet.result == BetStatus.settled and bet.profit > 0


def _is_void(bet: SimulatedBet) -> bool:
    return bet.result == BetStatus.void


def _is_settled(bet: SimulatedBet) -> bool:
    return bet.result in (BetStatus.settled,)


def compute_performance(
    bets: list[SimulatedBet],
    bankroll_history: list[BankrollSnapshot] | None = None,
    starting_bankroll: float = 1000.0,
) -> PerformanceMetrics:
    if not bets:
        return PerformanceMetrics()

    settled = [b for b in bets if b.result in (BetStatus.settled, BetStatus.void)]
    valid = [b for b in settled if b.result != BetStatus.void]
    total_bets = len(bets)
    wins = sum(1 for b in valid if b.profit > 0)
    losses = sum(1 for b in valid if b.profit <= 0)
    voids = sum(1 for b in settled if b.result == BetStatus.void)

    strike_rate = wins / len(valid) if valid else 0.0
    gross_profit = sum(b.profit for b in valid if b.profit > 0)
    gross_losses = abs(sum(b.profit for b in valid if b.profit < 0))
    net_profit = sum(b.profit for b in valid)
    turnover = sum(b.stake for b in valid)
    commission_paid = sum(b.commission for b in valid)
    odds_list = [b.matched_price for b in valid if b.matched_price and b.matched_price > 0]
    average_odds = sum(odds_list) / len(odds_list) if odds_list else 0.0
    average_stake = turnover / len(valid) if valid else 0.0

    roi = net_profit / starting_bankroll * 100 if starting_bankroll > 0 else 0.0
    yield_pct = net_profit / turnover * 100 if turnover > 0 else 0.0
    profit_factor = gross_profit / gross_losses if gross_losses > 0 else float("inf") if gross_profit > 0 else 0.0
    expectancy = net_profit / len(valid) if valid else 0.0

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for b in valid:
        if b.profit <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Drawdown from bankroll history
    max_dd = 0.0
    avg_dd = 0.0
    dd_durations: list[int] = []
    recovery_times: list[int] = []

    if bankroll_history:
        peak = bankroll_history[0].balance if bankroll_history else starting_bankroll
        dd_values: list[float] = []
        in_dd = False
        dd_start = 0
        dd_len = 0

        for i, snap in enumerate(bankroll_history):
            if snap.balance > peak:
                peak = snap.balance
                if in_dd:
                    dd_durations.append(dd_len)
                    recovery_times.append(i - dd_start)
                in_dd = False
                dd_len = 0
            dd = (peak - snap.balance) / peak if peak > 0 else 0.0
            dd_values.append(dd)
            max_dd = max(max_dd, dd)
            if dd > 0:
                if not in_dd:
                    dd_start = i
                    in_dd = True
                dd_len += 1

        if in_dd:
            dd_durations.append(dd_len)
        avg_dd = sum(dd_values) / len(dd_values) if dd_values else 0.0

    # Volatility & downside deviation
    returns = [b.profit / starting_bankroll for b in valid] if starting_bankroll > 0 else []
    if returns:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        neg_returns = [r for r in returns if r < 0]
        if neg_returns:
            down_var = sum(r ** 2 for r in neg_returns) / len(neg_returns)
            downside_dev = down_var ** 0.5
        else:
            downside_dev = 0.0
    else:
        volatility = 0.0
        downside_dev = 0.0

    return PerformanceMetrics(
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        voids=voids,
        strike_rate=strike_rate,
        gross_profit=gross_profit,
        net_profit=net_profit,
        roi=roi,
        yield_pct=yield_pct,
        average_odds=average_odds,
        average_stake=average_stake,
        turnover=turnover,
        commission_paid=commission_paid,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_consecutive_losses=max_consec,
        max_drawdown=max_dd,
        average_drawdown=avg_dd,
        drawdown_duration=max(dd_durations) if dd_durations else 0,
        recovery_time=max(recovery_times) if recovery_times else 0,
        volatility=volatility,
        downside_deviation=downside_dev,
    )
