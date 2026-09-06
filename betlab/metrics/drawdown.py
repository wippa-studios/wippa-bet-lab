from __future__ import annotations

from dataclasses import dataclass, field

from betlab.core.schemas.models import BankrollSnapshot


@dataclass
class DrawdownPeriod:
    start_index: int = 0
    end_index: int = 0
    depth: float = 0.0
    duration: int = 0


@dataclass
class DrawdownReport:
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    recovery_time: int = 0
    drawdown_periods: list[DrawdownPeriod] = field(default_factory=list)


def compute_drawdown(bankroll_history: list[BankrollSnapshot]) -> DrawdownReport:
    if not bankroll_history:
        return DrawdownReport()

    peak = bankroll_history[0].balance
    dd_values: list[float] = []
    periods: list[DrawdownPeriod] = []
    in_dd = False
    dd_start = 0
    dd_peak = peak
    dd_len = 0

    for i, snap in enumerate(bankroll_history):
        if snap.balance > peak:
            peak = snap.balance
            if in_dd:
                depth = (dd_peak - bankroll_history[dd_start].balance) / dd_peak if dd_peak > 0 else 0.0
                # Find the lowest point in this drawdown period
                min_bal = min(s.balance for s in bankroll_history[dd_start:i])
                actual_depth = (dd_peak - min_bal) / dd_peak if dd_peak > 0 else 0.0
                periods.append(DrawdownPeriod(
                    start_index=dd_start,
                    end_index=i,
                    depth=actual_depth,
                    duration=i - dd_start,
                ))
            in_dd = False
            dd_len = 0

        dd = (peak - snap.balance) / peak if peak > 0 else 0.0
        dd_values.append(dd)

        if dd > 0:
            if not in_dd:
                dd_start = i
                dd_peak = peak
                in_dd = True
            dd_len += 1

    # Close open drawdown at end
    if in_dd:
        min_bal = min(s.balance for s in bankroll_history[dd_start:])
        actual_depth = (dd_peak - min_bal) / dd_peak if dd_peak > 0 else 0.0
        periods.append(DrawdownPeriod(
            start_index=dd_start,
            end_index=len(bankroll_history) - 1,
            depth=actual_depth,
            duration=dd_len,
        ))

    max_dd = max(dd_values) if dd_values else 0.0
    avg_dd = sum(dd_values) / len(dd_values) if dd_values else 0.0
    max_dur = max((p.duration for p in periods), default=0)
    last_recovery = max((p.end_index - p.start_index for p in periods), default=0) if periods else 0

    return DrawdownReport(
        max_drawdown=max_dd,
        max_drawdown_duration=max_dur,
        average_drawdown=avg_dd,
        recovery_time=last_recovery,
        drawdown_periods=periods,
    )
