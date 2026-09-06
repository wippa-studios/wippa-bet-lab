from __future__ import annotations

import math
from dataclasses import dataclass, field

from betlab.core.schemas.models import SimulatedBet


@dataclass
class BucketEntry:
    predicted_prob: float = 0.0
    actual_rate: float = 0.0
    count: int = 0


@dataclass
class CalibrationReport:
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration_by_bucket: list[BucketEntry] = field(default_factory=list)


def compute_calibration(bets: list[SimulatedBet]) -> CalibrationReport:
    valid = [
        b for b in bets
        if b.result != "void"
        and b.clv_implied_probability is not None
        and b.clv_implied_probability > 0
    ]
    if not valid:
        return CalibrationReport()

    brier = sum(
        (b.clv_implied_probability - (1.0 if b.profit > 0 else 0.0)) ** 2
        for b in valid
    ) / len(valid)

    log_loss_sum = 0.0
    for b in valid:
        p = max(min(b.clv_implied_probability, 1.0 - 1e-15), 1e-15)
        if b.profit > 0:
            log_loss_sum += -math.log(p)
        else:
            log_loss_sum += -math.log(1.0 - p)
    log_loss = log_loss_sum / len(valid)

    buckets: dict[int, list[SimulatedBet]] = {i: [] for i in range(10)}
    for b in valid:
        bucket_idx = min(int(b.clv_implied_probability * 10), 9)
        buckets[bucket_idx].append(b)

    calibration_by_bucket = []
    for i in range(10):
        bucket_bets = buckets[i]
        if bucket_bets:
            predicted = sum(b.clv_implied_probability for b in bucket_bets) / len(bucket_bets)
            actual = sum(1 for b in bucket_bets if b.profit > 0) / len(bucket_bets)
            calibration_by_bucket.append(BucketEntry(
                predicted_prob=predicted,
                actual_rate=actual,
                count=len(bucket_bets),
            ))
        else:
            calibration_by_bucket.append(BucketEntry(
                predicted_prob=(i + 0.5) / 10.0,
                actual_rate=0.0,
                count=0,
            ))

    return CalibrationReport(
        brier_score=brier,
        log_loss=log_loss,
        calibration_by_bucket=calibration_by_bucket,
    )
