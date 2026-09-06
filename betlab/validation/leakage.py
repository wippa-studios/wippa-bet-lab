from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

from pydantic import BaseModel, Field


class LeakageReport(BaseModel):
    is_clean: bool = True
    warnings: list[str] = Field(default_factory=list)
    severity: str = "low"
    affected_rows: int = 0


class LeakageDetector:
    """Heuristic-based detector for data leakage in betting datasets."""

    def detect(self, strategy: Any, dataset_path: str | Path) -> LeakageReport:
        path = Path(dataset_path)
        if not path.exists():
            return LeakageReport(is_clean=False, warnings=["Dataset not found"])

        events = self._load(path)
        warnings: list[str] = []
        affected = 0

        affected += self._check_duplicate_events(events, warnings)
        self._check_timestamp_ordering(events, warnings)
        self._check_future_data_correlation(events, warnings)
        self._check_lookahead_joins(events, warnings)
        self._check_post_event_rankings(events, warnings)
        self._check_closing_price_usage(events, warnings)

        severity = "low"
        if len(warnings) >= 4 or affected > len(events) * 0.1:
            severity = "high"
        elif len(warnings) >= 2:
            severity = "medium"

        return LeakageReport(
            is_clean=len(warnings) == 0,
            warnings=warnings,
            severity=severity,
            affected_rows=affected,
        )

    # ── Individual checks ─────────────────────────────────────────────────

    def _check_duplicate_events(
        self, events: list[dict[str, Any]], warnings: list[str]
    ) -> int:
        seen: dict[str, int] = {}
        dupes = 0
        for e in events:
            eid = e.get("id", "")
            if eid in seen:
                seen[eid] += 1
                dupes += 1
            else:
                seen[eid] = 1
        dup_count = sum(1 for v in seen.values() if v > 1)
        if dup_count > 0:
            warnings.append(
                f"Duplicate event IDs detected: {dup_count} events appear more than once"
            )
        return dupes

    def _check_timestamp_ordering(
        self, events: list[dict[str, Any]], warnings: list[str]
    ) -> None:
        out_of_order = 0
        for i in range(1, len(events)):
            prev = events[i - 1].get("timestamp", events[i - 1].get("start_time", ""))
            curr = events[i].get("timestamp", events[i].get("start_time", ""))
            if prev and curr and prev > curr:
                out_of_order += 1
        if out_of_order > 0:
            warnings.append(
                f"Timestamps out of order: {out_of_order} events precede their predecessors"
            )

    def _check_future_data_correlation(
        self, events: list[dict[str, Any]], warnings: list[str]
    ) -> None:
        for e in events:
            for mkt in e.get("markets", []):
                for runner in mkt.get("runners", []):
                    prices = runner.get("prices", [])
                    result = runner.get("result", "")
                    if result and len(prices) > 1:
                        closing = prices[-1].get("back_price", 0)
                        if closing > 0:
                            result_val = 1.0 if result == "winner" else 0.0
                            implied = 1.0 / closing
                            if abs(implied - result_val) < 0.05:
                                warnings.append(
                                    f"Possible future data usage: runner {runner.get('id', '?')} "
                                    f"closing odds strongly predict result"
                                )
                                return

    def _check_lookahead_joins(
        self, events: list[dict[str, Any]], warnings: list[str]
    ) -> None:
        known_fields = {
            "id", "sport", "competition", "start_time", "venue", "status",
            "result_status", "home_away", "participants", "markets",
        }
        for e in events:
            extra = set(e.keys()) - known_fields
            for key in extra:
                if "result" in key.lower() or "outcome" in key.lower():
                    warnings.append(
                        f"Possible lookahead join: field '{key}' on event {e.get('id', '?')} "
                        f"may contain post-event data"
                    )

    def _check_post_event_rankings(
        self, events: list[dict[str, Any]], warnings: list[str]
    ) -> None:
        for e in events:
            start = e.get("start_time", "")
            for p in e.get("participants", []):
                meta = p.get("metadata", {})
                rank_ts = meta.get("ranking_timestamp", "")
                if rank_ts and start and rank_ts > start:
                    warnings.append(
                        f"Post-event ranking: participant {p.get('id', '?')} has ranking "
                        f"timestamp after event start"
                    )

    def _check_closing_price_usage(
        self, events: list[dict[str, Any]], warnings: list[str]
    ) -> None:
        for e in events:
            for mkt in e.get("markets", []):
                for runner in mkt.get("runners", []):
                    prices = runner.get("prices", [])
                    if len(prices) >= 2:
                        first_ts = prices[0].get("timestamp", "")
                        last_ts = prices[-1].get("timestamp", "")
                        event_ts = e.get("start_time", e.get("timestamp", ""))
                        if event_ts and last_ts and last_ts > event_ts:
                            warnings.append(
                                f"Closing price recorded after event start for runner "
                                f"{runner.get('id', '?')}"
                            )

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("events", [])
