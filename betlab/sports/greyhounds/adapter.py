from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from betlab.sports.base import SportAdapter


class GreyhoundAdapter(SportAdapter):
    sport = "greyhounds"

    def load_events(self, dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        if not path.exists():
            return []
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("events", [])

    def compute_features(self, event: dict[str, Any]) -> dict[str, Any]:
        runners = []
        for mkt in event.get("markets", []):
            runners.extend(mkt.get("runners", []))

        trap = event.get("metadata", {}).get("trap", 0)
        wom_ratio = event.get("metadata", {}).get("wom_ratio", 0.0)
        matched_volume = 0.0
        price_changes: list[float] = []

        for runner in runners:
            prices = runner.get("prices", [])
            if prices and len(prices) >= 2:
                first = float(prices[0].get("back_price", 0))
                last = float(prices[-1].get("back_price", 0))
                if first > 0:
                    price_changes.append((last - first) / first)
                vol = sum(float(p.get("traded_volume", 0) or 0) for p in prices)
                matched_volume += vol

        avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0.0
        venue = event.get("venue", "")
        venue_bias = event.get("metadata", {}).get("venue_bias", 0.0)

        return {
            "trap": trap,
            "wom_ratio": wom_ratio,
            "matched_volume": matched_volume,
            "price_change": avg_price_change,
            "venue": venue,
            "venue_bias": venue_bias,
        }

    def get_closing_odds(self, event: dict[str, Any], market: str = "win") -> float | None:
        for mkt in event.get("markets", []):
            mkt_name = mkt.get("market_name", "").lower()
            mkt_type = mkt.get("market_type", "").lower()
            if mkt_type == market or mkt_name == market:
                for runner in mkt.get("runners", []):
                    prices = runner.get("prices", [])
                    if prices:
                        return float(prices[-1].get("back_price", 0))
        for mkt in event.get("markets", []):
            for runner in mkt.get("runners", []):
                prices = runner.get("prices", [])
                if prices:
                    return float(prices[-1].get("back_price", 0))
        return None

    def get_result(self, event: dict[str, Any]) -> str | None:
        for mkt in event.get("markets", []):
            for runner in mkt.get("runners", []):
                result = runner.get("result", "")
                if result:
                    return runner.get("id")
        return None

    def is_eligible(self, event: dict[str, Any]) -> bool:
        return event.get("sport", "").lower() in ("greyhounds", "greyhound", "dog")
