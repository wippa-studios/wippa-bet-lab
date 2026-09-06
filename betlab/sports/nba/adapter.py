from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from betlab.sports.base import SportAdapter


class NBAAdapter(SportAdapter):
    sport = "nba"

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
        meta = event.get("metadata", {})
        participants = event.get("participants", [])

        rest_days = meta.get("rest_days", [0, 0])
        travel_distance = meta.get("travel_distance", [0, 0])
        home_advantage = 1.0 if event.get("home_away", False) else 0.0

        off_ratings = [p.get("metadata", {}).get("offensive_rating", 0) for p in participants]
        def_ratings = [p.get("metadata", {}).get("defensive_rating", 0) for p in participants]
        pace_vals = [p.get("metadata", {}).get("pace", 0) for p in participants]

        offensive_rating_diff = off_ratings[0] - off_ratings[1] if len(off_ratings) >= 2 else 0.0
        defensive_rating_diff = def_ratings[0] - def_ratings[1] if len(def_ratings) >= 2 else 0.0
        pace = sum(pace_vals) / len(pace_vals) if pace_vals else 0.0

        return {
            "rest_days": rest_days,
            "travel_distance": travel_distance,
            "home_advantage": home_advantage,
            "offensive_rating_diff": offensive_rating_diff,
            "defensive_rating_diff": defensive_rating_diff,
            "pace": pace,
        }

    def get_closing_odds(self, event: dict[str, Any], market: str = "moneyline") -> float | None:
        for mkt in event.get("markets", []):
            mkt_type = mkt.get("market_type", "").lower()
            mkt_name = mkt.get("market_name", "").lower()
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
        return event.get("sport", "").lower() in ("nba", "basketball")
