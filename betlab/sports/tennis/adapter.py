from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from betlab.sports.base import SportAdapter


class TennisAdapter(SportAdapter):
    sport = "tennis"

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
        participants = event.get("participants", [])
        rankings = [p.get("ranking") for p in participants if p.get("ranking") is not None]
        elo_ratings = [p.get("elo_rating") for p in participants if p.get("elo_rating") is not None]

        rank_gap = 0
        if len(rankings) >= 2:
            rank_gap = abs(rankings[0] - rankings[1])

        elo_diff = 0.0
        if len(elo_ratings) >= 2:
            elo_diff = elo_ratings[0] - elo_ratings[1]

        surface = event.get("metadata", {}).get("surface", "unknown")
        is_indoor = event.get("metadata", {}).get("is_indoor", False)

        tour = event.get("competition", "").upper()
        is_atp = "ATP" in tour
        is_wta = "WTA" in tour

        h2h = event.get("metadata", {}).get("head_to_head", {})

        return {
            "rank_gap": rank_gap,
            "elo_diff": elo_diff,
            "surface": surface,
            "is_atp": is_atp,
            "is_wta": is_wta,
            "is_indoor": is_indoor,
            "head_to_head": h2h,
        }

    def get_closing_odds(self, event: dict[str, Any], market: str = "match_odds") -> float | None:
        for mkt in event.get("markets", []):
            if mkt.get("market_type") == market or mkt.get("market_name", "").lower() == market:
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
        participants = event.get("participants", [])
        if len(participants) != 2:
            return False
        tour = event.get("competition", "").upper()
        if "ATP" not in tour and "WTA" not in tour:
            return False
        return True
