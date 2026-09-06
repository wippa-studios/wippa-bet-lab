from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betlab.sports.tennis.adapter import TennisAdapter
from betlab.sports.greyhounds.adapter import GreyhoundAdapter
from betlab.sports.nba.adapter import NBAAdapter


# ── Tennis adapter tests ─────────────────────────────────────────────────────

class TestTennisAdapter:
    def test_compute_features(self):
        adapter = TennisAdapter()
        event = {
            "id": "e1",
            "competition": "ATP Tour",
            "metadata": {"surface": "clay", "is_indoor": False, "head_to_head": {"a": 3, "b": 2}},
            "participants": [
                {"id": "p1", "ranking": 5, "elo_rating": 1900.0},
                {"id": "p2", "ranking": 12, "elo_rating": 1750.0},
            ],
            "markets": [],
        }
        features = adapter.compute_features(event)
        assert features["rank_gap"] == 7
        assert features["elo_diff"] == 150.0
        assert features["surface"] == "clay"
        assert features["is_atp"] is True
        assert features["is_wta"] is False
        assert features["is_indoor"] is False

    def test_is_eligible_atp(self):
        adapter = TennisAdapter()
        event = {"id": "e1", "competition": "ATP Tour", "participants": [{"id": "p1"}, {"id": "p2"}]}
        assert adapter.is_eligible(event) is True

    def test_is_eligible_wta(self):
        adapter = TennisAdapter()
        event = {"id": "e1", "competition": "WTA Tour", "participants": [{"id": "p1"}, {"id": "p2"}]}
        assert adapter.is_eligible(event) is True

    def test_not_eligible_three_participants(self):
        adapter = TennisAdapter()
        event = {"id": "e1", "competition": "ATP", "participants": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]}
        assert adapter.is_eligible(event) is False

    def test_not_eligible_wrong_tour(self):
        adapter = TennisAdapter()
        event = {"id": "e1", "competition": "Challenger", "participants": [{"id": "p1"}, {"id": "p2"}]}
        assert adapter.is_eligible(event) is False

    def test_get_closing_odds(self):
        adapter = TennisAdapter()
        event = {
            "id": "e1",
            "markets": [
                {
                    "market_type": "match_odds",
                    "runners": [
                        {
                            "id": "r1",
                            "prices": [
                                {"back_price": 2.0},
                                {"back_price": 1.9},
                            ],
                        }
                    ],
                }
            ],
        }
        odds = adapter.get_closing_odds(event)
        assert odds == 1.9

    def test_get_result(self):
        adapter = TennisAdapter()
        event = {
            "id": "e1",
            "markets": [
                {
                    "runners": [
                        {"id": "r1", "result": "winner"},
                        {"id": "r2", "result": ""},
                    ]
                }
            ],
        }
        assert adapter.get_result(event) == "r1"


# ── Greyhound adapter tests ──────────────────────────────────────────────────

class TestGreyhoundAdapter:
    def test_compute_features(self):
        adapter = GreyhoundAdapter()
        event = {
            "id": "g1",
            "venue": "Wentworth Park",
            "metadata": {"trap": 3, "wom_ratio": 0.65, "venue_bias": 0.12},
            "markets": [
                {
                    "runners": [
                        {
                            "id": "dog1",
                            "prices": [
                                {"back_price": 3.0, "traded_volume": 100},
                                {"back_price": 2.5, "traded_volume": 200},
                            ],
                        }
                    ]
                }
            ],
        }
        features = adapter.compute_features(event)
        assert features["trap"] == 3
        assert features["wom_ratio"] == 0.65
        assert features["venue"] == "Wentworth Park"
        assert features["venue_bias"] == 0.12
        assert features["matched_volume"] == 300.0
        assert features["price_change"] == pytest.approx((2.5 - 3.0) / 3.0)

    def test_is_eligible(self):
        adapter = GreyhoundAdapter()
        assert adapter.is_eligible({"sport": "greyhounds"}) is True
        assert adapter.is_eligible({"sport": "dog"}) is True
        assert adapter.is_eligible({"sport": "tennis"}) is False

    def test_get_closing_odds(self):
        adapter = GreyhoundAdapter()
        event = {
            "id": "g1",
            "markets": [
                {
                    "market_name": "win",
                    "runners": [
                        {"id": "d1", "prices": [{"back_price": 4.0}]},
                    ],
                }
            ],
        }
        odds = adapter.get_closing_odds(event)
        assert odds == 4.0

    def test_get_result(self):
        adapter = GreyhoundAdapter()
        event = {
            "markets": [
                {"runners": [{"id": "d1", "result": "winner"}, {"id": "d2", "result": ""}]}
            ]
        }
        assert adapter.get_result(event) == "d1"


# ── NBA adapter tests ────────────────────────────────────────────────────────

class TestNBAAdapter:
    def test_compute_features(self):
        adapter = NBAAdapter()
        event = {
            "id": "n1",
            "home_away": True,
            "metadata": {"rest_days": [1, 2], "travel_distance": [200, 800]},
            "participants": [
                {
                    "id": "team1",
                    "metadata": {"offensive_rating": 115.0, "defensive_rating": 108.0, "pace": 100.5},
                },
                {
                    "id": "team2",
                    "metadata": {"offensive_rating": 110.0, "defensive_rating": 112.0, "pace": 99.5},
                },
            ],
            "markets": [],
        }
        features = adapter.compute_features(event)
        assert features["rest_days"] == [1, 2]
        assert features["travel_distance"] == [200, 800]
        assert features["home_advantage"] == 1.0
        assert features["offensive_rating_diff"] == 5.0
        assert features["defensive_rating_diff"] == -4.0
        assert features["pace"] == pytest.approx(100.0)

    def test_is_eligible(self):
        adapter = NBAAdapter()
        assert adapter.is_eligible({"sport": "nba"}) is True
        assert adapter.is_eligible({"sport": "basketball"}) is True
        assert adapter.is_eligible({"sport": "tennis"}) is False

    def test_get_closing_odds(self):
        adapter = NBAAdapter()
        event = {
            "markets": [
                {
                    "market_type": "moneyline",
                    "runners": [
                        {"id": "lakers", "prices": [{"back_price": 1.65}]},
                    ],
                }
            ]
        }
        odds = adapter.get_closing_odds(event)
        assert odds == 1.65

    def test_get_result(self):
        adapter = NBAAdapter()
        event = {
            "markets": [
                {"runners": [{"id": "lakers", "result": "winner"}, {"id": "celtics", "result": ""}]}
            ]
        }
        assert adapter.get_result(event) == "lakers"

    def test_home_advantage_absent(self):
        adapter = NBAAdapter()
        event = {"home_away": False, "participants": [], "markets": [], "metadata": {}}
        features = adapter.compute_features(event)
        assert features["home_advantage"] == 0.0
