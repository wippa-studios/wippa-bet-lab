from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SportAdapter(ABC):
    """Base class for sport-specific adapters."""

    sport: str = ""

    @abstractmethod
    def load_events(self, dataset_path: str) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def compute_features(self, event: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_closing_odds(self, event: dict[str, Any], market: str = "match_odds") -> float | None:
        ...

    @abstractmethod
    def get_result(self, event: dict[str, Any]) -> str | None:
        ...

    def is_eligible(self, event: dict[str, Any]) -> bool:
        return True
