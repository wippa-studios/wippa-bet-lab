from __future__ import annotations

from typing import Any

import json
from pathlib import Path


class DataSplitter:
    """Splits datasets into train/test by time, ratio, or k-fold."""

    def split_by_time(
        self, dataset: list[dict[str, Any]], train_end: str, test_start: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train = [
            e for e in dataset
            if self._ts(e) <= train_end
        ]
        test = [
            e for e in dataset
            if self._ts(e) >= test_start
        ]
        return train, test

    def split_by_ratio(
        self, dataset: list[dict[str, Any]], train_ratio: float = 0.8
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        n = len(dataset)
        split_idx = int(n * train_ratio)
        return dataset[:split_idx], dataset[split_idx:]

    def k_fold_split(
        self, dataset: list[dict[str, Any]], k: int = 5
    ) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
        n = len(dataset)
        fold_size = n // k
        folds: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

        for i in range(k):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < k - 1 else n
            test = dataset[test_start:test_end]
            train = dataset[:test_start] + dataset[test_end:]
            folds.append((train, test))

        return folds

    @staticmethod
    def _ts(event: dict[str, Any]) -> str:
        return str(event.get("timestamp", event.get("start_time", "")))
