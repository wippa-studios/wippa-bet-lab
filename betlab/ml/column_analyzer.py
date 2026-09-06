from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from betlab.ml.schemas import DatasetColumnInfo, DatasetPreview, FeatureConfig


class ColumnAnalyzer:
    def analyze(self, df: pd.DataFrame) -> DatasetPreview:
        columns_info: list[DatasetColumnInfo] = []
        warnings: list[str] = []

        for col in df.columns:
            series = df[col]
            stats = self.compute_column_stats(series)
            col_type = self.detect_column_type(series)

            col_info = DatasetColumnInfo(
                name=col,
                dtype=col_type,
                missing_pct=stats["missing_pct"],
                unique_count=stats["unique_count"],
                min=stats.get("min"),
                max=stats.get("max"),
                mean=stats.get("mean"),
                median=stats.get("median"),
                sample_values=stats["sample_values"],
            )
            columns_info.append(col_info)

            if stats["missing_pct"] > 50:
                warnings.append(f"Column '{col}' has {stats['missing_pct']:.1f}% missing values")

        preview_rows = df.head(5).to_dict(orient="records")

        return DatasetPreview(
            columns=columns_info,
            row_count=len(df),
            preview_rows=preview_rows,
            warnings=warnings,
        )

    def detect_column_type(self, series: pd.Series) -> str:
        if series.dtype == bool or (
            series.dropna().isin([True, False, 0, 1]).all()
            and series.nunique() <= 3
        ):
            return "boolean"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_numeric_dtype(series):
            nunique = series.nunique()
            if nunique <= 2:
                return "boolean"
            if nunique > len(series) * 0.5 and nunique > 100:
                return "identifier"
            return "numeric"

        nunique = series.nunique()
        if nunique <= 2:
            return "boolean"
        if nunique == len(series) and nunique > 1:
            return "identifier"
        if nunique > len(series) * 0.5 and nunique > 100:
            return "text"
        return "categorical"

    def compute_column_stats(self, series: pd.Series) -> dict[str, Any]:
        total = len(series)
        missing = series.isna().sum()
        missing_pct = round((missing / total) * 100, 2) if total > 0 else 0.0
        unique_count = series.nunique()

        result: dict[str, Any] = {
            "missing_pct": missing_pct,
            "unique_count": int(unique_count),
            "sample_values": series.dropna().head(5).tolist(),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                result["min"] = float(clean.min())
                result["max"] = float(clean.max())
                result["mean"] = round(float(clean.mean()), 4)
                result["median"] = round(float(clean.median()), 4)

        return result

    def suggest_target(self, columns_info: list[DatasetColumnInfo]) -> list[str]:
        candidates: list[str] = []
        for col_info in columns_info:
            name = col_info.name
            lower = name.lower()
            if col_info.dtype == "identifier":
                continue
            if any(
                kw in lower
                for kw in [
                    "result",
                    "outcome",
                    "won",
                    "win",
                    "target",
                    "label",
                    "class",
                    "direction",
                ]
            ):
                if col_info.unique_count <= 20:
                    candidates.append(name)
        return candidates

    def suggest_features(
        self, columns_info: list[DatasetColumnInfo], target_col: str
    ) -> list[FeatureConfig]:
        suggestions: list[FeatureConfig] = []
        leakage = set(self.detect_leakage_columns(columns_info))

        for col_info in columns_info:
            name = col_info.name
            if name == target_col:
                continue
            if name in leakage:
                continue

            if col_info.dtype == "identifier":
                continue

            suggested = col_info.dtype in ("numeric", "categorical", "boolean")
            warning: str | None = None
            if col_info.missing_pct > 30:
                warning = f"High missing rate: {col_info.missing_pct}%"
                suggested = False

            suggestions.append(
                FeatureConfig(
                    column=name,
                    role="feature",
                    data_type=col_info.dtype,
                    missing_pct=col_info.missing_pct,
                    unique_count=col_info.unique_count,
                    warning=warning,
                    suggested=suggested,
                )
            )
        return suggestions

    def detect_leakage_columns(self, columns_info: list[DatasetColumnInfo]) -> list[str]:
        leakage_keywords = [
            "result",
            "outcome",
            "won",
            "win",
            "profit",
            "pnl",
            "return",
            "closing",
            "final",
            "settlement",
            "fair_odds",
            "true_prob",
            "target",
            "label",
        ]
        detected: list[str] = []
        for col_info in columns_info:
            lower = col_info.name.lower()
            if any(kw in lower for kw in leakage_keywords):
                detected.append(col_info.name)
        return detected

    def compute_correlation_matrix(
        self, df: pd.DataFrame, numeric_columns: list[str]
    ) -> dict[str, Any]:
        if not numeric_columns:
            return {"pairs": [], "matrix": {}}

        subset = df[numeric_columns].dropna()
        if len(subset) < 2:
            return {"pairs": [], "matrix": {}}

        corr = subset.corr()
        pairs: list[dict[str, Any]] = []

        for i, col_a in enumerate(numeric_columns):
            for col_b in numeric_columns[i + 1 :]:
                val = corr.loc[col_a, col_b]
                if not math.isnan(val):
                    pairs.append(
                        {
                            "column_a": col_a,
                            "column_b": col_b,
                            "correlation": round(float(val), 4),
                        }
                    )

        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        matrix = {
            col: {col2: round(float(corr.loc[col, col2]), 4) for col2 in numeric_columns}
            for col in numeric_columns
        }

        return {"pairs": pairs, "matrix": matrix}
