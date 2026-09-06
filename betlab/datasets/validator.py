from __future__ import annotations

from dataclasses import dataclass, field

import duckdb


REQUIRED_COLUMNS = ["event_id", "start_time"]
PRICE_COLUMNS = [
    "p1_back_price", "p2_back_price",
    "p1_lay_price", "p2_lay_price",
    "p1_fair_odds", "p2_fair_odds",
]
RESULT_COLUMN = "result"


@dataclass
class ValidationReport:
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class DatasetValidator:
    def validate(self, dataset, conn: duckdb.DuckDBPyConnection) -> ValidationReport:
        warnings: list[str] = []
        errors: list[str] = []

        stats = {
            "row_count": dataset.row_count,
            "column_count": len(dataset.columns),
            "columns": dataset.columns,
        }

        missing = [c for c in REQUIRED_COLUMNS if c not in dataset.columns]
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")

        if "event_id" in dataset.columns:
            dup_count = conn.execute(
                f"SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM {dataset.table_name}"
            ).fetchone()[0]
            if dup_count > 0:
                errors.append(f"Found {dup_count} duplicate event_id values")

        if "start_time" in dataset.columns:
            invalid_ts = conn.execute(
                f"SELECT COUNT(*) FROM {dataset.table_name} WHERE TRY_CAST(start_time AS TIMESTAMP) IS NULL AND start_time IS NOT NULL"
            ).fetchone()[0]
            if invalid_ts > 0:
                errors.append(f"Found {invalid_ts} invalid timestamp values in start_time")

            future_count = conn.execute(
                f"SELECT COUNT(*) FROM {dataset.table_name} WHERE TRY_CAST(start_time AS TIMESTAMP) > NOW()"
            ).fetchone()[0]
            if future_count > 0:
                warnings.append(f"Found {future_count} rows with future timestamps")

        for col in PRICE_COLUMNS:
            if col in dataset.columns:
                invalid = conn.execute(
                    f"SELECT COUNT(*) FROM {dataset.table_name} WHERE TRY_CAST({col} AS DOUBLE) IS NOT NULL AND (TRY_CAST({col} AS DOUBLE) <= 0 OR TRY_CAST({col} AS DOUBLE) >= 1000)"
                ).fetchone()[0]
                if invalid > 0:
                    errors.append(f"Found {invalid} rows with invalid price in {col} (must be 0 < odds < 1000)")

        if RESULT_COLUMN in dataset.columns:
            valid_results = {"p1", "p2", "draw", "void", "layup", None}
            invalid = conn.execute(
                f"SELECT COUNT(*) FROM {dataset.table_name} WHERE result NOT IN ('p1', 'p2', 'draw', 'void', 'layup') AND result IS NOT NULL"
            ).fetchone()[0]
            if invalid > 0:
                warnings.append(f"Found {invalid} rows with non-standard result values")

        stats["warnings_count"] = len(warnings)
        stats["errors_count"] = len(errors)

        return ValidationReport(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            stats=stats,
        )
