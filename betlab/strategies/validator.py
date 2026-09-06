from __future__ import annotations

from dataclasses import dataclass, field

from betlab.strategies.builder import StrategyDefinition


VALID_SPORTS = {"tennis", "football", "basketball", "horse_racing", "greyhounds", "cricket", "baseball", "mma", "esports"}
VALID_MARKETS = {"match_winner", "set_winner", "game_winner", "handicap", "total_games", "correct_score", "over_under"}
VALID_OPERATORS = {"=", "!=", ">", ">=", "<", "<=", "in", "not_in", "between", "exists", "contains"}
VALID_STAKING_METHODS = {"flat", "kelly", "proportional", "fibonacci", "martingale", "custom"}


@dataclass
class StrategyValidationReport:
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class StrategyValidator:
    def validate(self, strategy: StrategyDefinition) -> StrategyValidationReport:
        warnings: list[str] = []
        errors: list[str] = []

        if not strategy.name:
            errors.append("Strategy name is required")

        if not strategy.sport:
            errors.append("Strategy sport is required")
        elif strategy.sport.lower() not in VALID_SPORTS:
            warnings.append(f"Sport '{strategy.sport}' is not in known sports list")

        if not strategy.market:
            errors.append("Strategy market is required")
        elif strategy.market.lower() not in VALID_MARKETS:
            warnings.append(f"Market '{strategy.market}' is not in known markets list")

        for i, filt in enumerate(strategy.filters):
            if not filt.field:
                errors.append(f"Filter {i}: field is required")
            if filt.operator not in VALID_OPERATORS:
                errors.append(f"Filter {i}: invalid operator '{filt.operator}'")

        if strategy.signal:
            if not strategy.signal.field:
                errors.append("Signal: field is required")
            if strategy.signal.operator not in VALID_OPERATORS:
                errors.append(f"Signal: invalid operator '{strategy.signal.operator}'")

        if strategy.staking:
            if strategy.staking.method not in VALID_STAKING_METHODS:
                warnings.append(f"Staking method '{strategy.staking.method}' is not a standard method")

        if strategy.risk:
            if strategy.risk.max_open_exposure is not None and strategy.risk.max_open_exposure <= 0:
                errors.append("Risk: max_open_exposure must be positive")
            if strategy.risk.max_daily_loss is not None and strategy.risk.max_daily_loss <= 0:
                errors.append("Risk: max_daily_loss must be positive")
            if (
                strategy.risk.max_open_exposure is not None
                and strategy.risk.max_daily_loss is not None
                and strategy.risk.max_daily_loss > strategy.risk.max_open_exposure
            ):
                warnings.append("Risk: max_daily_loss exceeds max_open_exposure")

        if strategy.validation:
            if strategy.validation.method not in ("walk_forward", "k_fold", "expanding_window", "random_split"):
                warnings.append(f"Validation method '{strategy.validation.method}' is not standard")
            if strategy.validation.training_window is not None and strategy.validation.training_window <= 0:
                errors.append("Validation: training_window must be positive")
            if strategy.validation.test_window is not None and strategy.validation.test_window <= 0:
                errors.append("Validation: test_window must be positive")

        return StrategyValidationReport(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
        )
