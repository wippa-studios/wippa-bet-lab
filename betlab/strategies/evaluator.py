from __future__ import annotations

from betlab.strategies.builder import StrategyDefinition


class StrategyEvaluator:
    def evaluate(self, strategy: StrategyDefinition, row: dict) -> bool:
        return self.evaluate_filters(strategy, row) and self.evaluate_signal(strategy, row)

    def evaluate_filters(self, strategy: StrategyDefinition, row: dict) -> bool:
        for filt in strategy.filters:
            if not self._eval_condition(filt.field, filt.operator, filt.value, row):
                return False
        return True

    def evaluate_signal(self, strategy: StrategyDefinition, row: dict) -> bool:
        if strategy.signal is None:
            return True
        return self._eval_condition(
            strategy.signal.field,
            strategy.signal.operator,
            strategy.signal.value,
            row,
        )

    def _eval_condition(self, field: str, operator: str, value: any, row: dict) -> bool:
        if field not in row or row[field] is None:
            if operator == "exists":
                return False
            return False

        cell = row[field]

        try:
            cell_num = float(cell)
            val_num = float(value) if not isinstance(value, (list, tuple)) else None
        except (ValueError, TypeError):
            cell_num = None
            val_num = None

        if operator == "=":
            if cell_num is not None and val_num is not None:
                return cell_num == val_num
            return str(cell) == str(value)
        elif operator == "!=":
            if cell_num is not None and val_num is not None:
                return cell_num != val_num
            return str(cell) != str(value)
        elif operator == ">":
            if cell_num is not None and val_num is not None:
                return cell_num > val_num
            return False
        elif operator == ">=":
            if cell_num is not None and val_num is not None:
                return cell_num >= val_num
            return False
        elif operator == "<":
            if cell_num is not None and val_num is not None:
                return cell_num < val_num
            return False
        elif operator == "<=":
            if cell_num is not None and val_num is not None:
                return cell_num <= val_num
            return False
        elif operator == "in":
            if isinstance(value, (list, tuple)):
                return cell in value or (cell_num is not None and any(
                    cell_num == float(v) for v in value if _is_numeric(v)
                ))
            return cell == value
        elif operator == "not_in":
            if isinstance(value, (list, tuple)):
                return cell not in value and not (cell_num is not None and any(
                    cell_num == float(v) for v in value if _is_numeric(v)
                ))
            return cell != value
        elif operator == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2 and cell_num is not None:
                lo, hi = float(value[0]), float(value[1])
                return lo <= cell_num <= hi
            return False
        elif operator == "exists":
            return True
        elif operator == "contains":
            return str(value) in str(cell)
        else:
            return False


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False
