import json

import pytest

from betlab.strategies.builder import StrategyBuilder, StrategyDefinition
from betlab.strategies.validator import StrategyValidator, StrategyValidationReport
from betlab.strategies.versioner import StrategyVersioner
from betlab.strategies.evaluator import StrategyEvaluator


class TestStrategyBuilder:
    def test_build_with_fluent_api(self):
        strategy = (
            StrategyBuilder()
            .set_name("Tennis Value Bet")
            .set_sport("tennis")
            .set_market("match_winner")
            .add_filter("p1_rank", "<=", 10)
            .add_filter("sport", "=", "tennis")
            .set_signal("p1_back_price", "<", "p1_fair_odds")
            .set_selection(side="p1")
            .set_price(source="betfair", minimum=1.2, maximum=5.0)
            .set_staking(method="kelly", fraction=0.25)
            .set_risk(max_open_exposure=1000, max_daily_loss=200)
            .set_validation(method="walk_forward", training_window=100, test_window=20, step=10)
            .build()
        )
        assert isinstance(strategy, StrategyDefinition)
        assert strategy.name == "Tennis Value Bet"
        assert strategy.sport == "tennis"
        assert strategy.market == "match_winner"
        assert len(strategy.filters) == 2
        assert strategy.signal.field == "p1_back_price"
        assert strategy.selection.side == "p1"
        assert strategy.price.source == "betfair"
        assert strategy.staking.method == "kelly"
        assert strategy.risk.max_open_exposure == 1000
        assert strategy.validation.training_window == 100

    def test_json_round_trip(self):
        original = (
            StrategyBuilder()
            .set_name("Test Strategy")
            .set_sport("tennis")
            .set_market("match_winner")
            .add_filter("p1_rank", "<=", 5)
            .set_signal("p1_back_price", "<", "p1_fair_odds")
            .set_selection(side="p1")
            .build()
        )
        builder_json = (
            StrategyBuilder()
            .set_name("Test Strategy")
            .set_sport("tennis")
            .set_market("match_winner")
            .add_filter("p1_rank", "<=", 5)
            .set_signal("p1_back_price", "<", "p1_fair_odds")
            .set_selection(side="p1")
            .to_json()
        )
        restored = StrategyBuilder.from_json(builder_json).build()
        assert restored.name == original.name
        assert restored.sport == original.sport
        assert len(restored.filters) == len(original.filters)
        assert restored.signal.field == original.signal.field
        assert restored.selection.side == original.selection.side


class TestStrategyEvaluator:
    def _make_strategy(self):
        return (
            StrategyBuilder()
            .set_name("Eval Test")
            .set_sport("tennis")
            .set_market("match_winner")
            .add_filter("sport", "=", "tennis")
            .add_filter("p1_rank", "<=", 10)
            .set_signal("p1_back_price", "<", 2.0)
            .set_selection(side="p1")
            .build()
        )

    def test_evaluate_passes_for_matching_row(self):
        strategy = self._make_strategy()
        row = {
            "sport": "tennis",
            "p1_rank": 3,
            "p1_back_price": 1.60,
        }
        evaluator = StrategyEvaluator()
        assert evaluator.evaluate(strategy, row)

    def test_evaluate_fails_filter(self):
        strategy = self._make_strategy()
        row = {
            "sport": "tennis",
            "p1_rank": 25,
            "p1_back_price": 1.60,
        }
        evaluator = StrategyEvaluator()
        assert not evaluator.evaluate(strategy, row)

    def test_evaluate_fails_signal(self):
        strategy = self._make_strategy()
        row = {
            "sport": "tennis",
            "p1_rank": 3,
            "p1_back_price": 2.50,
        }
        evaluator = StrategyEvaluator()
        assert not evaluator.evaluate(strategy, row)

    def test_evaluate_filters_only(self):
        strategy = self._make_strategy()
        row = {"sport": "tennis", "p1_rank": 3, "p1_back_price": 3.0}
        evaluator = StrategyEvaluator()
        assert evaluator.evaluate_filters(strategy, row)
        assert not evaluator.evaluate_signal(strategy, row)

    def test_evaluate_signal_only(self):
        strategy = self._make_strategy()
        row = {"sport": "football", "p1_rank": 50, "p1_back_price": 1.50}
        evaluator = StrategyEvaluator()
        assert not evaluator.evaluate_filters(strategy, row)
        assert evaluator.evaluate_signal(strategy, row)

    def test_operator_equals(self):
        strategy = StrategyBuilder().set_name("x").set_sport("tennis").set_market("match_winner").add_filter("status", "=", "active").build()
        evaluator = StrategyEvaluator()
        assert evaluator.evaluate_filters(strategy, {"status": "active"})
        assert not evaluator.evaluate_filters(strategy, {"status": "inactive"})

    def test_operator_in(self):
        strategy = StrategyBuilder().set_name("x").set_sport("tennis").set_market("match_winner").add_filter("tier", "in", ["gold", "silver"]).build()
        evaluator = StrategyEvaluator()
        assert evaluator.evaluate_filters(strategy, {"tier": "gold"})
        assert evaluator.evaluate_filters(strategy, {"tier": "silver"})
        assert not evaluator.evaluate_filters(strategy, {"tier": "bronze"})

    def test_operator_between(self):
        strategy = StrategyBuilder().set_name("x").set_sport("tennis").set_market("match_winner").add_filter("price", "between", [1.5, 3.0]).build()
        evaluator = StrategyEvaluator()
        assert evaluator.evaluate_filters(strategy, {"price": 2.0})
        assert evaluator.evaluate_filters(strategy, {"price": 1.5})
        assert not evaluator.evaluate_filters(strategy, {"price": 1.0})

    def test_missing_data_fails(self):
        strategy = self._make_strategy()
        row = {"sport": "tennis"}
        evaluator = StrategyEvaluator()
        assert not evaluator.evaluate(strategy, row)

    def test_missing_filter_field_returns_false(self):
        strategy = StrategyBuilder().set_name("x").set_sport("tennis").set_market("match_winner").add_filter("nonexistent", "=", "val").build()
        evaluator = StrategyEvaluator()
        assert not evaluator.evaluate_filters(strategy, {"other_field": "val"})

    def test_exists_operator(self):
        strategy = StrategyBuilder().set_name("x").set_sport("tennis").set_market("match_winner").add_filter("bonus", "exists", True).build()
        evaluator = StrategyEvaluator()
        assert evaluator.evaluate_filters(strategy, {"bonus": "yes"})
        assert not evaluator.evaluate_filters(strategy, {"other": "val"})


class TestStrategyValidator:
    def test_valid_strategy(self):
        strategy = (
            StrategyBuilder()
            .set_name("Valid")
            .set_sport("tennis")
            .set_market("match_winner")
            .build()
        )
        validator = StrategyValidator()
        report = validator.validate(strategy)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_missing_name(self):
        strategy = StrategyBuilder().set_sport("tennis").set_market("match_winner").build()
        validator = StrategyValidator()
        report = validator.validate(strategy)
        assert not report.is_valid
        assert any("name" in e.lower() for e in report.errors)

    def test_unknown_sport_warning(self):
        strategy = StrategyBuilder().set_name("x").set_sport("curling").set_market("match_winner").build()
        validator = StrategyValidator()
        report = validator.validate(strategy)
        assert report.is_valid
        assert any("curling" in w for w in report.warnings)

    def test_invalid_filter_operator(self):
        strategy = (
            StrategyBuilder()
            .set_name("x")
            .set_sport("tennis")
            .set_market("match_winner")
            .add_filter("field", "INVALID_OP", "val")
            .build()
        )
        validator = StrategyValidator()
        report = validator.validate(strategy)
        assert not report.is_valid
        assert any("operator" in e.lower() for e in report.errors)

    def test_risk_limits_reasonable(self):
        strategy = (
            StrategyBuilder()
            .set_name("x")
            .set_sport("tennis")
            .set_market("match_winner")
            .set_risk(max_open_exposure=100, max_daily_loss=200)
            .build()
        )
        validator = StrategyValidator()
        report = validator.validate(strategy)
        assert report.is_valid
        assert any("daily_loss" in w.lower() for w in report.warnings)


class TestStrategyVersioner:
    def test_create_and_list_versions(self):
        strategy = (
            StrategyBuilder()
            .set_name("Version Test")
            .set_sport("tennis")
            .set_market("match_winner")
            .build()
        )
        versioner = StrategyVersioner(":memory:")
        v1 = versioner.create_version(strategy)
        v2 = versioner.create_version(strategy)
        assert v1.version_number == 1
        assert v2.version_number == 2

        versions = versioner.list_versions(strategy.id)
        assert len(versions) == 2
        assert versions[0].version_number == 2

    def test_compute_hash_deterministic(self):
        strategy = (
            StrategyBuilder()
            .set_name("Hash Test")
            .set_sport("tennis")
            .set_market("match_winner")
            .add_filter("rank", "<=", 5)
            .build()
        )
        versioner = StrategyVersioner(":memory:")
        h1 = versioner.compute_hash(strategy)
        h2 = versioner.compute_hash(strategy)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_strategies_different_hashes(self):
        s1 = StrategyBuilder().set_name("A").set_sport("tennis").set_market("match_winner").build()
        s2 = StrategyBuilder().set_name("B").set_sport("tennis").set_market("match_winner").build()
        versioner = StrategyVersioner(":memory:")
        assert versioner.compute_hash(s1) != versioner.compute_hash(s2)
