from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from betlab.ml.column_analyzer import ColumnAnalyzer
from betlab.ml.evaluator import ModelEvaluator
from betlab.ml.preprocessor import PreprocessingPipeline
from betlab.ml.schemas import (
    BettingConversionConfig,
    FeatureConfig,
    ModelConfig,
    ModelType,
    PreprocessingConfig,
    TargetConfig,
    TargetType,
    ValidationConfig,
)
from betlab.ml.strategy_converter import StrategyConverter
from betlab.ml.trainer import ModelTrainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_tennis_dataset.csv")


@pytest.fixture
def analyzer() -> ColumnAnalyzer:
    return ColumnAnalyzer()


@pytest.fixture
def binary_target() -> TargetConfig:
    return TargetConfig(column="result", type=TargetType.binary_classification, positive_class="p1")


@pytest.fixture
def feature_cols() -> list[str]:
    return [
        "p1_rank",
        "p2_rank",
        "rank_gap",
        "p1_back_price",
        "p2_back_price",
        "p1_elo",
        "p2_elo",
        "elo_diff",
    ]


# --- ColumnAnalyzer Tests ---


class TestColumnAnalyzer:
    def test_analyze_returns_preview(self, analyzer: ColumnAnalyzer, sample_df: pd.DataFrame):
        preview = analyzer.analyze(sample_df)
        assert preview.row_count == 15
        assert len(preview.columns) == sample_df.shape[1]
        assert preview.preview_rows.__class__ is list

    def test_detect_numeric_type(self, analyzer: ColumnAnalyzer):
        s = pd.Series([1.0, 2.5, 3.7, np.nan, 5.1])
        assert analyzer.detect_column_type(s) == "numeric"

    def test_detect_categorical_type(self, analyzer: ColumnAnalyzer):
        s = pd.Series(["hard", "clay", "grass", "hard", "clay"])
        assert analyzer.detect_column_type(s) == "categorical"

    def test_detect_boolean_type(self, analyzer: ColumnAnalyzer):
        s = pd.Series([True, False, True, False])
        assert analyzer.detect_column_type(s) == "boolean"

    def test_detect_identifier_type(self, analyzer: ColumnAnalyzer):
        s = pd.Series(range(1000))
        assert analyzer.detect_column_type(s) == "identifier"

    def test_suggest_features_excludes_target(self, analyzer: ColumnAnalyzer, sample_df: pd.DataFrame):
        preview = analyzer.analyze(sample_df)
        suggestions = analyzer.suggest_features(preview.columns, target_col="result")
        col_names = [f.column for f in suggestions]
        assert "result" not in col_names
        assert "event_id" not in col_names

    def test_suggest_target_identifies_result(self, analyzer: ColumnAnalyzer, sample_df: pd.DataFrame):
        preview = analyzer.analyze(sample_df)
        targets = analyzer.suggest_target(preview.columns)
        assert "result" in targets


# --- PreprocessingPipeline Tests ---


class TestPreprocessingPipeline:
    def test_handles_missing_values(self, sample_df: pd.DataFrame):
        df = sample_df.copy()
        df.loc[0, "p1_rank"] = np.nan
        df.loc[1, "p2_rank"] = np.nan
        config = PreprocessingConfig(numeric_imputation="median", scaling="standard")
        feats = [FeatureConfig(column=c, data_type="numeric") for c in ["p1_rank", "p2_rank", "rank_gap"]]
        pipe = PreprocessingPipeline(config, feats)
        pipe.fit(df, "result", ["p1_rank", "p2_rank", "rank_gap"])
        result = pipe.transform(df)
        assert result.isna().sum().sum() == 0
        assert result.shape[0] == 15

    def test_encodes_categoricals(self, sample_df: pd.DataFrame):
        config = PreprocessingConfig(categorical_encoding="one_hot", scaling="none")
        feats = [
            FeatureConfig(column="surface", data_type="categorical"),
            FeatureConfig(column="p1_rank", data_type="numeric"),
        ]
        pipe = PreprocessingPipeline(config, feats)
        pipe.fit(sample_df, "result", ["surface", "p1_rank"])
        result = pipe.transform(sample_df)
        assert result.shape[0] == 15
        assert result.shape[1] >= 3  # p1_rank + at least 2 surface dummies


# --- ModelTrainer Tests ---


class TestModelTrainer:
    def _make_trainer(self, model_type: ModelType) -> ModelTrainer:
        return ModelTrainer(
            model_config=ModelConfig(type=model_type),
            preprocessing_config=PreprocessingConfig(),
            validation_config=ValidationConfig(),
        )

    def test_trains_logistic_regression(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = self._make_trainer(ModelType.logistic_regression)
        result = trainer.train(sample_df, binary_target, feature_cols)
        assert result.model_id
        assert result.execution_time_ms > 0
        assert trainer.get_model() is not None

    def test_trains_random_forest(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = self._make_trainer(ModelType.random_forest)
        result = trainer.train(sample_df, binary_target, feature_cols)
        assert result.model_id
        assert "accuracy" in result.metrics

    def test_computes_metrics(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = self._make_trainer(ModelType.logistic_regression)
        result = trainer.train(sample_df, binary_target, feature_cols)
        expected_keys = {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}
        assert expected_keys.issubset(result.metrics.keys())

    def test_computes_feature_importance(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = self._make_trainer(ModelType.random_forest)
        result = trainer.train(sample_df, binary_target, feature_cols)
        assert len(result.feature_importance) > 0
        assert "feature" in result.feature_importance[0]
        assert "importance" in result.feature_importance[0]

    def test_computes_calibration(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = self._make_trainer(ModelType.logistic_regression)
        result = trainer.train(sample_df, binary_target, feature_cols)
        assert "brier_score" in result.calibration


# --- ModelEvaluator Tests ---


class TestModelEvaluator:
    def test_computes_accuracy(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = ModelTrainer(
            model_config=ModelConfig(type=ModelType.logistic_regression),
            preprocessing_config=PreprocessingConfig(),
            validation_config=ValidationConfig(),
        )
        result = trainer.train(sample_df, binary_target, feature_cols)
        evaluator = ModelEvaluator()
        preprocessor = trainer.get_preprocessor()
        X = preprocessor.transform(sample_df)
        y = sample_df[binary_target.column].map(lambda x: 1 if x == binary_target.positive_class else 0)
        metrics = evaluator.evaluate(trainer.get_model(), X, y)
        assert metrics["accuracy"] >= 0.0
        assert metrics["accuracy"] <= 1.0

    def test_computes_feature_importance(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = ModelTrainer(
            model_config=ModelConfig(type=ModelType.random_forest),
            preprocessing_config=PreprocessingConfig(),
            validation_config=ValidationConfig(),
        )
        result = trainer.train(sample_df, binary_target, feature_cols)
        evaluator = ModelEvaluator()
        preprocessor = trainer.get_preprocessor()
        X = preprocessor.transform(sample_df)
        y = sample_df[binary_target.column].map(lambda x: 1 if x == binary_target.positive_class else 0)
        names = preprocessor.get_feature_names_out()
        importance = evaluator.compute_feature_importance(trainer.get_model(), names, X, y)
        assert len(importance) > 0
        assert importance[0]["importance"] >= 0


# --- StrategyConverter Tests ---


class TestStrategyConverter:
    def test_produces_valid_strategy_json(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = ModelTrainer(
            model_config=ModelConfig(type=ModelType.logistic_regression),
            preprocessing_config=PreprocessingConfig(),
            validation_config=ValidationConfig(),
        )
        result = trainer.train(sample_df, binary_target, feature_cols)
        converter = StrategyConverter()
        strategy = converter.convert_to_strategy(
            trainer.get_model(), feature_cols, BettingConversionConfig(), result.metrics
        )
        assert "filters" in strategy
        assert "selection" in strategy
        assert "staking" in strategy
        assert strategy["type"] == "model_based"

    def test_includes_edge_threshold(self, sample_df: pd.DataFrame, binary_target: TargetConfig, feature_cols: list[str]):
        trainer = ModelTrainer(
            model_config=ModelConfig(type=ModelType.logistic_regression),
            preprocessing_config=PreprocessingConfig(),
            validation_config=ValidationConfig(),
        )
        result = trainer.train(sample_df, binary_target, feature_cols)
        converter = StrategyConverter()
        betting_config = BettingConversionConfig(minimum_edge=0.05)
        strategy = converter.convert_to_strategy(
            trainer.get_model(), feature_cols, betting_config, result.metrics
        )
        edge_filters = [f for f in strategy["filters"] if f["type"] == "edge_check"]
        assert len(edge_filters) == 1
        assert edge_filters[0]["value"] == 0.05


# --- Round-trip Test ---


class TestRoundTrip:
    def test_train_evaluate_convert(self, sample_df: pd.DataFrame):
        binary_target = TargetConfig(column="result", type=TargetType.binary_classification, positive_class="p1")
        feature_cols = ["p1_rank", "p2_rank", "rank_gap", "p1_back_price", "p2_back_price", "p1_elo", "p2_elo", "elo_diff"]

        trainer = ModelTrainer(
            model_config=ModelConfig(type=ModelType.random_forest),
            preprocessing_config=PreprocessingConfig(),
            validation_config=ValidationConfig(),
        )
        train_result = trainer.train(sample_df, binary_target, feature_cols)

        evaluator = ModelEvaluator()
        preprocessor = trainer.get_preprocessor()
        X = preprocessor.transform(sample_df)
        y = sample_df[binary_target.column].map(lambda x: 1 if x == binary_target.positive_class else 0)
        eval_metrics = evaluator.evaluate(trainer.get_model(), X, y)

        converter = StrategyConverter()
        strategy = converter.convert_to_strategy(
            trainer.get_model(), feature_cols, BettingConversionConfig(), eval_metrics
        )

        assert strategy["type"] == "model_based"
        assert len(strategy["filters"]) >= 3
        assert "model_probability" in [s["field"] for s in strategy["signals"]]
        assert strategy["staking"]["method"] == "kelly"
        assert train_result.metrics["accuracy"] >= 0.0
