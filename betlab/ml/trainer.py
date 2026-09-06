from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from betlab.ml.evaluator import ModelEvaluator
from betlab.ml.preprocessor import PreprocessingPipeline
from betlab.ml.schemas import (
    FeatureConfig,
    ModelConfig,
    ModelType,
    PreprocessingConfig,
    TargetConfig,
    TrainResult,
    ValidationConfig,
)


class ModelTrainer:
    def __init__(
        self,
        model_config: ModelConfig,
        preprocessing_config: PreprocessingConfig,
        validation_config: ValidationConfig,
        seed: int = 42,
    ):
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.validation_config = validation_config
        self.seed = seed
        self._evaluator = ModelEvaluator()
        self._preprocessor: PreprocessingPipeline | None = None
        self._model: Any = None
        self._feature_names: list[str] = []

    def train(
        self,
        df: pd.DataFrame,
        target_config: TargetConfig,
        feature_columns: list[str],
    ) -> TrainResult:
        start = time.time()
        warnings: list[str] = []

        target_series = df[target_config.column]

        positive_class = target_config.positive_class
        if positive_class is not None:
            target_series = target_series.map(lambda x: 1 if x == positive_class else 0)

        exclude_values = target_config.exclude_values
        if exclude_values:
            mask = ~target_series.isin(exclude_values)
            df = df[mask].reset_index(drop=True)
            target_series = target_series[mask].reset_index(drop=True)

        feature_configs = [
            FeatureConfig(column=c, data_type="numeric") for c in feature_columns
        ]

        self._preprocessor = PreprocessingPipeline(self.preprocessing_config, feature_configs)
        self._preprocessor.fit(df, target_config.column, feature_columns)
        X = self._preprocessor.transform(df)
        y = target_series
        self._feature_names = self._preprocessor.get_feature_names_out()

        self._model = self._build_model()
        self._model.fit(X, y)

        if self.model_config.probability_calibration != "none" and hasattr(self._model, "predict_proba"):
            method = self.model_config.probability_calibration
            self._model = CalibratedClassifierCV(self._model, method=method, cv=3)
            self._model.fit(X, y)

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        metrics = self._evaluator.evaluate(self._model, pd.DataFrame(X_array, columns=self._feature_names), pd.Series(y_array))
        feature_importance = self._evaluator.compute_feature_importance(
            self._model, self._feature_names, pd.DataFrame(X_array, columns=self._feature_names), pd.Series(y_array)
        )
        calibration = self._evaluator.compute_calibration(
            self._model, pd.DataFrame(X_array, columns=self._feature_names), pd.Series(y_array)
        )

        elapsed_ms = (time.time() - start) * 1000

        if len(df) < self.validation_config.minimum_training_events:
            warnings.append(f"Only {len(df)} events (minimum: {self.validation_config.minimum_training_events})")

        small_preds = None
        if len(df) <= 50:
            preds = self._model.predict_proba(X_array)[:, 1] if hasattr(self._model, "predict_proba") else None
            if preds is not None:
                small_preds = [
                    {"index": int(i), "prediction": float(preds[i])}
                    for i in range(len(preds))
                ]

        return TrainResult(
            model_id=str(uuid.uuid4()),
            metrics=metrics,
            feature_importance=feature_importance,
            calibration=calibration,
            predictions=small_preds,
            warnings=warnings,
            execution_time_ms=round(elapsed_ms, 1),
        )

    def _build_model(self) -> Any:
        mt = self.model_config.type
        params = {k: v for k, v in self.model_config.parameters.items()}

        if mt == ModelType.logistic_regression:
            params.setdefault("max_iter", 1000)
            params.setdefault("random_state", self.seed)
            return LogisticRegression(**params)
        elif mt == ModelType.random_forest:
            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", self.seed)
            params.setdefault("n_jobs", -1)
            return RandomForestClassifier(**params)
        elif mt == ModelType.gradient_boosting:
            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", self.seed)
            return GradientBoostingClassifier(**params)
        elif mt == ModelType.extra_trees:
            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", self.seed)
            params.setdefault("n_jobs", -1)
            return ExtraTreesClassifier(**params)
        elif mt == ModelType.dummy_baseline:
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(strategy="most_frequent", random_state=self.seed)
        else:
            raise ValueError(f"Unsupported model type: {mt}")

    def get_model(self) -> Any:
        return self._model

    def get_preprocessor(self) -> PreprocessingPipeline | None:
        return self._preprocessor

    def to_joblib(self, model: Any, path: str) -> None:
        joblib.dump(model, path)

    def from_joblib(self, path: str) -> Any:
        return joblib.load(path)
