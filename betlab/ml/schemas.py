from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TargetType(str, Enum):
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"
    regression = "regression"


class TargetConfig(BaseModel):
    column: str
    type: TargetType
    positive_class: Optional[str | int] = None
    exclude_values: list[Any] = Field(default_factory=list)
    minimum_class_count: int = 5


class FeatureConfig(BaseModel):
    column: str
    role: str = "feature"
    data_type: str = "numeric"
    missing_pct: float = 0.0
    unique_count: int = 0
    warning: Optional[str] = None
    suggested: bool = False


class ModelType(str, Enum):
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    gradient_boosting = "gradient_boosting"
    extra_trees = "extra_trees"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    dummy_baseline = "dummy_baseline"


class ModelConfig(BaseModel):
    type: ModelType
    parameters: dict[str, Any] = Field(default_factory=dict)
    probability_calibration: str = "none"


class PreprocessingConfig(BaseModel):
    numeric_imputation: str = "median"
    categorical_encoding: str = "one_hot"
    scaling: str = "standard"
    datetime_extraction: bool = True


class ValidationConfig(BaseModel):
    method: str = "walk_forward"
    training_window: int = 1000
    test_window: int = 200
    step: int = 200
    purge_gap: int = 10
    minimum_training_events: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    seed: int = 42


class BettingConversionConfig(BaseModel):
    probability_threshold: float = 0.55
    minimum_edge: float = 0.03
    minimum_odds: float = 1.2
    maximum_odds: float = 10.0
    market_side: str = "back"
    staking_method: str = "kelly"
    kelly_fraction: float = 0.25
    maximum_stake_percent: float = 5.0


class ModelBuilderState(BaseModel):
    dataset_id: str = ""
    dataset_version: str = ""
    target: TargetConfig
    features: list[FeatureConfig] = Field(default_factory=list)
    model: ModelConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    betting: BettingConversionConfig = Field(default_factory=BettingConversionConfig)


class DatasetColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_pct: float
    unique_count: int
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    sample_values: list[Any] = Field(default_factory=list)


class DatasetPreview(BaseModel):
    columns: list[DatasetColumnInfo]
    row_count: int
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrainResult(BaseModel):
    model_id: str
    metrics: dict[str, float]
    feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    calibration: dict[str, Any] = Field(default_factory=dict)
    predictions: Optional[list[dict[str, Any]]] = None
    warnings: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class ModelManifest(BaseModel):
    id: str
    name: str
    version: str
    target_config: TargetConfig
    feature_columns: list[str]
    model_type: ModelType
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_config: PreprocessingConfig
    validation_config: ValidationConfig
    dataset_id: str = ""
    dataset_version: str = ""
    metrics_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
