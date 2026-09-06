from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler

from betlab.ml.schemas import FeatureConfig, PreprocessingConfig


class PreprocessingPipeline:
    def __init__(
        self,
        config: PreprocessingConfig,
        feature_configs: list[FeatureConfig],
    ):
        self.config = config
        self.feature_configs = feature_configs
        self._fitted = False
        self._pipeline: ColumnTransformer | None = None
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._boolean_cols: list[str] = []
        self._feature_names_out: list[str] = []

    def fit(self, df: pd.DataFrame, target: str, features: list[str]) -> "PreprocessingPipeline":
        self._numeric_cols = []
        self._categorical_cols = []
        self._boolean_cols = []

        feat_map = {fc.column: fc for fc in self.feature_configs}

        for col in features:
            if col == target:
                continue
            fc = feat_map.get(col)
            dtype = fc.data_type if fc else "numeric"
            if dtype == "numeric":
                self._numeric_cols.append(col)
            elif dtype in ("categorical",):
                self._categorical_cols.append(col)
            elif dtype == "boolean":
                self._boolean_cols.append(col)

        transformers: list[tuple[str, Any, list[str]]] = []

        if self._numeric_cols:
            steps: list[tuple[str, Any]] = []
            if self.config.numeric_imputation == "median":
                steps.append(("imputer", SimpleImputer(strategy="median")))
            elif self.config.numeric_imputation == "mean":
                steps.append(("imputer", SimpleImputer(strategy="mean")))
            else:
                steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))

            if self.config.scaling == "standard":
                steps.append(("scaler", StandardScaler()))
            elif self.config.scaling == "robust":
                steps.append(("scaler", RobustScaler()))

            transformers.append(("numeric", Pipeline(steps), self._numeric_cols))

        if self._categorical_cols:
            if self.config.categorical_encoding == "one_hot":
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                transformers.append(("categorical", encoder, self._categorical_cols))
            elif self.config.categorical_encoding == "frequency":
                transformers.append(
                    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self._categorical_cols)
                )
            else:
                transformers.append(
                    ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self._categorical_cols)
                )

        if self._boolean_cols:
            transformers.append(("boolean", "passthrough", self._boolean_cols))

        if not transformers:
            transformers.append(("skip", "passthrough", features[:1] if features else ["__none__"]))

        self._pipeline = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
        self._pipeline.fit(df[features])
        self._fitted = True

        try:
            self._feature_names_out = list(self._pipeline.get_feature_names_out())
        except Exception:
            self._feature_names_out = self._numeric_cols + self._categorical_cols + self._boolean_cols

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Pipeline has not been fitted yet.")

        all_cols = self._numeric_cols + self._categorical_cols + self._boolean_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        result = self._pipeline.transform(df[all_cols])

        names = self._feature_names_out
        if result.shape[1] == len(names):
            return pd.DataFrame(result, columns=names, index=df.index)
        return pd.DataFrame(result, index=df.index)

    def get_feature_names_out(self) -> list[str]:
        return list(self._feature_names_out)
