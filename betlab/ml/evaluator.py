from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance


class ModelEvaluator:
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        y_pred = model.predict(X_test)
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_prob_positive = y_proba[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob_positive))
                metrics["brier_score"] = float(brier_score_loss(y_test, y_prob_positive))
                metrics["log_loss"] = float(log_loss(y_test, y_proba))
            else:
                metrics["log_loss"] = float(log_loss(y_test, y_proba))
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))

        return metrics

    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            raise ModelDoesNotSupportProbabilityError("Model does not support predict_proba")
        return model.predict_proba(X)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)

    def compute_feature_importance(
        self, model: Any, feature_names: list[str], X: pd.DataFrame, y: pd.Series
    ) -> list[dict[str, Any]]:
        importances: dict[str, float] = {}

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for name, val in zip(feature_names, imp):
                importances[name] = float(val)
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 1:
                for name, val in zip(feature_names, coef):
                    importances[name] = float(abs(val))
            else:
                mean_abs = np.mean(np.abs(coef), axis=0)
                for name, val in zip(feature_names, mean_abs):
                    importances[name] = float(val)

        try:
            perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            for name, imp_val in zip(feature_names, perm.importances_mean):
                key = f"{name}_permutation"
                importances[key] = float(abs(imp_val))
        except Exception:
            pass

        result = [{"feature": k, "importance": v} for k, v in importances.items()]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result

    def compute_calibration(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 10
    ) -> dict[str, Any]:
        if not hasattr(model, "predict_proba"):
            return {"bins": [], "brier_score": None}

        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] != 2:
            return {"bins": [], "brier_score": None}

        prob_positive = y_proba[:, 1]
        fraction_of_positives, mean_predicted = calibration_curve(y_test, prob_positive, n_bins=n_bins)

        bins = []
        for frac, mean_pred in zip(fraction_of_positives, mean_predicted):
            bins.append({"predicted_prob": float(mean_pred), "actual_rate": float(frac)})

        brier = float(brier_score_loss(y_test, prob_positive))
        return {"bins": bins, "brier_score": brier}

    def compute_confusion_matrix(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        y_pred = model.predict(X_test)
        labels = sorted(set(y_test) | set(y_pred))
        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        return {"matrix": matrix.tolist(), "labels": labels}

    def compute_roc_curve(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        if not hasattr(model, "predict_proba"):
            return {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0}

        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] != 2:
            return {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0}

        prob_positive = y_proba[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, prob_positive)
        auc = float(roc_auc_score(y_test, prob_positive))

        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": auc,
        }

    def compare_against_baselines(
        self, model_metrics: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, Any]:
        from sklearn.dummy import DummyClassifier

        baselines: dict[str, dict[str, Any]] = {}

        for strategy in ("most_frequent", "stratified", "uniform"):
            clf = DummyClassifier(strategy=strategy, random_state=42)
            clf.fit(X_test, y_test)
            y_pred = clf.predict(X_test)
            baselines[strategy] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            }

        return baselines


class ModelDoesNotSupportProbabilityError(Exception):
    pass
