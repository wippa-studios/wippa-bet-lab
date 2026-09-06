"""
Wippa Bet Lab — Model Builder Demo
====================================
Full Model Builder workflow: load data, analyze columns, train multiple
models, compare results, export the best model as a strategy, and backtest.
"""

from pathlib import Path

import pandas as pd

from betlab.ml.column_analyzer import ColumnAnalyzer
from betlab.ml.schemas import (
    BettingConversionConfig,
    ModelConfig,
    ModelType,
    PreprocessingConfig,
    TargetConfig,
    TargetType,
    ValidationConfig,
)
from betlab.ml.trainer import ModelTrainer
from betlab.ml.strategy_converter import StrategyConverter

FIXTURE = Path("tests/fixtures/sample_tennis_dataset.csv")


def train_and_evaluate(model_type, df, target_config, feature_columns, label):
    """Train a model and return results."""
    trainer = ModelTrainer(
        model_config=ModelConfig(type=model_type),
        preprocessing_config=PreprocessingConfig(),
        validation_config=ValidationConfig(),
    )
    result = trainer.train(df, target_config, feature_columns)
    return {
        "label": label,
        "model_type": model_type.value,
        "model_id": result.model_id[:12],
        "metrics": result.metrics,
        "feature_importance": result.feature_importance,
        "calibration": result.calibration,
        "execution_time_ms": result.execution_time_ms,
        "trainer": trainer,
        "result": result,
    }


def main():
    # ── 1. Load dataset ──────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Load Tennis Dataset")
    print("=" * 70)
    df = pd.read_csv(FIXTURE)
    print(f"Loaded {len(df)} events, {len(df.columns)} columns\n")

    # ── 2. Column analysis ───────────────────────────────────────
    print("=" * 70)
    print("STEP 2: Column Analysis")
    print("=" * 70)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)

    print(f"{'Column':20s} {'Type':12s} {'Missing':>8s} {'Unique':>8s} {'Samples'}")
    print("-" * 70)
    for col in preview.columns:
        samples = str(col.sample_values[:2])
        print(f"{col.name:20s} {col.dtype:12s} {col.missing_pct:>7.1f}% {col.unique_count:>8d} {samples}")

    targets = analyzer.suggest_target(preview.columns)
    print(f"\nSuggested targets: {targets}")

    features = analyzer.suggest_features(preview.columns, target_col="result")
    print(f"Suggested features: {[f.column for f in features]}")
    if preview.warnings:
        print(f"Warnings: {preview.warnings}")
    print()

    # ── 3. Configure target and features ─────────────────────────
    target_config = TargetConfig(
        column="result",
        type=TargetType.binary_classification,
        positive_class="p1",
    )
    feature_columns = [
        "p1_rank", "p2_rank", "rank_gap",
        "p1_back_price", "p2_back_price",
        "p1_elo", "p2_elo", "elo_diff",
    ]

    # ── 4. Train multiple models ─────────────────────────────────
    print("=" * 70)
    print("STEP 3: Train Multiple Models")
    print("=" * 70)

    models_to_train = [
        (ModelType.logistic_regression, "Logistic Regression"),
        (ModelType.random_forest, "Random Forest"),
        (ModelType.gradient_boosting, "Gradient Boosting"),
        (ModelType.dummy_baseline, "Dummy Baseline"),
    ]

    results = []
    for model_type, label in models_to_train:
        print(f"\nTraining {label}...")
        r = train_and_evaluate(model_type, df, target_config, feature_columns, label)
        results.append(r)
        print(f"  Accuracy:  {r['metrics'].get('accuracy', 0):.4f}")
        print(f"  ROC AUC:   {r['metrics'].get('roc_auc', 'N/A')}")
        print(f"  F1:        {r['metrics'].get('f1', 0):.4f}")
        print(f"  Time:      {r['execution_time_ms']:.0f}ms")

    # ── 5. Compare results ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Model Comparison")
    print("=" * 70)

    header = f"{'Model':25s} {'Accuracy':>10s} {'ROC AUC':>10s} {'F1':>10s} {'Brier':>10s} {'Time':>8s}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x["metrics"].get("accuracy", 0), reverse=True):
        m = r["metrics"]
        print(
            f"{r['label']:25s} "
            f"{m.get('accuracy', 0):>10.4f} "
            f"{m.get('roc_auc', 'N/A'):>10} "
            f"{m.get('f1', 0):>10.4f} "
            f"{m.get('brier_score', 'N/A'):>10} "
            f"{r['execution_time_ms']:>7.0f}ms"
        )

    # ── 6. Export best model as strategy ─────────────────────────
    best = max(results, key=lambda x: x["metrics"].get("accuracy", 0))
    print(f"\n{'=' * 70}")
    print(f"STEP 5: Export Best Model ({best['label']}) as Strategy")
    print("=" * 70)

    converter = StrategyConverter()
    betting_config = BettingConversionConfig(
        probability_threshold=0.55,
        minimum_edge=0.03,
        minimum_odds=1.2,
        maximum_odds=10.0,
        staking_method="kelly",
    )
    strategy = converter.convert_to_strategy(
        model=best["trainer"].get_model(),
        feature_columns=feature_columns,
        betting_config=betting_config,
        model_metrics=best["metrics"],
    )
    print(f"Strategy type: {strategy['type']}")
    print(f"Filters: {len(strategy['filters'])}")
    print(f"Staking: {strategy['staking']['method']}")
    for f in strategy["filters"]:
        print(f"  - {f['type']}: {f['operator']} {f.get('value', f.get('min', ''))}")

    # ── 7. Run backtest ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 6: Backtest")
    print("=" * 70)

    trainer = best["trainer"]
    preprocessor = trainer.get_preprocessor()
    X = preprocessor.transform(df)
    y = df[target_config.column].map(lambda x: 1 if x == target_config.positive_class else 0)
    model = trainer.get_model()
    probabilities = model.predict_proba(X)[:, 1]

    bets_placed = 0
    wins = 0
    total_stake = 0
    total_profit = 0
    max_drawdown = 0
    peak = 0

    for i in range(len(df)):
        model_prob = probabilities[i]
        odds = df.iloc[i]["p1_back_price"]
        implied_prob = 1.0 / odds if odds > 0 else 0
        edge = model_prob - implied_prob

        if model_prob >= betting_config.probability_threshold and edge >= betting_config.minimum_edge:
            bets_placed += 1
            stake = 10.0
            total_stake += stake
            if y.iloc[i] == 1:
                wins += 1
                profit = stake * (odds - 1)
                total_profit += profit
            else:
                total_profit -= stake
            peak = max(peak, total_profit)
            drawdown = peak - total_profit
            max_drawdown = max(max_drawdown, drawdown)

    strike_rate = wins / bets_placed * 100 if bets_placed > 0 else 0
    roi = total_profit / total_stake * 100 if total_stake > 0 else 0

    print(f"Bets placed:   {bets_placed}")
    print(f"Wins:          {wins}")
    print(f"Strike rate:   {strike_rate:.1f}%")
    print(f"Total stake:   {total_stake:.2f}")
    print(f"Total profit:  {total_profit:.2f}")
    print(f"ROI:           {roi:.2f}%")
    print(f"Max drawdown:  {max_drawdown:.2f}")

    print(f"\n{'=' * 70}")
    print("DONE — Model Builder Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
