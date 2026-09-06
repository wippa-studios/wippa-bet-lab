"""
Wippa Bet Lab — Quickstart Example
===================================
Load a sample tennis dataset, train a Logistic Regression model,
evaluate performance, export as a betting strategy, and run a backtest.
"""

from pathlib import Path

import pandas as pd

from betlab.ml.column_analyzer import ColumnAnalyzer
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
from betlab.ml.trainer import ModelTrainer
from betlab.ml.evaluator import ModelEvaluator
from betlab.ml.strategy_converter import StrategyConverter

FIXTURE = Path("tests/fixtures/sample_tennis_dataset.csv")


def main():
    # ── 1. Load dataset ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Load Dataset")
    print("=" * 60)
    df = pd.read_csv(FIXTURE)
    print(f"Loaded {len(df)} events with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}\n")

    # ── 2. Analyze columns ───────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Column Analysis")
    print("=" * 60)
    analyzer = ColumnAnalyzer()
    preview = analyzer.analyze(df)

    for col in preview.columns:
        print(f"  {col.name:20s}  {col.dtype:12s}  missing={col.missing_pct}%  unique={col.unique_count}")
    print()

    # ── 3. Select target and features ────────────────────────────
    print("=" * 60)
    print("STEP 3: Target & Feature Selection")
    print("=" * 60)
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
    print(f"Target: {target_config.column} (positive_class={target_config.positive_class})")
    print(f"Features: {feature_columns}\n")

    # ── 4. Train Logistic Regression ─────────────────────────────
    print("=" * 60)
    print("STEP 4: Train Logistic Regression")
    print("=" * 60)
    trainer = ModelTrainer(
        model_config=ModelConfig(type=ModelType.logistic_regression),
        preprocessing_config=PreprocessingConfig(),
        validation_config=ValidationConfig(),
    )
    result = trainer.train(df, target_config, feature_columns)
    print(f"Model ID:      {result.model_id[:12]}...")
    print(f"Execution:     {result.execution_time_ms:.0f}ms")
    print(f"Warnings:      {result.warnings or 'none'}\n")

    # ── 5. Evaluate performance ──────────────────────────────────
    print("=" * 60)
    print("STEP 5: Performance Metrics")
    print("=" * 60)
    for metric, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {metric:25s}  {value:.4f}")
        else:
            print(f"  {metric:25s}  {value}")
    print()

    # ── 6. Feature importance ────────────────────────────────────
    print("=" * 60)
    print("STEP 6: Feature Importance")
    print("=" * 60)
    for item in result.feature_importance[:5]:
        print(f"  {item['feature']:25s}  {item['importance']:.4f}")
    print()

    # ── 7. Export as betting strategy ────────────────────────────
    print("=" * 60)
    print("STEP 7: Export as Betting Strategy")
    print("=" * 60)
    converter = StrategyConverter()
    betting_config = BettingConversionConfig(
        probability_threshold=0.55,
        minimum_edge=0.03,
        minimum_odds=1.2,
        maximum_odds=10.0,
        staking_method="kelly",
    )
    strategy = converter.convert_to_strategy(
        model=trainer.get_model(),
        feature_columns=feature_columns,
        betting_config=betting_config,
        model_metrics=result.metrics,
    )
    print(f"Strategy type: {strategy['type']}")
    print(f"Filters:       {len(strategy['filters'])}")
    print(f"Staking:       {strategy['staking']['method']}")
    for f in strategy["filters"]:
        print(f"  Filter: {f['type']}  {f['operator']}  {f.get('value', f.get('min', ''))}")
    print()

    # ── 8. Backtest simulation ───────────────────────────────────
    print("=" * 60)
    print("STEP 8: Simulated Backtest")
    print("=" * 60)
    preprocessor = trainer.get_preprocessor()
    X = preprocessor.transform(df)
    y = df[target_config.column].map(lambda x: 1 if x == target_config.positive_class else 0)
    model = trainer.get_model()
    probabilities = model.predict_proba(X)[:, 1]

    edge_values = []
    bets_placed = 0
    wins = 0
    total_stake = 0
    total_profit = 0

    for i, row in df.iterrows():
        model_prob = probabilities[i]
        odds = row["p1_back_price"]
        implied_prob = 1.0 / odds if odds > 0 else 0
        edge = model_prob - implied_prob

        if model_prob >= betting_config.probability_threshold and edge >= betting_config.minimum_edge:
            bets_placed += 1
            stake = 10.0  # flat stake for simplicity
            total_stake += stake
            if y.iloc[i] == 1:
                wins += 1
                profit = stake * (odds - 1)
                total_profit += profit
            else:
                total_profit -= stake
            edge_values.append(edge)

    strike_rate = wins / bets_placed * 100 if bets_placed > 0 else 0
    roi = total_profit / total_stake * 100 if total_stake > 0 else 0
    avg_edge = sum(edge_values) / len(edge_values) if edge_values else 0

    print(f"Bets placed:   {bets_placed}")
    print(f"Wins:          {wins}")
    print(f"Strike rate:   {strike_rate:.1f}%")
    print(f"Total stake:   {total_stake:.2f}")
    print(f"Total profit:  {total_profit:.2f}")
    print(f"ROI:           {roi:.2f}%")
    print(f"Avg edge:      {avg_edge:.4f}")
    print()

    print("=" * 60)
    print("DONE — Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
