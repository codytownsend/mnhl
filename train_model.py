"""
Main training script for NHL SOG prediction model.

Usage:
    python train_model.py --config config/model_config.yaml
    
This script:
1. Collects historical data
2. (Optional) Runs rolling time-series cross-validation diagnostics
3. Splits into train/val/test
4. Trains baseline models
5. Trains main LightGBM model
6. Calibrates predictive dispersion for intervals
7. Calibrates probabilities
8. Fits conformal interval adjustments
9. Evaluates and compares models
10. Saves artifacts
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.config import load_config, get_config
from src.data_pipeline.nhl_api import create_nhl_client
from src.data_pipeline.pipeline import (
    HistoricalDataCollector, 
    setup_logging
)
from src.data_pipeline.features import FeatureEngineer
from src.modeling.lgbm_model import LGBMNegativeBinomialModel
from src.modeling.calibration import create_calibrator, save_calibrator
from src.modeling.conformal import (
    ConformalIntervalCalibrator,
    save_conformal_calibrator
)
from src.validation.baselines import create_baseline_models, evaluate_baseline
from src.validation.metrics import MetricsCalculator
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def build_feature_matrix(feature_engineer: FeatureEngineer, df: pd.DataFrame) -> tuple:
    """
    Build model features and targets for the provided dataframe using the
    supplied feature engineer (which should be fitted on training data).
    """
    if df.empty:
        return pd.DataFrame(), np.array([])
    
    tasks = []
    for _, row in df.iterrows():
        tasks.append({
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'position': row['position'],
            'team_id': row['team_id'],
            'opponent_team_id': row['opponent_team_id'],
            'game_id': row['game_id'],
            'game_date': row['game_date'],
            'venue_name': row['venue_name'],
            'is_home': row['home_away'] == 'home',
        })
    
    features = feature_engineer.bulk_build_features(tasks)
    targets = df['shots'].values
    return features, targets


def collect_historical_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Collect or load historical data.
    
    Args:
        force_refresh: Force re-collection even if cached data exists
        
    Returns:
        DataFrame with historical player game statistics
    """
    config = get_config()
    cache_path = config.data.raw_data_dir / "historical_data.parquet"
    
    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached historical data from {cache_path}")
        
        df = pd.read_parquet(cache_path)

        # Deduplicate historical data
        original_len = len(df)
        df = df.drop_duplicates(subset=['player_id', 'game_id', 'game_date'])
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate records from historical data")

        return df
    
    logger.info("Collecting historical data from NHL API")
    collector = HistoricalDataCollector()
    
    # Use configurable start date
    start_date = datetime.fromisoformat(config.data.collection_start_date)
    end_date = datetime.now()
    
    logger.info(f"Collecting from {start_date.date()} to {end_date.date()}")
    
    # Collect data for training period
    historical_data = collector.collect_date_range(
        start_date=start_date,
        end_date=end_date
    )
    
    # Save to cache
    collector.save_to_disk(historical_data, "historical_data.parquet")
    
    return historical_data


def prepare_training_data(historical_data: pd.DataFrame) -> tuple:
    """
    Split data and prepare features.
    
    Args:
        historical_data: Raw historical data
        
    Returns:
        (train_features, train_targets, val_features, val_targets, test_features, test_targets, feature_engineer)
    """
    config = get_config()
    
    logger.info("Splitting data into train/val/test")
    
    # Sort by date
    historical_data = historical_data.sort_values('game_date').reset_index(drop=True)
    
    # Calculate split indices based on percentages
    n_samples = len(historical_data)
    train_end_idx = int(n_samples * config.validation.train_pct)
    val_end_idx = int(n_samples * (config.validation.train_pct + config.validation.val_pct))
    
    train_df = historical_data[:train_end_idx].copy()
    val_df = historical_data[train_end_idx:val_end_idx].copy()
    test_df = historical_data[val_end_idx:].copy()
    
    logger.info(f"Train: {len(train_df)} records ({train_df['game_date'].min().date()} to {train_df['game_date'].max().date()})")
    logger.info(f"Val:   {len(val_df)} records ({val_df['game_date'].min().date() if len(val_df) > 0 else 'N/A'} to {val_df['game_date'].max().date() if len(val_df) > 0 else 'N/A'})")
    logger.info(f"Test:  {len(test_df)} records ({test_df['game_date'].min().date() if len(test_df) > 0 else 'N/A'} to {test_df['game_date'].max().date() if len(test_df) > 0 else 'N/A'})")
    
    # Check minimum samples
    if len(train_df) < config.validation.min_train_samples:
        logger.warning(f"Training set has only {len(train_df)} samples (minimum: {config.validation.min_train_samples})")
    if len(val_df) < config.validation.min_val_samples:
        logger.warning(f"Validation set has only {len(val_df)} samples (minimum: {config.validation.min_val_samples})")
    if len(test_df) < config.validation.min_test_samples:
        logger.warning(f"Test set has only {len(test_df)} samples (minimum: {config.validation.min_test_samples})")
    
    # Initialize feature engineer with training data
    feature_engineer = FeatureEngineer(train_df)
    
    logger.info("Building features for train set")
    X_train, y_train = build_feature_matrix(feature_engineer, train_df)
    
    logger.info("Building features for validation set")
    X_val, y_val = build_feature_matrix(feature_engineer, val_df)
    
    logger.info("Building features for test set")
    X_test, y_test = build_feature_matrix(feature_engineer, test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer


def run_time_series_cv(historical_data: pd.DataFrame) -> None:
    """
    Perform rolling time-series cross-validation to assess model stability.
    """
    config = get_config()
    cv_splits = config.validation.cv_n_splits
    if cv_splits < 2:
        logger.info("Skipping time-series CV (cv_n_splits < 2)")
        return
    
    historical_data = historical_data.sort_values('game_date').reset_index(drop=True)
    n_samples = len(historical_data)
    if n_samples < cv_splits * 10:
        logger.info("Skipping time-series CV (insufficient samples)")
        return
    
    unique_dates = historical_data['game_date'].dt.normalize().unique()
    avg_rows_per_day = max(1, int(np.floor(n_samples / max(1, len(unique_dates)))))
    
    test_size = max(
        1,
        int(avg_rows_per_day * config.validation.cv_test_size_days)
    )
    gap = max(
        0,
        int(avg_rows_per_day * config.validation.cv_gap_days)
    )
    
    # Ensure test size is feasible
    max_test_size = n_samples // (cv_splits + 1)
    test_size = max(1, min(test_size, max_test_size))
    gap = min(gap, max(0, n_samples - test_size - 1))
    
    splitter = TimeSeriesSplit(
        n_splits=cv_splits,
        test_size=test_size,
        gap=gap
    )
    
    logger.info(
        f"Running time-series CV (splits={cv_splits}, test_size={test_size}, gap={gap})"
    )
    
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(historical_data), start=1):
        train_df = historical_data.iloc[train_idx].copy()
        val_df = historical_data.iloc[val_idx].copy()
        
        feature_engineer = FeatureEngineer(train_df)
        X_train, y_train = build_feature_matrix(feature_engineer, train_df)
        X_val, y_val = build_feature_matrix(feature_engineer, val_df)
        
        model = LGBMNegativeBinomialModel()
        model.fit(X_train, y_train, verbose=False)
        
        metrics = model.evaluate(X_val, y_val)
        fold_metrics.append(metrics)
        logger.info(
            f"CV Fold {fold_idx}/{cv_splits} - CRPS: {metrics['crps']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}"
        )
    
    if fold_metrics:
        avg_crps = np.mean([m['crps'] for m in fold_metrics])
        avg_mae = np.mean([m['mae'] for m in fold_metrics])
        avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
        logger.info(
            f"Time-series CV summary (avg over {len(fold_metrics)} folds): "
            f"CRPS={avg_crps:.4f}, MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}"
        )
def train_and_evaluate_baselines(X_train, y_train, X_test, y_test, train_df, test_df) -> dict:
    """
    Train and evaluate baseline models.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        train_df: Raw training DataFrame for baseline models
        test_df: Raw test DataFrame
        
    Returns:
        Dict of baseline results
    """
    if len(test_df) == 0:
        logger.warning("Test set is empty, skipping baseline evaluation")
        return {}
    
    logger.info("Training baseline models")
    
    baselines = create_baseline_models()
    baseline_results = {}
    
    for name, model in baselines.items():
        logger.info(f"Training {name}")
        model.fit(train_df)
        
        # Evaluate
        test_data = test_df.copy()
        test_data['player_id'] = X_test['player_id'].values
        
        metrics = evaluate_baseline(
            model, 
            test_data[['player_id', 'shots']],
            X_test
        )
        
        baseline_results[name] = metrics
        logger.info(f"{name} - CRPS: {metrics['crps']:.3f}")
    
    return baseline_results


def train_main_model(X_train, y_train, X_val, y_val) -> LGBMNegativeBinomialModel:
    """
    Train main LightGBM model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained model
    """
    logger.info("Training main LightGBM model")
    
    model = LGBMNegativeBinomialModel()
    history = model.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    return model


def calibrate_model(model, X_val, y_val):
    """
    Fit calibrator on validation set.
    
    Args:
        model: Trained model
        X_val, y_val: Validation data
        
    Returns:
        Fitted calibrator
    """
    logger.info("Fitting calibrator")
    
    # Get validation predictions
    val_predictions = model.predict_with_uncertainty_adjustment(X_val)
    
    # Create and fit calibrator
    calibrator = create_calibrator()
    calibrator.fit(val_predictions, y_val)
    
    # Evaluate calibration improvement
    eval_results = calibrator.evaluate_calibration(val_predictions, y_val)
    
    logger.info("Calibration results:")
    for line, metrics in eval_results.items():
        if 'calibrated' in line:
            logger.info(f"  {line}: cal_error={metrics['calibration_error']:.4f}, "
                       f"brier={metrics['brier_score']:.4f}")
    
    return calibrator


def evaluate_final_model(model, calibrator, interval_calibrator,
                        X_test, y_test, baseline_results):
    """
    Comprehensive evaluation on test set.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator
        interval_calibrator: Conformal interval calibrator
        X_test, y_test: Test data
        baseline_results: Results from baseline models
        
    Returns:
        Dict of evaluation results
    """
    logger.info("Evaluating final model on test set")
    
    # Get predictions
    test_predictions = model.predict_with_uncertainty_adjustment(X_test)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    
    predictions_df = test_predictions.copy()
    predictions_df['player_id'] = X_test['player_id']
    predictions_df['game_id'] = X_test['game_id']
    
    actuals_df = pd.DataFrame({
        'player_id': X_test['player_id'],
        'game_id': X_test['game_id'],
        'actual_shots': y_test
    })
    
    metrics = calculator.evaluate_predictions(predictions_df, actuals_df)
    
    if interval_calibrator is not None:
        interval_df = interval_calibrator.predict(test_predictions)
        for level in interval_calibrator.confidence_levels:
            key = int(level * 100)
            lower = interval_df[f'ci_{key}_lower'].values
            upper = interval_df[f'ci_{key}_upper'].values
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            sharpness = np.mean(upper - lower)
            metrics[f'coverage_{key}'] = coverage
            metrics[f'sharpness_{key}'] = sharpness
    
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("="*60)
    
    # Compare to baselines
    if baseline_results:
        logger.info("\nBaseline Comparison:")
        logger.info(f"{'Model':<30} {'CRPS':<10} {'Brier 2.5':<10}")
        logger.info("-" * 50)
        
        for name, baseline_metrics in baseline_results.items():
            logger.info(f"{name:<30} {baseline_metrics['crps']:>8.3f}  {baseline_metrics.get('brier_2.5', 0):>8.3f}")
        
        logger.info(f"{'LightGBM (ours)':<30} {metrics['crps']:>8.3f}  {metrics['brier_2.5']:>8.3f}")
    
    # Detailed metrics
    logger.info("\nDetailed Metrics:")
    logger.info(f"  CRPS: {metrics['crps']:.4f} ± {metrics.get('crps_std', 0):.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    
    logger.info("\nBrier Scores by Line:")
    for line in [1.5, 2.5, 3.5, 4.5]:
        if f'brier_{line}' in metrics:
            logger.info(f"  {line}: {metrics[f'brier_{line}']:.4f}")
    
    logger.info("\nCalibration:")
    logger.info(f"  Calibration Error: {metrics['calibration_error']:.4f}")
    
    logger.info("\nCoverage:")
    coverage_keys = sorted(
        [k for k in metrics.keys() if k.startswith('coverage_')],
        key=lambda k: int(k.split('_')[1])
    )
    for key in coverage_keys:
        conf_level = int(key.split('_')[1])
        actual_coverage = metrics[key]
        expected_coverage = conf_level / 100
        logger.info(f"  {conf_level}% CI: {actual_coverage:.3f} (expected: {expected_coverage:.3f})")
    
    logger.info("="*60 + "\n")
    
    # Check if we beat baselines
    if baseline_results:
        best_baseline_crps = min(m['crps'] for m in baseline_results.values())
        improvement = (best_baseline_crps - metrics['crps']) / best_baseline_crps * 100
        
        if metrics['crps'] < best_baseline_crps:
            logger.info(f"✓ Model beats best baseline by {improvement:.1f}%")
        else:
            logger.warning(f"✗ Model does not beat best baseline (worse by {-improvement:.1f}%)")
    
    return metrics


def save_model_artifacts(model, calibrator, interval_calibrator, feature_engineer):
    """
    Save all model artifacts.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator
        interval_calibrator: Conformal interval calibrator
        feature_engineer: Feature engineer (for metadata)
    """
    config = get_config()
    
    # Create versioned directory
    version = config.get_model_version_string()
    output_dir = config.data.model_artifacts_dir / version
    
    logger.info(f"Saving model artifacts to {output_dir}")
    
    # Save model
    model.save(output_dir)
    
    # Save calibrator
    save_calibrator(calibrator, output_dir / "calibrator.pkl")
    
    # Save interval calibrator
    save_conformal_calibrator(interval_calibrator, output_dir / "interval_calibrator.pkl")
    
    # Save feature importance
    importance = model.get_feature_importance('mu', top_n=50)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    
    logger.info(f"Model artifacts saved successfully")


def main():
    """Main training workflow."""
    parser = argparse.ArgumentParser(description="Train NHL SOG prediction model")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/model_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--refresh-data',
        action='store_true',
        help='Force refresh of historical data'
    )
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline model training'
    )
    parser.add_argument(
    '--save-test-data',
    action='store_true',
    help='Save test features and targets for diagnostics'
    )
    parser.add_argument(
        '--run-cv',
        action='store_true',
        help='Run time-series cross-validation diagnostics before training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    load_config(args.config)
    config = get_config()
    setup_logging()
    
    logger.info("="*60)
    logger.info("NHL SOG PREDICTION MODEL TRAINING")
    logger.info("="*60)
    
    # Step 1: Collect historical data
    logger.info("\n[1/10] Collecting historical data")
    historical_data = collect_historical_data(force_refresh=args.refresh_data)
    
    if args.run_cv:
        run_time_series_cv(historical_data)
    
    # Step 2: Prepare training data
    logger.info("\n[2/10] Preparing training data")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer = prepare_training_data(
        historical_data
    )

    # Save test data if requested (for diagnostics)
    if args.save_test_data:
        logger.info("Saving test data for diagnostics...")
        test_data_dir = Path("data/processed")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        X_test.to_parquet(test_data_dir / "test_features.parquet", index=False)
        logger.info(f"  Saved {len(X_test)} test features to {test_data_dir / 'test_features.parquet'}")
        
        # Save targets
        test_targets_df = pd.DataFrame({'shots': y_test})
        test_targets_df.to_parquet(test_data_dir / "test_targets.parquet", index=False)
        logger.info(f"  Saved {len(y_test)} test targets to {test_data_dir / 'test_targets.parquet'}")
    
    # Get raw dataframes for baselines
    historical_data_sorted = historical_data.sort_values('game_date')
    n_samples = len(historical_data_sorted)
    train_end_idx = int(n_samples * config.validation.train_pct)
    val_end_idx = int(n_samples * (config.validation.train_pct + config.validation.val_pct))
    
    train_df = historical_data_sorted[:train_end_idx]
    test_df = historical_data_sorted[val_end_idx:]
    
    # Step 3: Train and evaluate baselines
    baseline_results = {}
    if not args.skip_baselines:
        logger.info("\n[3/10] Training baseline models")
        baseline_results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test, train_df, test_df)
    else:
        logger.info("\n[3/10] Skipping baseline training")
    
    # Step 4: Train main model
    logger.info("\n[4/10] Training main model")
    model = train_main_model(X_train, y_train, X_val, y_val)
    
    # Step 5: Dispersion calibration for intervals
    logger.info("\n[5/10] Calibrating dispersion")
    model.calibrate_dispersion(X_val, y_val)
    
    # Step 6: Probability calibration
    logger.info("\n[6/10] Calibrating predictions")
    calibrator = calibrate_model(model, X_val, y_val)
    
    # Step 7: Conformal interval calibration
    logger.info("\n[7/10] Fitting conformal interval calibrator")
    val_dist = model.predict_with_uncertainty_adjustment(X_val)
    interval_calibrator = ConformalIntervalCalibrator(config.validation.confidence_levels)
    interval_calibrator.fit(val_dist, y_val)
    logger.info(f"Conformal adjustments: {interval_calibrator.get_metadata()}")
    
    # Step 8: Evaluate
    logger.info("\n[8/10] Evaluating on test set")
    final_metrics = evaluate_final_model(model, calibrator, interval_calibrator, X_test, y_test, baseline_results)
    
    # Step 9: Save artifacts
    logger.info("\n[9/10] Saving model artifacts")
    save_model_artifacts(model, calibrator, interval_calibrator, feature_engineer)
    
    logger.info("\n[10/10] Training workflow complete")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
