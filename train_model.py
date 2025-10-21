"""
Main training script for NHL SOG prediction model.

Usage:
    python train_model.py --config config/model_config.yaml
    
This script:
1. Collects historical data
2. Splits into train/val/test
3. Trains baseline models
4. Trains main LightGBM model
5. Calibrates predictions
6. Evaluates and compares models
7. Saves artifacts
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
from src.validation.baselines import create_baseline_models, evaluate_baseline
from src.validation.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


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
        return pd.read_parquet(cache_path)
    
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
    
    # Build features for each split
    def prepare_features(df):
        """Prepare features from raw data."""
        if df.empty:
            return pd.DataFrame(), np.array([])
            
        # Build prediction tasks
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
        
        # Generate features
        features = feature_engineer.bulk_build_features(tasks)
        targets = df['shots'].values
        
        return features, targets
    
    logger.info("Building features for train set")
    X_train, y_train = prepare_features(train_df)
    
    logger.info("Building features for validation set")
    X_val, y_val = prepare_features(val_df)
    
    logger.info("Building features for test set")
    X_test, y_test = prepare_features(test_df)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer


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
    val_predictions = model.predict_distribution(X_val)
    
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


def evaluate_final_model(model, calibrator, X_test, y_test, baseline_results):
    """
    Comprehensive evaluation on test set.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator
        X_test, y_test: Test data
        baseline_results: Results from baseline models
        
    Returns:
        Dict of evaluation results
    """
    logger.info("Evaluating final model on test set")
    
    # Get predictions
    test_predictions = model.predict_distribution(X_test)
    
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
    for conf_level in [50, 80, 90]:
        if f'coverage_{conf_level}' in metrics:
            actual_coverage = metrics[f'coverage_{conf_level}']
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


def save_model_artifacts(model, calibrator, feature_engineer):
    """
    Save all model artifacts.
    
    Args:
        model: Trained model
        calibrator: Fitted calibrator
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
    
    args = parser.parse_args()
    
    # Load configuration
    load_config(args.config)
    config = get_config()
    setup_logging()
    
    logger.info("="*60)
    logger.info("NHL SOG PREDICTION MODEL TRAINING")
    logger.info("="*60)
    
    # Step 1: Collect historical data
    logger.info("\n[1/7] Collecting historical data")
    historical_data = collect_historical_data(force_refresh=args.refresh_data)
    
    # Step 2: Prepare training data
    logger.info("\n[2/7] Preparing training data")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_engineer = prepare_training_data(
        historical_data
    )
    
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
        logger.info("\n[3/7] Training baseline models")
        baseline_results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test, train_df, test_df)
    else:
        logger.info("\n[3/7] Skipping baseline training")
    
    # Step 4: Train main model
    logger.info("\n[4/7] Training main model")
    model = train_main_model(X_train, y_train, X_val, y_val)
    
    # Step 5: Calibrate
    logger.info("\n[5/7] Calibrating predictions")
    calibrator = calibrate_model(model, X_val, y_val)
    
    # Step 6: Evaluate
    logger.info("\n[6/7] Evaluating on test set")
    final_metrics = evaluate_final_model(model, calibrator, X_test, y_test, baseline_results)
    
    # Step 7: Save artifacts
    logger.info("\n[7/7] Saving model artifacts")
    save_model_artifacts(model, calibrator, feature_engineer)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
