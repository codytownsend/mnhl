"""
Time-series cross-validation framework for SOG prediction models.

Provides walk-forward validation with proper temporal ordering
to prevent data leakage and accurately assess model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from src.utils.config import get_config
from src.validation.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesSplit:
    """
    Represents a single train/test split in time series.
    
    Attributes:
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date (after purge period)
        test_end: Test end date
        split_id: Unique identifier for this split
    """
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    split_id: int
    
    def __repr__(self) -> str:
        return (f"Split {self.split_id}: "
                f"Train[{self.train_start.date()} to {self.train_end.date()}] "
                f"Test[{self.test_start.date()} to {self.test_end.date()}]")


class TimeSeriesBacktester:
    """
    Walk-forward backtesting framework for time series models.
    
    Ensures proper temporal ordering and prevents look-ahead bias.
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'game_date'):
        """
        Initialize backtester.
        
        Args:
            data: Historical data with date column
            date_column: Name of date column
        """
        self.config = get_config()
        self.data = data.copy()
        self.date_column = date_column
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Sort by date
        self.data = self.data.sort_values(date_column).reset_index(drop=True)
        
        self.min_date = self.data[date_column].min()
        self.max_date = self.data[date_column].max()
        
        logger.info(f"Backtester initialized: {len(data)} records from "
                   f"{self.min_date.date()} to {self.max_date.date()}")
    
    def generate_splits(self, 
                       n_splits: Optional[int] = None,
                       test_size_days: Optional[int] = None,
                       gap_days: Optional[int] = None,
                       min_train_size_days: int = 90) -> List[TimeSeriesSplit]:
        """
        Generate time series splits for cross-validation.
        
        Uses expanding window: each split has more training data than the last,
        but test window size stays constant.
        
        Args:
            n_splits: Number of splits (uses config if None)
            test_size_days: Size of test window in days (uses config if None)
            gap_days: Gap between train and test (uses config if None)
            min_train_size_days: Minimum training period
            
        Returns:
            List of TimeSeriesSplit objects
        """
        # Use config defaults if not provided
        if n_splits is None:
            n_splits = self.config.validation.cv_n_splits
        if test_size_days is None:
            test_size_days = self.config.validation.cv_test_size_days
        if gap_days is None:
            gap_days = self.config.validation.cv_gap_days
        
        splits = []
        
        # Calculate split dates
        # Start with minimum training period, then create test windows
        current_train_end = self.min_date + timedelta(days=min_train_size_days)
        
        for split_id in range(n_splits):
            # Test period starts after gap
            test_start = current_train_end + timedelta(days=gap_days)
            test_end = test_start + timedelta(days=test_size_days)
            
            # Check if we have enough data
            if test_end > self.max_date:
                logger.warning(f"Not enough data for {n_splits} splits. Got {split_id} splits.")
                break
            
            split = TimeSeriesSplit(
                train_start=self.min_date,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
                split_id=split_id
            )
            
            splits.append(split)
            
            # Move train window forward for next split (expanding window)
            current_train_end = test_end
        
        logger.info(f"Generated {len(splits)} time series splits")
        for split in splits:
            logger.debug(str(split))
        
        return splits
    
    def get_split_data(self, split: TimeSeriesSplit) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and test data for a split.
        
        Args:
            split: TimeSeriesSplit object
            
        Returns:
            (train_data, test_data)
        """
        # Training data
        train_mask = (
            (self.data[self.date_column] >= split.train_start) &
            (self.data[self.date_column] <= split.train_end)
        )
        train_data = self.data[train_mask].copy()
        
        # Test data
        test_mask = (
            (self.data[self.date_column] >= split.test_start) &
            (self.data[self.date_column] <= split.test_end)
        )
        test_data = self.data[test_mask].copy()
        
        logger.debug(f"Split {split.split_id}: {len(train_data)} train, {len(test_data)} test")
        
        return train_data, test_data
    
    def cross_validate(self,
                      model_class: type,
                      feature_builder: Callable,
                      model_params: Optional[Dict] = None,
                      n_splits: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            model_class: Model class to instantiate
            feature_builder: Function to build features from raw data
            model_params: Parameters to pass to model constructor
            n_splits: Number of splits
            
        Returns:
            Dict with CV results
        """
        model_params = model_params or {}
        splits = self.generate_splits(n_splits=n_splits)
        
        cv_results = {
            'split_metrics': [],
            'predictions': [],
            'feature_importance': [],
        }
        
        calculator = MetricsCalculator()
        
        for split in splits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Cross-validation: {split}")
            logger.info('='*60)
            
            # Get data for this split
            train_data, test_data = self.get_split_data(split)
            
            # Build features
            logger.info("Building features...")
            X_train, y_train = feature_builder(train_data)
            X_test, y_test = feature_builder(test_data)
            
            # Train model
            logger.info("Training model...")
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict on test set
            logger.info("Generating predictions...")
            predictions = model.predict_distribution(X_test)
            
            # Evaluate
            predictions_df = predictions.copy()
            predictions_df['player_id'] = X_test['player_id']
            predictions_df['game_id'] = X_test['game_id']
            
            actuals_df = pd.DataFrame({
                'player_id': X_test['player_id'],
                'game_id': X_test['game_id'],
                'actual_shots': y_test
            })
            
            metrics = calculator.evaluate_predictions(predictions_df, actuals_df)
            
            # Store results
            split_result = {
                'split_id': split.split_id,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'test_start': split.test_start,
                'test_end': split.test_end,
                'metrics': metrics
            }
            
            cv_results['split_metrics'].append(split_result)
            
            # Store predictions
            predictions_df['split_id'] = split.split_id
            predictions_df['actual_shots'] = y_test
            cv_results['predictions'].append(predictions_df)
            
            # Store feature importance if available
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance('mu', top_n=20)
                importance['split_id'] = split.split_id
                cv_results['feature_importance'].append(importance)
            
            # Log split results
            logger.info(f"\nSplit {split.split_id} Results:")
            logger.info(f"  CRPS: {metrics['crps']:.4f}")
            logger.info(f"  Brier 2.5: {metrics.get('brier_2.5', 0):.4f}")
            logger.info(f"  Calibration Error: {metrics.get('calibration_error', 0):.4f}")
        
        # Aggregate results
        cv_results['summary'] = self._aggregate_cv_results(cv_results['split_metrics'])
        
        # Concatenate all predictions
        if cv_results['predictions']:
            cv_results['all_predictions'] = pd.concat(
                cv_results['predictions'], ignore_index=True
            )
        
        # Aggregate feature importance
        if cv_results['feature_importance']:
            cv_results['avg_feature_importance'] = self._aggregate_feature_importance(
                cv_results['feature_importance']
            )
        
        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("="*60)
        self._log_cv_summary(cv_results['summary'])
        
        return cv_results
    
    def _aggregate_cv_results(self, split_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics across CV splits.
        
        Args:
            split_metrics: List of metric dicts from each split
            
        Returns:
            Aggregated metrics
        """
        # Collect all metric values
        all_metrics = {}
        for split_result in split_metrics:
            for metric_name, value in split_result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate mean and std for each metric
        summary = {}
        for metric_name, values in all_metrics.items():
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        
        return summary
    
    def _aggregate_feature_importance(self, 
                                     importance_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate feature importance across splits.
        
        Args:
            importance_list: List of feature importance DataFrames
            
        Returns:
            Averaged feature importance
        """
        # Combine all importance scores
        all_importance = pd.concat(importance_list, ignore_index=True)
        
        # Average by feature
        avg_importance = all_importance.groupby('feature')['importance'].agg([
            ('mean_importance', 'mean'),
            ('std_importance', 'std'),
            ('min_importance', 'min'),
            ('max_importance', 'max')
        ]).reset_index()
        
        # Sort by mean importance
        avg_importance = avg_importance.sort_values('mean_importance', ascending=False)
        
        return avg_importance
    
    def _log_cv_summary(self, summary: Dict[str, float]) -> None:
        """Log CV summary statistics."""
        # Main metrics
        main_metrics = ['crps', 'mae', 'rmse', 'calibration_error']
        
        logger.info("\nKey Metrics:")
        for metric in main_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in summary:
                logger.info(f"  {metric}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")
        
        # Brier scores
        brier_metrics = [k for k in summary.keys() if k.startswith('brier_') and k.endswith('_mean')]
        if brier_metrics:
            logger.info("\nBrier Scores:")
            for metric in sorted(brier_metrics):
                base_name = metric.replace('_mean', '')
                std_key = metric.replace('_mean', '_std')
                line = base_name.replace('brier_', '')
                logger.info(f"  {line}: {summary[metric]:.4f} ± {summary[std_key]:.4f}")
        
        # Coverage
        coverage_metrics = [k for k in summary.keys() if k.startswith('coverage_') and k.endswith('_mean')]
        if coverage_metrics:
            logger.info("\nCoverage:")
            for metric in sorted(coverage_metrics):
                std_key = metric.replace('_mean', '_std')
                level = metric.replace('coverage_', '').replace('_mean', '')
                logger.info(f"  {level}%: {summary[metric]:.3f} ± {summary[std_key]:.3f}")
    
    def compare_models(self,
                      models_config: Dict[str, Dict],
                      feature_builder: Callable,
                      n_splits: Optional[int] = None) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Args:
            models_config: Dict of model_name -> {'class': ModelClass, 'params': {...}}
            feature_builder: Function to build features
            n_splits: Number of splits
            
        Returns:
            DataFrame comparing model performance
        """
        results = []
        
        for model_name, config in models_config.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_name}")
            logger.info('='*60)
            
            model_class = config['class']
            model_params = config.get('params', {})
            
            cv_results = self.cross_validate(
                model_class=model_class,
                feature_builder=feature_builder,
                model_params=model_params,
                n_splits=n_splits
            )
            
            # Extract summary metrics
            summary = cv_results['summary']
            summary['model'] = model_name
            results.append(summary)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns
        metric_cols = [c for c in comparison_df.columns if c != 'model']
        comparison_df = comparison_df[['model'] + metric_cols]
        
        # Log comparison
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        # Show key metrics
        key_metrics = ['crps_mean', 'brier_2.5_mean', 'calibration_error_mean']
        for col in key_metrics:
            if col in comparison_df.columns:
                logger.info(f"\n{col}:")
                for _, row in comparison_df.iterrows():
                    logger.info(f"  {row['model']:<30} {row[col]:>10.4f}")
        
        return comparison_df
    
    def rolling_window_validation(self,
                                  model_class: type,
                                  feature_builder: Callable,
                                  train_window_days: int = 180,
                                  test_window_days: int = 30,
                                  step_days: int = 30,
                                  model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform rolling window validation.
        
        Unlike expanding window, this keeps training window size constant
        and slides it forward. Useful for detecting model drift.
        
        Args:
            model_class: Model class
            feature_builder: Feature builder function
            train_window_days: Size of training window
            test_window_days: Size of test window
            step_days: How many days to step forward each iteration
            model_params: Model parameters
            
        Returns:
            Dict with validation results
        """
        model_params = model_params or {}
        gap_days = self.config.validation.cv_gap_days
        
        results = []
        current_train_start = self.min_date
        split_id = 0
        
        calculator = MetricsCalculator()
        
        while True:
            # Define windows
            train_end = current_train_start + timedelta(days=train_window_days)
            test_start = train_end + timedelta(days=gap_days)
            test_end = test_start + timedelta(days=test_window_days)
            
            # Check if we have enough data
            if test_end > self.max_date:
                break
            
            split = TimeSeriesSplit(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                split_id=split_id
            )
            
            logger.info(f"\nRolling window: {split}")
            
            # Get data
            train_data, test_data = self.get_split_data(split)
            
            # Build features
            X_train, y_train = feature_builder(train_data)
            X_test, y_test = feature_builder(test_data)
            
            # Train and predict
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            predictions = model.predict_distribution(X_test)
            
            # Evaluate
            predictions_df = predictions.copy()
            predictions_df['player_id'] = X_test['player_id']
            predictions_df['game_id'] = X_test['game_id']
            
            actuals_df = pd.DataFrame({
                'player_id': X_test['player_id'],
                'game_id': X_test['game_id'],
                'actual_shots': y_test
            })
            
            metrics = calculator.evaluate_predictions(predictions_df, actuals_df)
            
            # Store results
            results.append({
                'split_id': split_id,
                'test_start': test_start,
                'test_end': test_end,
                'crps': metrics['crps'],
                'brier_2.5': metrics.get('brier_2.5', np.nan),
                'calibration_error': metrics.get('calibration_error', np.nan),
            })
            
            # Move window forward
            current_train_start += timedelta(days=step_days)
            split_id += 1
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"\nCompleted {len(results)} rolling window iterations")
        logger.info(f"Average CRPS: {results_df['crps'].mean():.4f}")
        
        return {
            'results': results_df,
            'avg_crps': results_df['crps'].mean(),
            'crps_trend': results_df['crps'].values,
        }


def create_backtester(data: pd.DataFrame, date_column: str = 'game_date') -> TimeSeriesBacktester:
    """
    Factory function to create backtester.
    
    Args:
        data: Historical data
        date_column: Date column name
        
    Returns:
        TimeSeriesBacktester instance
    """
    return TimeSeriesBacktester(data, date_column)