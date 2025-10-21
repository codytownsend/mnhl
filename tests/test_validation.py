"""
Unit tests for validation and metrics components.

Tests CRPS, Brier scores, calibration metrics, and backtesting framework.

Run with:
    pytest tests/test_validation.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.validation.metrics import (
    calculate_crps, calculate_brier_score, calculate_log_loss,
    get_probability_over_line, calculate_calibration_error,
    calculate_coverage, calculate_sharpness, generate_pmf,
    calculate_prediction_interval, MetricsCalculator
)
from src.validation.backtester import (
    TimeSeriesBacktester, TimeSeriesSplit, create_backtester
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def setup_test_config():
    """Load configuration for tests."""
    load_config("config/model_config.yaml")


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)
    
    n = 100
    predictions = pd.DataFrame({
        'player_id': np.arange(n),
        'game_id': np.repeat(np.arange(10), 10),
        'mu': np.random.uniform(2, 4, n),
        'alpha': np.random.uniform(1, 2, n)
    })
    
    return predictions


@pytest.fixture
def sample_actuals():
    """Create sample actual results."""
    np.random.seed(42)
    
    n = 100
    actuals = pd.DataFrame({
        'player_id': np.arange(n),
        'game_id': np.repeat(np.arange(10), 10),
        'actual_shots': np.random.poisson(3, n)
    })
    
    return actuals


@pytest.fixture
def time_series_data():
    """Create time series data for backtesting."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    data = []
    for date in dates:
        for player_id in range(1, 6):
            data.append({
                'player_id': player_id,
                'game_date': date,
                'game_id': int(date.strftime('%Y%m%d')),
                'shots': np.random.poisson(3),
                'toi_seconds': np.random.normal(1200, 180),
            })
    
    return pd.DataFrame(data)


class TestMetricsCalculations:
    """Tests for individual metric calculations."""
    
    def test_calculate_crps_perfect(self):
        """Test CRPS with perfect prediction."""
        actual = 3
        mu = 3.0
        alpha = 1.0
        
        crps = calculate_crps(actual, mu, alpha)
        
        # Perfect prediction should have low CRPS
        assert crps >= 0
        assert crps < 1.0
    
    def test_calculate_crps_poor(self):
        """Test CRPS with poor prediction."""
        actual = 3
        mu = 10.0  # Way off
        alpha = 1.0
        
        crps = calculate_crps(actual, mu, alpha)
        
        # Poor prediction should have higher CRPS
        assert crps > 2.0
    
    def test_calculate_brier_score(self):
        """Test Brier score calculation."""
        # Perfect prediction
        brier_perfect = calculate_brier_score(actual_over=1, predicted_prob=1.0)
        assert brier_perfect == 0.0
        
        # Worst prediction
        brier_worst = calculate_brier_score(actual_over=1, predicted_prob=0.0)
        assert brier_worst == 1.0
        
        # Intermediate
        brier_mid = calculate_brier_score(actual_over=1, predicted_prob=0.7)
        assert 0 < brier_mid < 1
    
    def test_calculate_log_loss(self):
        """Test log loss calculation."""
        # Good prediction
        log_loss_good = calculate_log_loss(actual_over=1, predicted_prob=0.9)
        
        # Poor prediction
        log_loss_poor = calculate_log_loss(actual_over=1, predicted_prob=0.1)
        
        assert log_loss_good < log_loss_poor
        assert log_loss_good > 0
    
    def test_get_probability_over_line(self):
        """Test probability calculation for Over line."""
        mu = 3.0
        alpha = 1.5
        
        # P(X > 2.5) should be > 0.5 when mu = 3
        p_over_2_5 = get_probability_over_line(mu, alpha, 2.5)
        
        assert 0 <= p_over_2_5 <= 1
        assert p_over_2_5 > 0.5
        
        # P(X > 5.5) should be lower
        p_over_5_5 = get_probability_over_line(mu, alpha, 5.5)
        
        assert p_over_5_5 < p_over_2_5
    
    def test_calculate_calibration_error(self):
        """Test calibration error calculation."""
        np.random.seed(42)
        
        # Perfectly calibrated
        predictions = np.linspace(0.1, 0.9, 100)
        actuals = (np.random.random(100) < predictions).astype(int)
        
        cal_error, pred_probs, true_probs = calculate_calibration_error(
            predictions.tolist(), actuals.tolist(), n_bins=5
        )
        
        assert cal_error >= 0
        assert len(pred_probs) <= 5
        assert len(true_probs) <= 5
    
    def test_calculate_coverage(self):
        """Test coverage calculation."""
        np.random.seed(42)
        
        # Generate predictions where intervals should cover ~80%
        n = 100
        predictions = [(3.0, 1.5) for _ in range(n)]
        actuals = np.random.poisson(3, n)
        
        coverage = calculate_coverage(predictions, actuals.tolist(), confidence_level=0.8)
        
        assert 0 <= coverage <= 1
        # Should be roughly 0.8, but allow variance
        assert 0.5 < coverage < 0.95
    
    def test_calculate_sharpness(self):
        """Test sharpness calculation."""
        # Narrow intervals (sharp)
        predictions_sharp = [(3.0, 0.5) for _ in range(10)]
        sharpness_sharp = calculate_sharpness(predictions_sharp, confidence_level=0.8)
        
        # Wide intervals (not sharp)
        predictions_wide = [(3.0, 3.0) for _ in range(10)]
        sharpness_wide = calculate_sharpness(predictions_wide, confidence_level=0.8)
        
        assert sharpness_sharp < sharpness_wide
        assert sharpness_sharp > 0
    
    def test_generate_pmf(self):
        """Test PMF generation."""
        mu = 3.0
        alpha = 1.5
        
        pmf = generate_pmf(mu, alpha, max_k=10)
        
        assert isinstance(pmf, dict)
        assert len(pmf) == 11  # 0 to 10
        
        # Probabilities should sum to approximately 1
        total_prob = sum(pmf.values())
        assert 0.95 < total_prob <= 1.0
        
        # All probabilities should be valid
        for k, prob in pmf.items():
            assert 0 <= prob <= 1
    
    def test_calculate_prediction_interval(self):
        """Test prediction interval calculation."""
        mu = 3.0
        alpha = 1.5
        
        lower, upper = calculate_prediction_interval(mu, alpha, confidence_level=0.8)
        
        assert isinstance(lower, int)
        assert isinstance(upper, int)
        assert upper >= lower
        assert lower >= 0


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    def test_initialization(self, setup_test_config):
        """Test calculator initializes."""
        calculator = MetricsCalculator()
        assert calculator.config is not None
    
    def test_evaluate_predictions(self, setup_test_config, sample_predictions, sample_actuals):
        """Test comprehensive prediction evaluation."""
        calculator = MetricsCalculator()
        
        metrics = calculator.evaluate_predictions(sample_predictions, sample_actuals)
        
        # Check all expected metrics are present
        assert 'crps' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'calibration_error' in metrics
        
        # Check Brier scores
        assert 'brier_1.5' in metrics or 'brier_2.5' in metrics
        
        # Check coverage
        assert any(k.startswith('coverage_') for k in metrics.keys())
        
        # All metrics should be positive
        for key, value in metrics.items():
            if key.startswith('coverage_'):
                assert 0 <= value <= 1
            else:
                assert value >= 0
    
    def test_evaluate_by_subgroup(self, setup_test_config, sample_predictions, sample_actuals):
        """Test subgroup evaluation."""
        # Add a subgroup column
        sample_predictions['position'] = ['C', 'RW'] * 50
        sample_actuals['position'] = ['C', 'RW'] * 50
        
        calculator = MetricsCalculator()
        
        subgroup_metrics = calculator.evaluate_by_subgroup(
            sample_predictions, sample_actuals, subgroup_col='position'
        )
        
        assert isinstance(subgroup_metrics, dict)
        # Should have metrics for each position if enough samples
        assert len(subgroup_metrics) >= 1
        
        for subgroup, metrics in subgroup_metrics.items():
            assert 'crps' in metrics
    
    def test_generate_reliability_diagram_data(self, setup_test_config, sample_predictions, sample_actuals):
        """Test reliability diagram data generation."""
        calculator = MetricsCalculator()
        
        reliability_data = calculator.generate_reliability_diagram_data(
            sample_predictions, sample_actuals, line=2.5, n_bins=5
        )
        
        assert isinstance(reliability_data, pd.DataFrame)
        assert 'predicted_prob' in reliability_data.columns
        assert 'true_prob' in reliability_data.columns
        assert 'count' in reliability_data.columns
        assert len(reliability_data) <= 5
    
    def test_compare_models(self, setup_test_config, sample_predictions, sample_actuals):
        """Test model comparison."""
        calculator = MetricsCalculator()
        
        # Create second set of predictions (slightly different)
        predictions2 = sample_predictions.copy()
        predictions2['mu'] = predictions2['mu'] * 1.1
        
        model_predictions = {
            'model1': sample_predictions,
            'model2': predictions2
        }
        
        comparison = calculator.compare_models(model_predictions, sample_actuals)
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'crps' in comparison.columns


class TestTimeSeriesBacktester:
    """Tests for time series backtesting framework."""
    
    def test_initialization(self, setup_test_config, time_series_data):
        """Test backtester initializes."""
        backtester = TimeSeriesBacktester(time_series_data, date_column='game_date')
        
        assert backtester.data is not None
        assert len(backtester.data) > 0
        assert backtester.min_date is not None
        assert backtester.max_date is not None
    
    def test_generate_splits(self, setup_test_config, time_series_data):
        """Test split generation."""
        backtester = TimeSeriesBacktester(time_series_data, date_column='game_date')
        
        splits = backtester.generate_splits(n_splits=3, test_size_days=30, gap_days=7)
        
        assert len(splits) > 0
        assert all(isinstance(split, TimeSeriesSplit) for split in splits)
        
        # Check temporal ordering
        for split in splits:
            assert split.train_start < split.train_end
            assert split.train_end < split.test_start
            assert split.test_start < split.test_end
    
    def test_get_split_data(self, setup_test_config, time_series_data):
        """Test retrieving data for a split."""
        backtester = TimeSeriesBacktester(time_series_data, date_column='game_date')
        
        splits = backtester.generate_splits(n_splits=2, test_size_days=30, gap_days=7)
        
        train_data, test_data = backtester.get_split_data(splits[0])
        
        assert len(train_data) > 0
        assert len(test_data) > 0
        
        # Verify no overlap
        train_dates = set(train_data['game_date'])
        test_dates = set(test_data['game_date'])
        assert len(train_dates.intersection(test_dates)) == 0
        
        # Verify chronological order
        assert train_data['game_date'].max() < test_data['game_date'].min()
    
    def test_expanding_window(self, setup_test_config, time_series_data):
        """Test expanding window validation."""
        backtester = TimeSeriesBacktester(time_series_data, date_column='game_date')
        
        splits = backtester.generate_splits(n_splits=3, test_size_days=30, gap_days=7)
        
        # Training window should expand
        train_sizes = []
        for split in splits:
            train_data, _ = backtester.get_split_data(split)
            train_sizes.append(len(train_data))
        
        # Each split should have more training data than the last
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1]
    
    def test_no_data_leakage(self, setup_test_config, time_series_data):
        """Test that splits don't leak future data."""
        backtester = TimeSeriesBacktester(time_series_data, date_column='game_date')
        
        splits = backtester.generate_splits(n_splits=2, test_size_days=30, gap_days=7)
        
        for split in splits:
            train_data, test_data = backtester.get_split_data(split)
            
            # All training dates should be before test dates
            assert train_data['game_date'].max() < test_data['game_date'].min()
            
            # Gap should be respected
            gap = (test_data['game_date'].min() - train_data['game_date'].max()).days
            assert gap >= 7
    
    def test_cross_validate(self, setup_test_config, time_series_data):
        """Test cross-validation workflow."""
        backtester = TimeSeriesBacktester(time_series_data, date_column='game_date')
        
        # Mock model and feature builder
        class MockModel:
            def fit(self, X, y):
                pass
            
            def predict_distribution(self, X):
                n = len(X)
                return pd.DataFrame({
                    'mu': np.full(n, 3.0),
                    'alpha': np.full(n, 1.5)
                })
        
        def mock_feature_builder(df):
            X = df[['player_id', 'game_id']].copy()
            y = df['shots'].values
            return X, y
        
        # Run CV with only 2 splits for speed
        cv_results = backtester.cross_validate(
            model_class=MockModel,
            feature_builder=mock_feature_builder,
            n_splits=2
        )
        
        assert 'split_metrics' in cv_results
        assert 'summary' in cv_results
        assert len(cv_results['split_metrics']) == 2
        
        # Check summary has aggregated metrics
        assert 'crps_mean' in cv_results['summary']
        assert 'crps_std' in cv_results['summary']
    
    def test_create_backtester_factory(self, setup_test_config, time_series_data):
        """Test factory function."""
        backtester = create_backtester(time_series_data, date_column='game_date')
        
        assert isinstance(backtester, TimeSeriesBacktester)
        assert backtester.data is not None


class TestMetricsEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_crps_with_extreme_values(self):
        """Test CRPS with extreme parameter values."""
        # Very high mu
        crps_high = calculate_crps(actual=3, mu=100.0, alpha=1.0)
        assert crps_high > 0
        
        # Very low mu
        crps_low = calculate_crps(actual=3, mu=0.1, alpha=1.0)
        assert crps_low > 0
    
    def test_brier_with_edge_probabilities(self):
        """Test Brier score with edge case probabilities."""
        # Probability near 0
        brier_near_zero = calculate_brier_score(1, 0.001)
        assert brier_near_zero > 0
        
        # Probability near 1
        brier_near_one = calculate_brier_score(1, 0.999)
        assert brier_near_one >= 0
        assert brier_near_one < 0.01
    
    def test_empty_predictions(self, setup_test_config):
        """Test metrics calculator with empty predictions."""
        calculator = MetricsCalculator()
        
        empty_preds = pd.DataFrame(columns=['player_id', 'game_id', 'mu', 'alpha'])
        empty_actuals = pd.DataFrame(columns=['player_id', 'game_id', 'actual_shots'])
        
        metrics = calculator.evaluate_predictions(empty_preds, empty_actuals)
        
        # Should return empty dict or handle gracefully
        assert isinstance(metrics, dict)
    
    def test_mismatched_predictions_actuals(self, setup_test_config):
        """Test with mismatched predictions and actuals."""
        calculator = MetricsCalculator()
        
        predictions = pd.DataFrame({
            'player_id': [1, 2, 3],
            'game_id': [100, 100, 100],
            'mu': [3.0, 3.0, 3.0],
            'alpha': [1.5, 1.5, 1.5]
        })
        
        # Actuals have different players
        actuals = pd.DataFrame({
            'player_id': [4, 5, 6],
            'game_id': [100, 100, 100],
            'actual_shots': [2, 3, 4]
        })
        
        metrics = calculator.evaluate_predictions(predictions, actuals)
        
        # Should handle gracefully (empty result or appropriate error)
        assert isinstance(metrics, dict)


class TestTimeSeriesSplit:
    """Tests for TimeSeriesSplit dataclass."""
    
    def test_time_series_split_creation(self):
        """Test creating TimeSeriesSplit."""
        split = TimeSeriesSplit(
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 7),
            test_end=datetime(2024, 7, 31),
            split_id=0
        )
        
        assert split.split_id == 0
        assert split.train_start < split.train_end
        assert split.train_end < split.test_start
    
    def test_time_series_split_repr(self):
        """Test string representation."""
        split = TimeSeriesSplit(
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 7),
            test_end=datetime(2024, 7, 31),
            split_id=0
        )
        
        repr_str = repr(split)
        
        assert 'Split 0' in repr_str
        assert 'Train' in repr_str
        assert 'Test' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])