"""
Unit tests for prediction models.

Tests LightGBM model, calibration, and baseline models.

Run with:
    pytest tests/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from src.modeling.lgbm_model import LGBMNegativeBinomialModel
from src.modeling.calibration import (
    IsotonicCalibrator, PlattCalibrator, BetaCalibrator,
    MultiLineCalibrator, DistributionCalibrator, create_calibrator
)
from src.validation.baselines import (
    SeasonMeanBaseline, OpponentAdjustedBaseline, EWMABaseline,
    TOIAdjustedEWMA, create_baseline_models
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def setup_test_config():
    """Load configuration for tests."""
    load_config("config/model_config.yaml")


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    data = {
        'player_id': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'game_id': np.arange(n_samples),
        'toi_per_game_l10': np.random.normal(18, 3, n_samples),
        'shots_per_game_l10': np.random.normal(2.5, 0.8, n_samples),
        'icf_per_60_l10': np.random.normal(12, 3, n_samples),
        'isf_per_60_l10': np.random.normal(8, 2, n_samples),
        'opponent_sa_per_60': np.random.normal(30, 5, n_samples),
        'home_away': np.random.choice([0, 1], n_samples),
        'pp_unit': np.random.choice([0, 1, 2], n_samples),
        'is_forward': np.random.choice([0, 1], n_samples),
        'is_defense': np.random.choice([0, 1], n_samples),
        'rest_days': np.random.choice([0, 1, 2, 3], n_samples),
        'games_played': np.random.randint(5, 60, n_samples),
        'lineup_confidence': np.random.uniform(0.6, 1.0, n_samples),
        'projected_toi': np.random.normal(18, 3, n_samples),
        'projected_toi_std': np.random.uniform(1, 3, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Generate targets (SOG) based on features
    base_rate = 2.5
    toi_effect = 0.1 * (X['toi_per_game_l10'] - 18)
    shots_effect = 0.5 * (X['shots_per_game_l10'] - 2.5)
    noise = np.random.normal(0, 1, n_samples)
    
    lambda_param = np.maximum(base_rate + toi_effect + shots_effect + noise, 0.5)
    y = np.random.poisson(lambda_param)
    
    return X, y


@pytest.fixture
def sample_historical_for_baselines():
    """Create sample data for baseline models."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-10-01', '2024-11-30', freq='D')
    player_ids = [1, 2, 3, 4, 5]
    
    data = []
    for player_id in player_ids:
        for date in dates:
            data.append({
                'player_id': player_id,
                'game_date': date,
                'shots': np.random.poisson(3),
                'toi_seconds': np.random.normal(1200, 180),
                'opponent_team_id': np.random.choice([10, 11, 12, 13]),
            })
    
    return pd.DataFrame(data)


class TestLGBMNegativeBinomialModel:
    """Tests for LightGBM Negative Binomial model."""
    
    def test_initialization(self, setup_test_config):
        """Test model initializes correctly."""
        model = LGBMNegativeBinomialModel()
        
        assert model.mu_model is None
        assert model.alpha_model is None
        assert model.feature_names is None
    
    def test_fit_basic(self, setup_test_config, sample_training_data):
        """Test basic model training."""
        X, y = sample_training_data
        
        # Split into train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LGBMNegativeBinomialModel()
        history = model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        assert model.mu_model is not None
        assert model.alpha_model is not None
        assert model.feature_names is not None
        assert len(model.feature_names) > 0
        assert isinstance(history, dict)
    
    def test_predict_distribution(self, setup_test_config, sample_training_data):
        """Test distribution prediction."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        predictions = model.predict_distribution(X[400:])
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'mu' in predictions.columns
        assert 'alpha' in predictions.columns
        assert len(predictions) == 100
        
        # Check predictions are in reasonable range
        assert (predictions['mu'] > 0).all()
        assert (predictions['alpha'] > 0).all()
    
    def test_predict_probabilities(self, setup_test_config, sample_training_data):
        """Test probability prediction for common lines."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        probs = model.predict_probabilities(X[400:])
        
        assert isinstance(probs, pd.DataFrame)
        assert 'p_over_1.5' in probs.columns
        assert 'p_over_2.5' in probs.columns
        
        # Probabilities should be between 0 and 1
        for col in probs.columns:
            assert (probs[col] >= 0).all()
            assert (probs[col] <= 1).all()
    
    def test_predict_intervals(self, setup_test_config, sample_training_data):
        """Test confidence interval prediction."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        intervals = model.predict_intervals(X[400:], confidence_level=0.8)
        
        assert isinstance(intervals, pd.DataFrame)
        assert 'lower' in intervals.columns
        assert 'upper' in intervals.columns
        
        # Upper should be >= lower
        assert (intervals['upper'] >= intervals['lower']).all()
    
    def test_evaluate(self, setup_test_config, sample_training_data):
        """Test model evaluation."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        metrics = model.evaluate(X[400:], y[400:])
        
        assert 'crps' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['crps'] > 0
        assert metrics['mae'] > 0
    
    def test_feature_importance(self, setup_test_config, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        importance_mu = model.get_feature_importance('mu', top_n=10)
        importance_alpha = model.get_feature_importance('alpha', top_n=10)
        
        assert isinstance(importance_mu, pd.DataFrame)
        assert isinstance(importance_alpha, pd.DataFrame)
        assert 'feature' in importance_mu.columns
        assert 'importance' in importance_mu.columns
        assert len(importance_mu) <= 10
    
    def test_save_load(self, setup_test_config, sample_training_data):
        """Test model persistence."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        # Predict before saving
        pred_before = model.predict_distribution(X[400:450])
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            model.save(save_path)
            
            # Load into new model
            new_model = LGBMNegativeBinomialModel()
            new_model.load(save_path)
            
            # Predict after loading
            pred_after = new_model.predict_distribution(X[400:450])
            
            # Predictions should be identical
            pd.testing.assert_frame_equal(pred_before, pred_after)
    
    def test_uncertainty_inflation(self, setup_test_config, sample_training_data):
        """Test uncertainty inflation for low confidence."""
        X, y = sample_training_data
        
        # Modify some samples to have low confidence
        X_test = X[400:].copy()
        X_test.loc[X_test.index[0], 'games_played'] = 3
        X_test.loc[X_test.index[1], 'lineup_confidence'] = 0.4
        X_test.loc[X_test.index[2], 'projected_toi_std'] = 5.0
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        # Predict without adjustment
        pred_normal = model.predict_distribution(X_test)
        
        # Predict with adjustment
        pred_adjusted = model.predict_with_uncertainty_adjustment(X_test)
        
        # Alpha should be higher for low-confidence samples
        assert pred_adjusted.loc[pred_adjusted.index[0], 'alpha'] >= pred_normal.loc[pred_normal.index[0], 'alpha']
        assert pred_adjusted.loc[pred_adjusted.index[1], 'alpha'] >= pred_normal.loc[pred_normal.index[1], 'alpha']
        assert pred_adjusted.loc[pred_adjusted.index[2], 'alpha'] >= pred_normal.loc[pred_normal.index[2], 'alpha']


class TestCalibration:
    """Tests for calibration methods."""
    
    def test_isotonic_calibrator(self, setup_test_config):
        """Test isotonic regression calibration."""
        np.random.seed(42)
        
        # Generate predictions that are slightly off
        predictions = np.random.uniform(0.2, 0.8, 200)
        # Actual outcomes with bias
        actuals = (predictions + np.random.normal(0, 0.1, 200) > 0.5).astype(int)
        
        calibrator = IsotonicCalibrator()
        calibrator.fit(predictions, actuals)
        
        calibrated = calibrator.transform(predictions)
        
        assert len(calibrated) == len(predictions)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()
    
    def test_platt_calibrator(self, setup_test_config):
        """Test Platt scaling."""
        np.random.seed(42)
        
        predictions = np.random.uniform(0.2, 0.8, 200)
        actuals = (predictions > 0.5).astype(int)
        
        calibrator = PlattCalibrator()
        calibrator.fit(predictions, actuals)
        
        calibrated = calibrator.transform(predictions)
        
        assert len(calibrated) == len(predictions)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()
    
    def test_beta_calibrator(self, setup_test_config):
        """Test beta calibration."""
        np.random.seed(42)
        
        predictions = np.random.uniform(0.2, 0.8, 200)
        actuals = (predictions > 0.5).astype(int)
        
        calibrator = BetaCalibrator()
        calibrator.fit(predictions, actuals)
        
        calibrated = calibrator.transform(predictions)
        
        assert len(calibrated) == len(predictions)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()
    
    def test_multi_line_calibrator(self, setup_test_config):
        """Test calibration for multiple lines."""
        np.random.seed(42)
        
        lines = [1.5, 2.5, 3.5]
        
        # Generate distribution parameters
        n = 100
        predictions_df = pd.DataFrame({
            'mu': np.random.uniform(2, 4, n),
            'alpha': np.random.uniform(1, 2, n)
        })
        
        # Generate actuals
        actuals = np.random.poisson(3, n)
        
        calibrator = MultiLineCalibrator(lines, method='isotonic')
        calibrator.fit(predictions_df, actuals)
        
        calibrated = calibrator.transform(predictions_df)
        
        # Check calibrated probabilities exist
        for line in lines:
            assert f'p_over_{line}_calibrated' in calibrated.columns
    
    def test_distribution_calibrator(self, setup_test_config):
        """Test distribution-level calibration."""
        np.random.seed(42)
        
        n = 100
        predictions_df = pd.DataFrame({
            'mu': np.random.uniform(2, 4, n),
            'alpha': np.random.uniform(1, 2, n)
        })
        
        actuals = np.random.poisson(3, n)
        
        calibrator = DistributionCalibrator()
        calibrator.fit(predictions_df, actuals)
        
        calibrated = calibrator.transform(predictions_df)
        
        assert 'mu' in calibrated.columns
        assert 'alpha' in calibrated.columns
        assert len(calibrated) == n


class TestBaselineModels:
    """Tests for baseline models."""
    
    def test_season_mean_baseline(self, setup_test_config, sample_historical_for_baselines):
        """Test season mean baseline."""
        model = SeasonMeanBaseline()
        model.fit(sample_historical_for_baselines)
        
        # Check fitted attributes
        assert len(model.player_means) > 0
        assert len(model.player_dispersions) > 0
        
        # Predict
        mu, alpha = model.predict_distribution(1)
        
        assert mu > 0
        assert alpha > 0
    
    def test_opponent_adjusted_baseline(self, setup_test_config, sample_historical_for_baselines):
        """Test opponent-adjusted baseline."""
        model = OpponentAdjustedBaseline()
        model.fit(sample_historical_for_baselines)
        
        # Predict without context
        mu1, alpha1 = model.predict_distribution(1)
        
        # Predict with context
        mu2, alpha2 = model.predict_distribution(1, context={'opponent_team_id': 10})
        
        assert mu1 > 0
        assert mu2 > 0
        # Mu might differ based on opponent
    
    def test_ewma_baseline(self, setup_test_config, sample_historical_for_baselines):
        """Test EWMA baseline."""
        model = EWMABaseline()
        model.fit(sample_historical_for_baselines)
        
        mu, alpha = model.predict_distribution(1)
        
        assert mu > 0
        assert alpha > 0
    
    def test_toi_adjusted_ewma(self, setup_test_config, sample_historical_for_baselines):
        """Test TOI-adjusted EWMA baseline."""
        model = TOIAdjustedEWMA()
        model.fit(sample_historical_for_baselines)
        
        # Predict with expected TOI
        mu, alpha = model.predict_distribution(1, context={'expected_toi_minutes': 20})
        
        assert mu > 0
        assert alpha > 0
    
    def test_create_baseline_models(self, setup_test_config):
        """Test baseline model factory."""
        baselines = create_baseline_models()
        
        assert isinstance(baselines, dict)
        assert len(baselines) > 0
        
        # Check each baseline is correct type
        for name, model in baselines.items():
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict_distribution')
    
    def test_baseline_predict_probabilities(self, setup_test_config, sample_historical_for_baselines):
        """Test baseline probability predictions."""
        model = SeasonMeanBaseline()
        model.fit(sample_historical_for_baselines)
        
        probs = model.predict_probabilities(1)
        
        assert isinstance(probs, dict)
        assert 'p_over_2.5' in probs
        
        # Check probabilities are valid
        for key, value in probs.items():
            assert 0 <= value <= 1


class TestModelIntegration:
    """Integration tests for complete modeling workflow."""
    
    def test_train_predict_evaluate_workflow(self, setup_test_config, sample_training_data):
        """Test complete workflow."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train
        model = LGBMNegativeBinomialModel()
        model.fit(X_train, y_train, verbose=False)
        
        # Predict
        predictions = model.predict_distribution(X_test)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        assert metrics['crps'] > 0
        assert metrics['crps'] < 5  # Should be reasonable
    
    def test_calibration_workflow(self, setup_test_config, sample_training_data):
        """Test model + calibration workflow."""
        X, y = sample_training_data
        
        # Split data
        train_idx = int(0.6 * len(X))
        val_idx = int(0.8 * len(X))
        
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
        
        # Train model
        model = LGBMNegativeBinomialModel()
        model.fit(X_train, y_train, verbose=False)
        
        # Get validation predictions
        val_predictions = model.predict_distribution(X_val)
        
        # Fit calibrator
        calibrator = create_calibrator(method='isotonic')
        calibrator.fit(val_predictions, y_val)
        
        # Apply calibration to test set
        test_predictions = model.predict_distribution(X_test)
        calibrated_predictions = calibrator.transform(test_predictions)
        
        assert len(calibrated_predictions) == len(test_predictions)
    
    def test_parameter_constraints(self, setup_test_config, sample_training_data):
        """Test that model respects parameter constraints."""
        X, y = sample_training_data
        
        model = LGBMNegativeBinomialModel()
        model.fit(X[:400], y[:400], verbose=False)
        
        predictions = model.predict_distribution(X[400:])
        
        # Check mu is positive
        assert (predictions['mu'] > 0).all()
        
        # Check alpha is within configured bounds
        config = model.config
        assert (predictions['alpha'] >= config.model.nb_min_dispersion).all()
        assert (predictions['alpha'] <= config.model.nb_max_dispersion).all()
    
    def test_reproducibility(self, setup_test_config, sample_training_data):
        """Test that results are reproducible with same seed."""
        X, y = sample_training_data
        
        # Train first model
        model1 = LGBMNegativeBinomialModel()
        model1.fit(X[:400], y[:400], verbose=False)
        pred1 = model1.predict_distribution(X[400:450])
        
        # Train second model with same data
        model2 = LGBMNegativeBinomialModel()
        model2.fit(X[:400], y[:400], verbose=False)
        pred2 = model2.predict_distribution(X[400:450])
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(pred1['mu'].values, pred2['mu'].values, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])