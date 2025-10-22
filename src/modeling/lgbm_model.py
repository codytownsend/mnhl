"""
LightGBM model for predicting Negative Binomial parameters.

Two-headed approach:
- Model 1: Predicts mu (mean SOG)
- Model 2: Predicts alpha (dispersion parameter)

Together they define the full distribution P(SOG = k).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
from scipy import stats
from datetime import datetime

from src.utils.config import get_config
from src.validation.metrics import (
    calculate_crps, get_probability_over_line, 
    generate_pmf, calculate_prediction_interval
)


class LGBMNegativeBinomialModel:
    """
    LightGBM model for SOG prediction using Negative Binomial distribution.
    
    Uses two separate models:
    1. Mu model: predicts expected SOG (Poisson objective)
    2. Alpha model: predicts dispersion (Gamma objective)
    
    The combination (mu, alpha) defines NB distribution for each player.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize LGBM NB model.
        
        Args:
            config_override: Optional dict to override config parameters
        """
        self.config = get_config()
        self.config_override = config_override or {}
        
        # Model objects
        self.mu_model = None
        self.alpha_model = None
        
        # Feature names (set during training)
        self.feature_names = None
        
        # Training metadata
        self.training_date = None
        self.version = None
        
        # Feature importance
        self.feature_importance_mu = None
        self.feature_importance_alpha = None
    
    def _get_model_params(self, model_type: str) -> Dict:
        """
        Get LightGBM parameters for mu or alpha model.
        
        Args:
            model_type: 'mu' or 'alpha'
            
        Returns:
            Dict of LightGBM parameters
        """
        if model_type == 'mu':
            base_params = self.config.model.mu_model.to_dict()
        else:
            base_params = self.config.model.alpha_model.to_dict()
        
        # Apply any overrides
        params = {**base_params, **self.config_override.get(model_type, {})}
        
        return params
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling.
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Processed features
        """
        X_processed = X.copy()
        
        # Drop non-feature columns if present
        drop_cols = ['player_id', 'game_id', 'player_name', 'game_date', 
                     'as_of_timestamp', 'shots', 'actual_shots']
        X_processed = X_processed.drop(
            columns=[col for col in drop_cols if col in X_processed.columns],
            errors='ignore'
        )
        
        # Handle missing values
        X_processed = X_processed.fillna(0)
        
        # Store feature names on first call
        if self.feature_names is None:
            self.feature_names = list(X_processed.columns)
        
        # Ensure consistent feature order
        X_processed = X_processed[self.feature_names]
        
        return X_processed
    
    def _calculate_residual_dispersion(self, y_true: np.ndarray, 
                                       mu_pred: np.ndarray) -> np.ndarray:
        """
        Calculate target dispersion values from residuals.
        
        For each prediction, estimate alpha from the squared residual.
        
        Args:
            y_true: Actual SOG values
            mu_pred: Predicted mean values
            
        Returns:
            Target alpha values for dispersion model
        """
        # Squared residuals as proxy for variance
        residuals = y_true - mu_pred
        squared_residuals = residuals ** 2
        
        # Estimate alpha from variance
        # var = mu + mu^2/alpha
        # alpha = mu^2 / (var - mu)
        
        alpha_targets = []
        for mu, sq_res in zip(mu_pred, squared_residuals):
            # Use squared residual as variance estimate
            var_est = max(sq_res, mu + 0.1)  # Ensure var > mu
            
            if var_est > mu:
                alpha = (mu ** 2) / (var_est - mu)
            else:
                alpha = self.config.model.nb_max_dispersion
            
            # Clip to reasonable range
            alpha = np.clip(
                alpha,
                self.config.model.nb_min_dispersion,
                self.config.model.nb_max_dispersion
            )
            alpha_targets.append(alpha)
        
        return np.array(alpha_targets)
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train both mu and alpha models.
        
        Args:
            X_train: Training features
            y_train: Training targets (actual SOG)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print training progress
            
        Returns:
            Dict with training history
        """
        X_train_processed = self._prepare_features(X_train)
        
        if verbose:
            print(f"Training with {len(X_train_processed)} samples, {len(self.feature_names)} features")
        
        # Step 1: Train mu model (predicts mean SOG)
        if verbose:
            print("\n=== Training Mu Model (Mean) ===")
        
        mu_params = self._get_model_params('mu')
        
        train_data_mu = lgb.Dataset(X_train_processed, label=y_train)
        
        eval_sets = [train_data_mu]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            X_val_processed = self._prepare_features(X_val)
            val_data_mu = lgb.Dataset(X_val_processed, label=y_val, reference=train_data_mu)
            eval_sets.append(val_data_mu)
            eval_names.append('valid')
        
        # Train mu model
        callbacks = []
        if self.config.training.early_stopping_enabled and X_val is not None:
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.config.training.early_stopping_rounds
            ))
        
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))
        
        self.mu_model = lgb.train(
            mu_params,
            train_data_mu,
            valid_sets=eval_sets,
            valid_names=eval_names,
            callbacks=callbacks
        )
        
        # Get mu predictions
        mu_train_pred = self.mu_model.predict(X_train_processed)
        
        # Step 2: Train alpha model (predicts dispersion)
        if verbose:
            print("\n=== Training Alpha Model (Dispersion) ===")
        
        # Calculate target dispersion values
        alpha_train_target = self._calculate_residual_dispersion(y_train, mu_train_pred)
        
        alpha_params = self._get_model_params('alpha')
        
        train_data_alpha = lgb.Dataset(X_train_processed, label=alpha_train_target)
        
        eval_sets_alpha = [train_data_alpha]
        
        if X_val is not None and y_val is not None:
            mu_val_pred = self.mu_model.predict(X_val_processed)
            alpha_val_target = self._calculate_residual_dispersion(y_val, mu_val_pred)
            val_data_alpha = lgb.Dataset(X_val_processed, label=alpha_val_target, 
                                         reference=train_data_alpha)
            eval_sets_alpha.append(val_data_alpha)
        
        # Train alpha model
        self.alpha_model = lgb.train(
            alpha_params,
            train_data_alpha,
            valid_sets=eval_sets_alpha,
            valid_names=eval_names,
            callbacks=callbacks
        )
        
        # Store feature importance
        self.feature_importance_mu = dict(zip(
            self.feature_names,
            self.mu_model.feature_importance(importance_type='gain')
        ))
        self.feature_importance_alpha = dict(zip(
            self.feature_names,
            self.alpha_model.feature_importance(importance_type='gain')
        ))
        
        # Set training metadata
        self.training_date = datetime.now()
        self.version = self.config.get_model_version_string()
        
        if verbose:
            print("\n=== Training Complete ===")
            print(f"Model version: {self.version}")
            print(f"Top 5 features (mu model):")
            for feat, imp in sorted(self.feature_importance_mu.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {feat}: {imp:.1f}")
        
        # Return training history (placeholder)
        history = {
            'train_loss_mu': [],
            'train_loss_alpha': [],
        }
        
        return history
    
    def predict_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict NB distribution parameters for each sample.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with columns [mu, alpha]
        """
        if self.mu_model is None or self.alpha_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_processed = self._prepare_features(X)
        
        # Predict mu and alpha
        mu_pred = self.mu_model.predict(X_processed)
        alpha_pred = self.alpha_model.predict(X_processed)
        
        # Clip predictions to valid ranges
        mu_pred = np.maximum(mu_pred, 0.1)  # Ensure positive mean
        alpha_pred = np.clip(
            alpha_pred,
            self.config.model.nb_min_dispersion,
            self.config.model.nb_max_dispersion
        )
        
        return pd.DataFrame({
            'mu': mu_pred,
            'alpha': alpha_pred
        })
    
    def predict_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probabilities for common Over/Under lines.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with p_over_K for each common line
        """
        dist_params = self.predict_distribution(X)
        
        probs = {}
        for line in self.config.model.common_lines:
            p_over = []
            for _, row in dist_params.iterrows():
                p = get_probability_over_line(row['mu'], row['alpha'], line)
                p_over.append(p)
            probs[f'p_over_{line}'] = p_over
        
        return pd.DataFrame(probs)
    
    def predict_pmf(self, X: pd.DataFrame, max_k: int = 15) -> List[Dict[int, float]]:
        """
        Predict full probability mass function for each sample.
        
        Args:
            X: Feature DataFrame
            max_k: Maximum SOG value to calculate
            
        Returns:
            List of PMF dicts (one per sample)
        """
        dist_params = self.predict_distribution(X)
        
        pmfs = []
        for _, row in dist_params.iterrows():
            pmf = generate_pmf(row['mu'], row['alpha'], max_k)
            pmfs.append(pmf)
        
        return pmfs
    
    def predict_intervals(self, X: pd.DataFrame, 
                        confidence_level: float = 0.8) -> pd.DataFrame:
        """
        Predict confidence intervals.
        
        Args:
            X: Feature DataFrame
            confidence_level: Confidence level (e.g., 0.8 for 80% CI)
            
        Returns:
            DataFrame with lower and upper bounds
        """
        dist_params = self.predict_distribution(X)
        
        intervals = []
        for _, row in dist_params.iterrows():
            lower, upper = calculate_prediction_interval(
                row['mu'], row['alpha'], confidence_level
            )
            intervals.append({'lower': lower, 'upper': upper})
        
        return pd.DataFrame(intervals)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dict of metrics
        """
        dist_params = self.predict_distribution(X)
        
        # Calculate CRPS
        crps_scores = []
        for (_, row), actual in zip(dist_params.iterrows(), y):
            crps = calculate_crps(actual, row['mu'], row['alpha'])
            crps_scores.append(crps)
        
        # Calculate Brier scores
        brier_scores = {}
        for line in [1.5, 2.5, 3.5, 4.5]:
            brier = []
            for (_, row), actual in zip(dist_params.iterrows(), y):
                p_over = get_probability_over_line(row['mu'], row['alpha'], line)
                actual_over = 1 if actual > line else 0
                brier.append((p_over - actual_over) ** 2)
            brier_scores[f'brier_{line}'] = np.mean(brier)
        
        metrics = {
            'crps': np.mean(crps_scores),
            'mae': np.mean(np.abs(y - dist_params['mu'])),
            'rmse': np.sqrt(np.mean((y - dist_params['mu']) ** 2)),
            **brier_scores
        }
        
        return metrics
    
    def get_feature_importance(self, model_type: str = 'mu', 
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            model_type: 'mu' or 'alpha'
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and importance scores
        """
        if model_type == 'mu':
            importance = self.feature_importance_mu
        else:
            importance = self.feature_importance_alpha
        
        if importance is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM models
        self.mu_model.save_model(str(path / 'mu_model.txt'))
        self.alpha_model.save_model(str(path / 'alpha_model.txt'))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'version': self.version,
            'feature_importance_mu': self.feature_importance_mu,
            'feature_importance_alpha': self.feature_importance_alpha,
            'config_override': self.config_override,
        }
        
        joblib.dump(metadata, path / 'metadata.pkl')
        
        print(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory containing model files
        """
        path = Path(path)
        
        # Load LightGBM models
        self.mu_model = lgb.Booster(model_file=str(path / 'mu_model.txt'))
        self.alpha_model = lgb.Booster(model_file=str(path / 'alpha_model.txt'))
        
        # Load metadata
        metadata = joblib.load(path / 'metadata.pkl')
        self.feature_names = metadata['feature_names']
        self.training_date = datetime.fromisoformat(metadata['training_date']) if metadata['training_date'] else None
        self.version = metadata['version']
        self.feature_importance_mu = metadata['feature_importance_mu']
        self.feature_importance_alpha = metadata['feature_importance_alpha']
        self.config_override = metadata.get('config_override', {})
        
        print(f"Model loaded from {path}")
        print(f"Version: {self.version}, trained: {self.training_date}")
    
    def inflate_uncertainty(self, X: pd.DataFrame, 
                           dist_params: pd.DataFrame) -> pd.DataFrame:
        """
        Inflate uncertainty for low-confidence predictions.
        
        Increases dispersion parameter when conditions indicate uncertainty
        (e.g., few games played, lineup changes).
        
        Args:
            X: Features (for checking conditions)
            dist_params: Current distribution parameters
            
        Returns:
            Adjusted distribution parameters
        """
        if not self.config.calibration.inflate_uncertainty_enabled:
            return dist_params
        
        adjusted = dist_params.copy()
        inflation_factor = self.config.calibration.inflation_factor
        
        for condition in self.config.calibration.inflation_conditions:
            # Parse condition string (simple eval, could be more sophisticated)
            if 'games_played' in condition and 'games_played' in X.columns:
                threshold = int(condition.split('<')[-1].strip())
                mask = X['games_played'] < threshold
                adjusted.loc[mask, 'alpha'] *= inflation_factor
            
            elif 'projected_toi_std' in condition and 'projected_toi_std' in X.columns:
                threshold = float(condition.split('>')[-1].strip())
                mask = X['projected_toi_std'] > threshold
                adjusted.loc[mask, 'alpha'] *= inflation_factor
            
            elif 'lineup_confidence' in condition and 'lineup_confidence' in X.columns:
                threshold = float(condition.split('<')[-1].strip())
                mask = X['lineup_confidence'] < threshold
                adjusted.loc[mask, 'alpha'] *= inflation_factor
        
        # Ensure still within valid range
        adjusted['alpha'] = np.clip(
            adjusted['alpha'],
            self.config.model.nb_min_dispersion,
            self.config.model.nb_max_dispersion
        )
        
        return adjusted
    
    def predict_with_uncertainty_adjustment(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with automatic uncertainty inflation.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Adjusted distribution parameters
        """
        dist_params = self.predict_distribution(X)
        return self.inflate_uncertainty(X, dist_params)