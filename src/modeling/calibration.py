"""
Calibration module for post-processing predictions.

Ensures predicted probabilities match empirical frequencies through:
- Isotonic regression
- Platt scaling (logistic calibration)
- Beta calibration

Critical for honest uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats, optimize
import joblib
from pathlib import Path

from src.utils.config import get_config
from src.validation.metrics import calculate_calibration_error, get_probability_over_line


class CalibrationError(Exception):
    """Raised when calibration fails."""
    pass


class Calibrator:
    """
    Base calibrator class.
    
    Post-processes raw model probabilities to improve calibration.
    """
    
    def __init__(self, method: str):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method name
        """
        self.method = method
        self.is_fitted = False
    
    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Fit calibrator on validation data.
        
        Args:
            predictions: Raw predicted probabilities
            actuals: Binary outcomes (0 or 1)
        """
        raise NotImplementedError
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new predictions.
        
        Args:
            predictions: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        raise NotImplementedError
    
    def fit_transform(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(predictions, actuals)
        return self.transform(predictions)


class IsotonicCalibrator(Calibrator):
    """
    Isotonic regression calibrator.
    
    Non-parametric method that fits a monotonic function
    from predicted to calibrated probabilities.
    
    Good when you have sufficient validation data (>1000 samples).
    """
    
    def __init__(self):
        """Initialize isotonic calibrator."""
        super().__init__("isotonic")
        self.config = get_config()
        self.calibrator = IsotonicRegression(
            out_of_bounds='clip',
            y_min=0.0,
            y_max=1.0
        )
    
    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Fit isotonic regression.
        
        Args:
            predictions: Raw predicted probabilities
            actuals: Binary outcomes
        """
        if len(predictions) < self.config.calibration.min_samples_per_bin:
            raise CalibrationError(
                f"Insufficient samples for isotonic calibration: {len(predictions)} "
                f"(need at least {self.config.calibration.min_samples_per_bin})"
            )
        
        # Clip predictions to avoid numerical issues
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        
        self.calibrator.fit(predictions, actuals)
        self.is_fitted = True
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            predictions: Raw probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise CalibrationError("Calibrator not fitted. Call fit() first.")
        
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        calibrated = self.calibrator.transform(predictions)
        
        return calibrated


class PlattCalibrator(Calibrator):
    """
    Platt scaling (logistic calibration).
    
    Fits a logistic regression: P_calibrated = sigmoid(a * logit(P_raw) + b)
    
    More parametric than isotonic, works well with less data.
    """
    
    def __init__(self):
        """Initialize Platt calibrator."""
        super().__init__("platt")
        self.logistic = LogisticRegression(C=1e10, solver='lbfgs')
    
    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Fit logistic calibration.
        
        Args:
            predictions: Raw predicted probabilities
            actuals: Binary outcomes
        """
        # Convert to logits
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        logits = np.log(predictions / (1 - predictions))
        
        self.logistic.fit(logits.reshape(-1, 1), actuals)
        self.is_fitted = True
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling.
        
        Args:
            predictions: Raw probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise CalibrationError("Calibrator not fitted. Call fit() first.")
        
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        logits = np.log(predictions / (1 - predictions))
        
        calibrated = self.logistic.predict_proba(logits.reshape(-1, 1))[:, 1]
        
        return calibrated


class BetaCalibrator(Calibrator):
    """
    Beta calibration.
    
    Fits a beta distribution to calibrate probabilities.
    More flexible than Platt scaling.
    
    Reference: Kull et al. "Beyond sigmoids: How to obtain well-calibrated 
    probabilities from binary classifiers with beta calibration" (2017)
    """
    
    def __init__(self):
        """Initialize beta calibrator."""
        super().__init__("beta")
        self.params = None
    
    def _beta_calibration_loss(self, params: np.ndarray, 
                               predictions: np.ndarray, 
                               actuals: np.ndarray) -> float:
        """
        Loss function for beta calibration.
        
        Args:
            params: [a, b, c] parameters
            predictions: Raw probabilities
            actuals: Binary outcomes
            
        Returns:
            Negative log-likelihood
        """
        a, b, c = params
        
        # Transform predictions
        calibrated = np.clip(
            1 / (1 + np.exp(-a * np.log(predictions / (1 - predictions)) - b)),
            1e-6, 1 - 1e-6
        )
        
        # Log loss
        log_loss = -np.mean(
            actuals * np.log(calibrated) + (1 - actuals) * np.log(1 - calibrated)
        )
        
        return log_loss
    
    def fit(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Fit beta calibration.
        
        Args:
            predictions: Raw predicted probabilities
            actuals: Binary outcomes
        """
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        
        # Optimize parameters
        result = optimize.minimize(
            self._beta_calibration_loss,
            x0=[1.0, 0.0, 0.0],
            args=(predictions, actuals),
            method='L-BFGS-B'
        )
        
        if not result.success:
            # Fall back to identity calibration
            self.params = [1.0, 0.0, 0.0]
        else:
            self.params = result.x
        
        self.is_fitted = True
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply beta calibration.
        
        Args:
            predictions: Raw probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise CalibrationError("Calibrator not fitted. Call fit() first.")
        
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        a, b, c = self.params
        
        calibrated = 1 / (1 + np.exp(-a * np.log(predictions / (1 - predictions)) - b))
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        return calibrated


class MultiLineCalibrator:
    """
    Calibrate probabilities for multiple Over/Under lines simultaneously.
    
    Fits separate calibrators for each common line (1.5, 2.5, 3.5, 4.5).
    """
    
    def __init__(self, lines: List[float], method: str = None):
        """
        Initialize multi-line calibrator.
        
        Args:
            lines: List of lines to calibrate (e.g., [1.5, 2.5, 3.5, 4.5])
            method: Calibration method (uses config default if None)
        """
        self.config = get_config()
        self.lines = lines
        self.method = method or self.config.calibration.method
        
        # Create calibrator for each line
        self.calibrators = {line: self._create_calibrator() for line in lines}
    
    def _create_calibrator(self) -> Calibrator:
        """Create calibrator instance based on method."""
        if self.method == 'isotonic':
            return IsotonicCalibrator()
        elif self.method == 'platt':
            return PlattCalibrator()
        elif self.method == 'beta':
            return BetaCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def fit(self, predictions_df: pd.DataFrame, actuals: np.ndarray) -> None:
        """
        Fit calibrators for all lines.
        
        Args:
            predictions_df: DataFrame with columns [mu, alpha] or [p_over_1.5, p_over_2.5, ...]
            actuals: Actual SOG values
        """
        for line in self.lines:
            # Get predictions for this line
            if f'p_over_{line}' in predictions_df.columns:
                p_over = predictions_df[f'p_over_{line}'].values
            elif 'mu' in predictions_df.columns and 'alpha' in predictions_df.columns:
                # Calculate from distribution parameters
                p_over = np.array([
                    get_probability_over_line(row['mu'], row['alpha'], line)
                    for _, row in predictions_df.iterrows()
                ])
            else:
                raise ValueError(
                    f"predictions_df must have either 'p_over_{line}' or 'mu'/'alpha' columns"
                )
            
            # Convert actuals to binary
            actuals_binary = (actuals > line).astype(int)
            
            # Fit calibrator
            try:
                self.calibrators[line].fit(p_over, actuals_binary)
            except CalibrationError as e:
                print(f"Warning: Calibration failed for line {line}: {e}")
                # Leave uncalibrated
    
    def transform(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibration to predictions.
        
        Args:
            predictions_df: DataFrame with raw predictions
            
        Returns:
            DataFrame with calibrated probabilities
        """
        calibrated = predictions_df.copy()
        
        for line in self.lines:
            if not self.calibrators[line].is_fitted:
                continue  # Skip if calibrator wasn't fitted
            
            # Get raw predictions
            if f'p_over_{line}' in predictions_df.columns:
                p_over = predictions_df[f'p_over_{line}'].values
            elif 'mu' in predictions_df.columns and 'alpha' in predictions_df.columns:
                p_over = np.array([
                    get_probability_over_line(row['mu'], row['alpha'], line)
                    for _, row in predictions_df.iterrows()
                ])
            else:
                continue
            
            # Apply calibration
            p_calibrated = self.calibrators[line].transform(p_over)
            
            # Store calibrated predictions
            calibrated[f'p_over_{line}_calibrated'] = p_calibrated
        
        return calibrated
    
    def fit_transform(self, predictions_df: pd.DataFrame, 
                     actuals: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(predictions_df, actuals)
        return self.transform(predictions_df)
    
    def evaluate_calibration(self, predictions_df: pd.DataFrame,
                            actuals: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate calibration quality for each line.
        
        Args:
            predictions_df: Predictions (raw or calibrated)
            actuals: Actual SOG values
            
        Returns:
            Dict of line -> metrics
        """
        results = {}
        
        for line in self.lines:
            # Get predictions
            if f'p_over_{line}_calibrated' in predictions_df.columns:
                p_over = predictions_df[f'p_over_{line}_calibrated'].values
                label = 'calibrated'
            elif f'p_over_{line}' in predictions_df.columns:
                p_over = predictions_df[f'p_over_{line}'].values
                label = 'raw'
            else:
                continue
            
            actuals_binary = (actuals > line).astype(int)
            
            # Calculate calibration error
            cal_error, pred_probs, true_probs = calculate_calibration_error(
                p_over.tolist(), actuals_binary.tolist(),
                n_bins=self.config.calibration.n_bins
            )
            
            # Calculate Brier score
            brier = np.mean((p_over - actuals_binary) ** 2)
            
            results[f'{line}_{label}'] = {
                'calibration_error': cal_error,
                'brier_score': brier,
                'mean_predicted': np.mean(p_over),
                'mean_actual': np.mean(actuals_binary)
            }
        
        return results


class DistributionCalibrator:
    """
    Calibrate full distribution parameters (mu, alpha) rather than individual lines.
    
    Adjusts the Negative Binomial parameters to improve overall calibration.
    """
    
    def __init__(self):
        """Initialize distribution calibrator."""
        self.config = get_config()
        self.mu_adjustment = 1.0
        self.alpha_adjustment = 1.0
    
    def fit(self, predictions_df: pd.DataFrame, actuals: np.ndarray) -> None:
        """
        Fit distribution-level calibration.
        
        Finds multiplicative adjustments to mu and alpha that minimize CRPS.
        
        Args:
            predictions_df: DataFrame with mu, alpha columns
            actuals: Actual SOG values
        """
        from src.validation.metrics import calculate_crps
        
        def objective(params):
            mu_adj, alpha_adj = params
            
            total_crps = 0
            for (_, row), actual in zip(predictions_df.iterrows(), actuals):
                adj_mu = row['mu'] * mu_adj
                adj_alpha = row['alpha'] * alpha_adj
                
                # Clip to valid ranges
                adj_mu = max(adj_mu, 0.1)
                adj_alpha = np.clip(
                    adj_alpha,
                    self.config.model.nb_min_dispersion,
                    self.config.model.nb_max_dispersion
                )
                
                total_crps += calculate_crps(actual, adj_mu, adj_alpha)
            
            return total_crps / len(actuals)
        
        # Optimize adjustments
        result = optimize.minimize(
            objective,
            x0=[1.0, 1.0],
            bounds=[(0.5, 2.0), (0.5, 2.0)],
            method='L-BFGS-B'
        )
        
        if result.success:
            self.mu_adjustment, self.alpha_adjustment = result.x
        else:
            print("Warning: Distribution calibration optimization failed")
    
    def transform(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply distribution calibration.
        
        Args:
            predictions_df: DataFrame with mu, alpha
            
        Returns:
            DataFrame with calibrated mu, alpha
        """
        calibrated = predictions_df.copy()
        
        calibrated['mu'] = calibrated['mu'] * self.mu_adjustment
        calibrated['alpha'] = calibrated['alpha'] * self.alpha_adjustment
        
        # Clip to valid ranges
        calibrated['mu'] = calibrated['mu'].clip(lower=0.1)
        calibrated['alpha'] = calibrated['alpha'].clip(
            lower=self.config.model.nb_min_dispersion,
            upper=self.config.model.nb_max_dispersion
        )
        
        return calibrated
    
    def fit_transform(self, predictions_df: pd.DataFrame,
                     actuals: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(predictions_df, actuals)
        return self.transform(predictions_df)


def create_calibrator(method: str = None) -> MultiLineCalibrator:
    """
    Factory function to create calibrator.
    
    Args:
        method: Calibration method (uses config if None)
        
    Returns:
        Configured MultiLineCalibrator
    """
    config = get_config()
    lines = [1.5, 2.5, 3.5, 4.5]  # Standard lines
    
    return MultiLineCalibrator(lines, method=method)


def should_recalibrate(calibration_error: float) -> bool:
    """
    Check if model needs recalibration.
    
    Args:
        calibration_error: Current calibration error
        
    Returns:
        True if recalibration needed
    """
    config = get_config()
    return calibration_error > config.calibration.recalibration_threshold


def save_calibrator(calibrator: MultiLineCalibrator, path: Path) -> None:
    """
    Save calibrator to disk.
    
    Args:
        calibrator: Calibrator to save
        path: File path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(calibrator, path)
    print(f"Calibrator saved to {path}")


def load_calibrator(path: Path) -> MultiLineCalibrator:
    """
    Load calibrator from disk.
    
    Args:
        path: File path
        
    Returns:
        Loaded calibrator
    """
    calibrator = joblib.load(path)
    print(f"Calibrator loaded from {path}")
    return calibrator