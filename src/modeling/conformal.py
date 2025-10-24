"""Symmetric conformal interval calibration around the model mean."""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib

from src.utils.config import get_config


class ConformalIntervalCalibrator:
    """Symmetric split-conformal interval calibrator centered on ``mu``."""

    def __init__(self, confidence_levels: List[float]):
        self.confidence_levels = sorted(confidence_levels)
        self.quantiles: Dict[float, float] = {}
        self.config = get_config()
        self.is_fitted = False

    def fit(self, dist_params: pd.DataFrame, actuals: np.ndarray) -> None:
        """
        Fit conformal adjustments from validation predictions.

        Args:
            dist_params: DataFrame with columns ['mu', 'alpha']
            actuals: Array of actual shot counts
        """
        if len(dist_params) == 0:
            raise ValueError("Cannot fit conformal calibrator on empty data.")

        mu = dist_params['mu'].to_numpy(dtype=float)
        actuals = np.asarray(actuals, dtype=float)
        residuals = np.abs(actuals - mu)

        for level in self.confidence_levels:
            q = level
            try:
                quantile = np.quantile(residuals, q, method='higher')
            except TypeError:
                quantile = np.quantile(residuals, q, interpolation='higher')
            self.quantiles[level] = float(max(0.0, quantile))

        self.is_fitted = True

    def predict(self, dist_params: pd.DataFrame) -> pd.DataFrame:
        """
        Apply conformal adjustments to new predictions.

        Args:
            dist_params: DataFrame with columns ['mu', 'alpha']

        Returns:
            DataFrame with calibrated intervals for each confidence level.
        """
        if not self.is_fitted:
            raise ValueError("Conformal calibrator is not fitted.")

        mu = dist_params['mu'].to_numpy(dtype=float)
        results = {}
        for level in self.confidence_levels:
            adjustment = self.quantiles.get(level, 0.0)
            lower_adj = np.maximum(0.0, mu - adjustment)
            upper_adj = mu + adjustment

            key = f'ci_{int(level*100)}'
            results[f'{key}_lower'] = lower_adj
            results[f'{key}_upper'] = upper_adj

        return pd.DataFrame(results, index=dist_params.index)

    def get_metadata(self) -> Dict[str, float]:
        """
        Return stored quantiles for logging/debugging.
        """
        return self.quantiles.copy()


def save_conformal_calibrator(calibrator: ConformalIntervalCalibrator, path: Path) -> None:
    """Persist calibrator to disk."""
    data = {
        'confidence_levels': calibrator.confidence_levels,
        'quantiles': calibrator.quantiles,
        'is_fitted': calibrator.is_fitted,
    }
    joblib.dump(data, Path(path))


def load_conformal_calibrator(path: Path) -> ConformalIntervalCalibrator:
    """Load calibrator from disk."""
    data = joblib.load(Path(path))
    calibrator = ConformalIntervalCalibrator(data['confidence_levels'])
    calibrator.quantiles = data['quantiles']
    calibrator.is_fitted = data.get('is_fitted', True)
    return calibrator
