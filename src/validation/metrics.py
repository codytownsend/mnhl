"""
Validation metrics for SOG prediction models.

Implements:
- CRPS (Continuous Ranked Probability Score)
- Brier scores for Over/Under lines
- Calibration error and reliability diagrams
- Log loss
- Coverage tests for confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

from src.utils.config import get_config


def calculate_crps(actual: int, mu: float, alpha: float) -> float:
    """
    Calculate Continuous Ranked Probability Score for Negative Binomial.
    
    CRPS measures the "distance" between predicted distribution and actual value.
    Lower is better (0 = perfect prediction).
    
    Args:
        actual: Actual SOG count
        mu: Predicted mean (NB parameter)
        alpha: Predicted dispersion (NB parameter)
        
    Returns:
        CRPS value
    """
    # CRPS for discrete distributions:
    # CRPS = sum_{k=0}^inf [F(k) - 1(actual <= k)]^2
    # Where F(k) is CDF
    
    # For computational efficiency, truncate at reasonable max
    max_k = int(mu + 5 * np.sqrt(mu + mu**2 / alpha))
    max_k = min(max_k, 20)  # Cap at 20 shots
    
    # NB CDF: F(k) = P(X <= k)
    p = alpha / (alpha + mu)  # Success probability
    
    crps = 0.0
    for k in range(max_k + 1):
        cdf_k = stats.nbinom.cdf(k, alpha, p)
        indicator = 1.0 if actual <= k else 0.0
        crps += (cdf_k - indicator) ** 2
    
    return crps


def calculate_crps_empirical(actual: int, pmf: Dict[int, float]) -> float:
    """
    Calculate CRPS from empirical PMF.
    
    Useful when you have pre-computed probability mass function.
    
    Args:
        actual: Actual SOG count
        pmf: Dict mapping k -> P(SOG = k)
        
    Returns:
        CRPS value
    """
    crps = 0.0
    cdf = 0.0
    
    max_k = max(pmf.keys())
    
    for k in range(max_k + 1):
        cdf += pmf.get(k, 0.0)
        indicator = 1.0 if actual <= k else 0.0
        crps += (cdf - indicator) ** 2
    
    return crps


def calculate_brier_score(actual_over: int, predicted_prob: float) -> float:
    """
    Calculate Brier score for binary outcome (Over/Under).
    
    Brier = (p - outcome)^2
    Where outcome is 1 if event occurred, 0 otherwise.
    Lower is better (0 = perfect).
    
    Args:
        actual_over: 1 if actual >= threshold, 0 otherwise
        predicted_prob: Predicted probability of Over
        
    Returns:
        Brier score
    """
    return (predicted_prob - actual_over) ** 2


def calculate_log_loss(actual_over: int, predicted_prob: float, 
                      epsilon: float = 1e-15) -> float:
    """
    Calculate log loss (cross-entropy) for binary outcome.
    
    Log Loss = -[y*log(p) + (1-y)*log(1-p)]
    Lower is better.
    
    Args:
        actual_over: 1 if actual >= threshold, 0 otherwise
        predicted_prob: Predicted probability of Over
        epsilon: Small value to prevent log(0)
        
    Returns:
        Log loss
    """
    # Clip probabilities to avoid log(0)
    p = np.clip(predicted_prob, epsilon, 1 - epsilon)
    
    if actual_over == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)


def get_probability_over_line(mu: float, alpha: float, line: float) -> float:
    """
    Calculate P(SOG > line) for Negative Binomial.
    
    Args:
        mu: Mean parameter
        alpha: Dispersion parameter
        line: Threshold (e.g., 2.5)
        
    Returns:
        Probability of exceeding line
    """
    k = int(np.ceil(line))  # Need at least k shots to be over
    p = alpha / (alpha + mu)
    
    # P(X >= k) = 1 - P(X <= k-1)
    p_over = 1 - stats.nbinom.cdf(k - 1, alpha, p)
    
    return p_over


def calculate_calibration_error(predictions: List[float], 
                                actuals: List[int],
                                n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate calibration error and reliability diagram data.
    
    Groups predictions into bins and compares predicted probability
    to empirical frequency in each bin.
    
    Args:
        predictions: List of predicted probabilities
        actuals: List of actual binary outcomes (0 or 1)
        n_bins: Number of bins for grouping
        
    Returns:
        (mean_absolute_error, bin_pred_probs, bin_true_probs)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Use sklearn's calibration_curve
    true_probs, pred_probs = calibration_curve(
        actuals, predictions, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate mean absolute calibration error
    calibration_error = np.mean(np.abs(true_probs - pred_probs))
    
    return calibration_error, pred_probs, true_probs


def calculate_coverage(predictions: List[Tuple[float, float]], 
                       actuals: List[int],
                       confidence_level: float = 0.8) -> float:
    """
    Calculate coverage: % of times actual value falls within confidence interval.
    
    For well-calibrated predictions, 80% CI should contain actual ~80% of time.
    
    Args:
        predictions: List of (mu, alpha) tuples
        actuals: List of actual SOG counts
        confidence_level: Target confidence level (e.g., 0.8 for 80% CI)
        
    Returns:
        Empirical coverage rate
    """
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    covered = 0
    actuals_array = np.asarray(actuals)
    if actuals_array.size == 0:
        return 0.0
    
    for (mu, alpha), actual in zip(predictions, actuals_array):
        p = alpha / (alpha + mu)
        
        # Get percentiles
        lower = stats.nbinom.ppf(lower_percentile, alpha, p)
        upper = stats.nbinom.ppf(upper_percentile, alpha, p)
        
        if lower <= actual <= upper:
            covered += 1
    
    return covered / actuals_array.size


def calculate_sharpness(predictions: List[Tuple[float, float]],
                        confidence_level: float = 0.8) -> float:
    """
    Calculate average width of confidence intervals.
    
    Sharper predictions (narrower intervals) are better, but only if calibrated.
    
    Args:
        predictions: List of (mu, alpha) tuples
        confidence_level: Target confidence level
        
    Returns:
        Average CI width
    """
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    widths = []
    
    for mu, alpha in predictions:
        p = alpha / (alpha + mu)
        lower = stats.nbinom.ppf(lower_percentile, alpha, p)
        upper = stats.nbinom.ppf(upper_percentile, alpha, p)
        widths.append(upper - lower)
    
    return np.mean(widths)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for model evaluation.
    
    Calculates all metrics specified in configuration and tracks
    performance across different subgroups.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.config = get_config()
    
    def evaluate_predictions(self, 
                            predictions_df: pd.DataFrame,
                            actuals_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all metrics for a set of predictions.
        
        Args:
            predictions_df: DataFrame with columns [player_id, game_id, mu, alpha]
            actuals_df: DataFrame with columns [player_id, game_id, actual_shots]
            
        Returns:
            Dict of metric name -> value
        """
        # Merge predictions with actuals
        merged = predictions_df.merge(
            actuals_df, 
            on=['player_id', 'game_id'],
            how='inner'
        )
        
        if merged.empty:
            return {}
        
        metrics = {}
        
        # CRPS
        crps_scores = [
            calculate_crps(row['actual_shots'], row['mu'], row['alpha'])
            for _, row in merged.iterrows()
        ]
        metrics['crps'] = np.mean(crps_scores)
        metrics['crps_std'] = np.std(crps_scores)
        
        # Brier scores for common lines
        for line in self.config.model.common_lines:
            if line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:  # Only standard lines
                brier_scores = []
                
                for _, row in merged.iterrows():
                    p_over = get_probability_over_line(row['mu'], row['alpha'], line)
                    actual_over = 1 if row['actual_shots'] > line else 0
                    brier = calculate_brier_score(actual_over, p_over)
                    brier_scores.append(brier)
                
                metrics[f'brier_{line}'] = np.mean(brier_scores)
        
        # Log loss for 2.5 line (most common)
        log_losses = []
        for _, row in merged.iterrows():
            p_over = get_probability_over_line(row['mu'], row['alpha'], 2.5)
            actual_over = 1 if row['actual_shots'] > 2.5 else 0
            log_loss = calculate_log_loss(actual_over, p_over)
            log_losses.append(log_loss)
        
        metrics['log_loss_2.5'] = np.mean(log_losses)
        
        # Calibration error for 2.5 line
        predictions_2_5 = [
            get_probability_over_line(row['mu'], row['alpha'], 2.5)
            for _, row in merged.iterrows()
        ]
        actuals_2_5 = [
            1 if row['actual_shots'] > 2.5 else 0
            for _, row in merged.iterrows()
        ]
        
        cal_error, _, _ = calculate_calibration_error(
            predictions_2_5, actuals_2_5, 
            n_bins=self.config.calibration.n_bins
        )
        metrics['calibration_error'] = cal_error
        
        # Coverage tests
        predictions_list = [(row['mu'], row['alpha']) for _, row in merged.iterrows()]
        actuals_list = merged['actual_shots'].tolist()
        
        for conf_level in self.config.validation.confidence_levels:
            coverage = calculate_coverage(predictions_list, actuals_list, conf_level)
            metrics[f'coverage_{int(conf_level*100)}'] = coverage
            
            # Sharpness (average CI width)
            sharpness = calculate_sharpness(predictions_list, conf_level)
            metrics[f'sharpness_{int(conf_level*100)}'] = sharpness
        
        # Mean absolute error
        mae = np.mean([
            abs(row['actual_shots'] - row['mu'])
            for _, row in merged.iterrows()
        ])
        metrics['mae'] = mae
        
        # Root mean squared error
        rmse = np.sqrt(np.mean([
            (row['actual_shots'] - row['mu']) ** 2
            for _, row in merged.iterrows()
        ]))
        metrics['rmse'] = rmse
        
        return metrics
    
    def evaluate_by_subgroup(self,
                            predictions_df: pd.DataFrame,
                            actuals_df: pd.DataFrame,
                            subgroup_col: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics separately for each subgroup.
        
        Useful for analyzing model performance across different contexts
        (e.g., by position, by opponent strength, by venue).
        
        Args:
            predictions_df: Predictions with subgroup column
            actuals_df: Actuals
            subgroup_col: Column to group by
            
        Returns:
            Dict of subgroup -> metrics dict
        """
        merged = predictions_df.merge(
            actuals_df,
            on=['player_id', 'game_id'],
            how='inner'
        )
        
        if subgroup_col not in merged.columns:
            return {}
        
        subgroup_metrics = {}
        
        for subgroup in merged[subgroup_col].unique():
            subgroup_data = merged[merged[subgroup_col] == subgroup]
            
            if len(subgroup_data) < 10:  # Skip small groups
                continue
            
            predictions_sub = subgroup_data[['player_id', 'game_id', 'mu', 'alpha']]
            actuals_sub = subgroup_data[['player_id', 'game_id', 'actual_shots']]
            
            metrics = self.evaluate_predictions(predictions_sub, actuals_sub)
            subgroup_metrics[str(subgroup)] = metrics
        
        return subgroup_metrics
    
    def generate_reliability_diagram_data(self,
                                         predictions_df: pd.DataFrame,
                                         actuals_df: pd.DataFrame,
                                         line: float = 2.5,
                                         n_bins: int = 10) -> pd.DataFrame:
        """
        Generate data for plotting reliability diagram.
        
        Args:
            predictions_df: Predictions
            actuals_df: Actuals
            line: Line to analyze
            n_bins: Number of bins
            
        Returns:
            DataFrame with columns [predicted_prob, true_prob, count]
        """
        merged = predictions_df.merge(
            actuals_df,
            on=['player_id', 'game_id'],
            how='inner'
        )
        
        predictions = [
            get_probability_over_line(row['mu'], row['alpha'], line)
            for _, row in merged.iterrows()
        ]
        
        actuals = [
            1 if row['actual_shots'] > line else 0
            for _, row in merged.iterrows()
        ]
        
        true_probs, pred_probs = calibration_curve(
            actuals, predictions, n_bins=n_bins, strategy='uniform'
        )
        
        # Count samples per bin
        bins = np.linspace(0, 1, n_bins + 1)
        counts = []
        for i in range(len(pred_probs)):
            lower = bins[i]
            upper = bins[i + 1]
            count = sum([lower <= p < upper for p in predictions])
            counts.append(count)
        
        return pd.DataFrame({
            'predicted_prob': pred_probs,
            'true_prob': true_probs,
            'count': counts
        })
    
    def compare_models(self,
                      model_predictions: Dict[str, pd.DataFrame],
                      actuals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare multiple models on same test set.
        
        Args:
            model_predictions: Dict of model_name -> predictions_df
            actuals_df: Actuals
            
        Returns:
            DataFrame comparing metrics across models
        """
        results = []
        
        for model_name, predictions_df in model_predictions.items():
            metrics = self.evaluate_predictions(predictions_df, actuals_df)
            metrics['model'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)


def calculate_prediction_interval(mu: float, alpha: float,
                                  confidence_level: float = 0.8) -> Tuple[int, int]:
    """
    Calculate prediction interval for given confidence level.
    
    Args:
        mu: Mean parameter
        alpha: Dispersion parameter
        confidence_level: Target confidence (e.g., 0.8 for 80% CI)
        
    Returns:
        (lower_bound, upper_bound) as integers
    """
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    
    p = alpha / (alpha + mu)
    
    lower = int(stats.nbinom.ppf(lower_percentile, alpha, p))
    upper = int(stats.nbinom.ppf(upper_percentile, alpha, p))
    
    return lower, upper


def generate_pmf(mu: float, alpha: float, max_k: int = 15) -> Dict[int, float]:
    """
    Generate probability mass function for Negative Binomial.
    
    Args:
        mu: Mean parameter
        alpha: Dispersion parameter
        max_k: Maximum value to calculate
        
    Returns:
        Dict mapping k -> P(SOG = k)
    """
    p = alpha / (alpha + mu)
    pmf = {}
    
    for k in range(max_k + 1):
        pmf[k] = stats.nbinom.pmf(k, alpha, p)
    
    return pmf


def calculate_expected_value(pmf: Dict[int, float]) -> float:
    """
    Calculate expected value from PMF.
    
    Args:
        pmf: Probability mass function
        
    Returns:
        Expected value (mean)
    """
    return sum(k * prob for k, prob in pmf.items())


def calculate_variance(pmf: Dict[int, float]) -> float:
    """
    Calculate variance from PMF.
    
    Args:
        pmf: Probability mass function
        
    Returns:
        Variance
    """
    mean = calculate_expected_value(pmf)
    return sum((k - mean)**2 * prob for k, prob in pmf.items())
