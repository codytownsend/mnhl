"""
Calibration Diagnostic Script

Analyzes why the model predictions are overconfident and visualizes
calibration quality across different prediction lines.

Usage:
    python diagnose_calibration.py --model-dir models/v20251021_crps

Outputs:
    - Reliability diagrams for each line (1.5, 2.5, 3.5, 4.5)
    - Alpha (dispersion) distribution analysis
    - Coverage analysis by prediction confidence
    - Recommendations for calibration fixes
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.modeling.lgbm_model import LGBMNegativeBinomialModel
from src.validation.metrics import MetricsCalculator
from src.utils.config import load_config, get_config


def load_model_and_data(model_dir: Path) -> Tuple[LGBMNegativeBinomialModel, pd.DataFrame, pd.DataFrame]:
    """Load trained model and test data."""
    print(f"Loading model from {model_dir}")
    
    # Load model
    model = LGBMNegativeBinomialModel()
    model.load(model_dir)
    
    # Load test data (should be saved during training)
    test_features_path = Path("data/processed/test_features.parquet")
    test_targets_path = Path("data/processed/test_targets.parquet")
    
    if not test_features_path.exists() or not test_targets_path.exists():
        raise FileNotFoundError(
            "Test data not found. Re-run training with --save-test-data flag."
        )
    
    X_test = pd.read_parquet(test_features_path)
    y_test = pd.read_parquet(test_targets_path)
    
    print(f"Loaded {len(X_test)} test samples")
    
    return model, X_test, y_test


def get_predictions(model: LGBMNegativeBinomialModel, X: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions with full distribution."""
    print("Generating predictions...")
    
    # Get distribution parameters
    distribution = model.predict_distribution(X)
    
    preds = pd.DataFrame({
        'mu': distribution['mu'],
        'alpha': distribution['alpha'],
    })
    
    # Calculate probabilities for common lines
    probs_df = model.predict_probabilities(X)
    for col in probs_df.columns:
        preds[col.replace('p_over_', 'prob_over_')] = probs_df[col]
    
    # Calculate prediction intervals
    for conf in [0.5, 0.8, 0.9, 0.95]:
        intervals = model.predict_intervals(X, confidence_level=conf)
        preds[f'ci_{int(conf*100)}_lower'] = intervals['lower']
        preds[f'ci_{int(conf*100)}_upper'] = intervals['upper']
    
    print(f"Generated predictions for {len(preds)} samples")
    return preds


def analyze_alpha_distribution(preds: pd.DataFrame) -> None:
    """Analyze the predicted dispersion parameter."""
    print("\n" + "="*70)
    print("ALPHA (DISPERSION) DISTRIBUTION ANALYSIS")
    print("="*70)
    
    alpha = preds['alpha']
    
    print(f"\nAlpha Statistics:")
    print(f"  Mean:   {alpha.mean():.4f}")
    print(f"  Median: {alpha.median():.4f}")
    print(f"  Std:    {alpha.std():.4f}")
    print(f"  Min:    {alpha.min():.4f}")
    print(f"  Max:    {alpha.max():.4f}")
    print(f"  Q25:    {alpha.quantile(0.25):.4f}")
    print(f"  Q75:    {alpha.quantile(0.75):.4f}")
    
    # Check for unrealistic values
    if alpha.min() < 0.1:
        print(f"\n⚠️  WARNING: {(alpha < 0.1).sum()} predictions have alpha < 0.1")
        print("    This indicates severe underdispersion (overconfidence)")
    
    if alpha.max() > 20:
        print(f"\n⚠️  WARNING: {(alpha > 20).sum()} predictions have alpha > 20")
        print("    This indicates severe overdispersion (underconfidence)")
    
    # Expected range for SOG
    print("\n📊 Expected Range for NHL Shots:")
    print("   Typical alpha for count data: 1-5")
    print("   For low-volume events (SOG): 0.5-3")
    print("   Very dispersed players: 3-10")
    
    in_range = ((alpha >= 0.5) & (alpha <= 10)).sum()
    print(f"\n   Predictions in reasonable range: {in_range}/{len(alpha)} ({100*in_range/len(alpha):.1f}%)")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(alpha, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(alpha.mean(), color='red', linestyle='--', label=f'Mean: {alpha.mean():.2f}')
    axes[0].axvline(alpha.median(), color='green', linestyle='--', label=f'Median: {alpha.median():.2f}')
    axes[0].set_xlabel('Alpha (Dispersion)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Predicted Alpha')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Log scale
    axes[1].hist(np.log10(alpha), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('log10(Alpha)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of log10(Alpha)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/alpha_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot: outputs/alpha_distribution.png")


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                             line: float, ax: plt.Axes) -> Dict[str, float]:
    """
    Plot reliability diagram for a specific line.
    
    Returns dictionary with calibration metrics.
    """
    # Bin predictions
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate observed frequency in each bin
    bin_counts = np.zeros(n_bins)
    bin_means_pred = np.zeros(n_bins)
    bin_means_obs = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_counts[i] = mask.sum()
            bin_means_pred[i] = y_prob[mask].mean()
            bin_means_obs[i] = (y_true[mask] > line).mean()
    
    # Remove empty bins
    valid_bins = bin_counts > 0
    bin_means_pred = bin_means_pred[valid_bins]
    bin_means_obs = bin_means_obs[valid_bins]
    bin_counts = bin_counts[valid_bins]
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax.scatter(bin_means_pred, bin_means_obs, s=bin_counts/10, alpha=0.6, 
               c=bin_means_pred, cmap='viridis')
    
    # Add error bars
    for pred, obs, count in zip(bin_means_pred, bin_means_obs, bin_counts):
        se = np.sqrt(obs * (1 - obs) / count) if count > 0 else 0
        ax.errorbar(pred, obs, yerr=1.96*se, fmt='none', color='gray', alpha=0.3)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title(f'Reliability Diagram: Over {line} Line')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Calculate calibration metrics
    ece = np.sum(bin_counts * np.abs(bin_means_pred - bin_means_obs)) / bin_counts.sum()
    mce = np.max(np.abs(bin_means_pred - bin_means_obs))
    
    # Add text box with metrics
    textstr = f'ECE: {ece:.4f}\nMCE: {mce:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    return {
        'ece': ece,
        'mce': mce,
        'n_bins': len(bin_means_pred)
    }


def analyze_calibration_by_line(preds: pd.DataFrame, y_true: pd.Series) -> None:
    """Create reliability diagrams for each common line."""
    print("\n" + "="*70)
    print("RELIABILITY DIAGRAM ANALYSIS")
    print("="*70)
    
    lines = [1.5, 2.5, 3.5, 4.5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics_by_line = {}
    
    for i, line in enumerate(lines):
        y_prob = preds[f'prob_over_{line}'].values
        metrics = plot_reliability_diagram(y_true.values, y_prob, line, axes[i])
        metrics_by_line[line] = metrics
        
        print(f"\nLine {line}:")
        print(f"  Expected Calibration Error (ECE): {metrics['ece']:.4f}")
        print(f"  Maximum Calibration Error (MCE): {metrics['mce']:.4f}")
        print(f"  Valid bins: {metrics['n_bins']}/10")
    
    plt.tight_layout()
    plt.savefig('outputs/reliability_diagrams.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot: outputs/reliability_diagrams.png")
    
    # Overall assessment
    avg_ece = np.mean([m['ece'] for m in metrics_by_line.values()])
    print(f"\n📊 Overall Calibration:")
    print(f"   Average ECE: {avg_ece:.4f}")
    if avg_ece < 0.01:
        print("   ✓ EXCELLENT calibration")
    elif avg_ece < 0.03:
        print("   ✓ GOOD calibration")
    elif avg_ece < 0.05:
        print("   ⚠️  ACCEPTABLE calibration (could be improved)")
    else:
        print("   ❌ POOR calibration (needs fixing)")


def analyze_coverage(preds: pd.DataFrame, y_true: pd.Series) -> None:
    """Analyze prediction interval coverage."""
    print("\n" + "="*70)
    print("PREDICTION INTERVAL COVERAGE ANALYSIS")
    print("="*70)
    
    confidence_levels = [0.5, 0.8, 0.9, 0.95]
    
    results = []
    for conf in confidence_levels:
        lower = preds[f'ci_{int(conf*100)}_lower']
        upper = preds[f'ci_{int(conf*100)}_upper']
        
        # Calculate coverage
        coverage = ((y_true >= lower) & (y_true <= upper)).mean()
        
        # Calculate interval width
        width = (upper - lower).mean()
        
        results.append({
            'confidence': conf,
            'expected_coverage': conf,
            'observed_coverage': coverage,
            'error': coverage - conf,
            'avg_width': width
        })
        
        status = "✓" if abs(coverage - conf) < 0.05 else "❌"
        print(f"\n{int(conf*100)}% CI:")
        print(f"  Expected coverage: {conf:.3f}")
        print(f"  Observed coverage: {coverage:.3f} {status}")
        print(f"  Error:             {coverage - conf:+.3f}")
        print(f"  Avg width:         {width:.2f} shots")
    
    # Plot coverage analysis
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Coverage comparison
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect coverage')
    axes[0].scatter(results_df['expected_coverage'], results_df['observed_coverage'], 
                    s=100, alpha=0.6)
    axes[0].set_xlabel('Expected Coverage')
    axes[0].set_ylabel('Observed Coverage')
    axes[0].set_title('Coverage Calibration')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0.4, 1.0])
    axes[0].set_ylim([0.4, 1.0])
    
    # Width by confidence level
    axes[1].bar(range(len(results_df)), results_df['avg_width'], alpha=0.7)
    axes[1].set_xticks(range(len(results_df)))
    axes[1].set_xticklabels([f"{int(c*100)}%" for c in confidence_levels])
    axes[1].set_xlabel('Confidence Level')
    axes[1].set_ylabel('Average Interval Width (shots)')
    axes[1].set_title('Prediction Interval Widths')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/coverage_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot: outputs/coverage_analysis.png")
    
    # Diagnosis
    print("\n📋 DIAGNOSIS:")
    avg_error = results_df['error'].abs().mean()
    if avg_error > 0.15:
        print("   ❌ SEVERE miscalibration - intervals are significantly wrong")
        print("   → Likely cause: Alpha (dispersion) is underestimated")
        print("   → Recommendation: Increase alpha predictions by 50-100%")
    elif avg_error > 0.08:
        print("   ⚠️  MODERATE miscalibration - intervals need adjustment")
        print("   → Recommendation: Apply temperature scaling or conformal prediction")
    else:
        print("   ✓ Coverage is reasonable")


def analyze_by_prediction_confidence(preds: pd.DataFrame, y_true: pd.Series) -> None:
    """Analyze calibration separately for high vs low confidence predictions."""
    print("\n" + "="*70)
    print("CALIBRATION BY PREDICTION CONFIDENCE")
    print("="*70)
    
    # Use 2.5 line as reference (most common)
    y_prob = preds['prob_over_2.5']
    
    # Define confidence categories
    very_confident = (y_prob > 0.7) | (y_prob < 0.3)
    uncertain = (y_prob >= 0.4) & (y_prob <= 0.6)
    
    print(f"\nVery Confident Predictions (p < 0.3 or p > 0.7):")
    print(f"  Count: {very_confident.sum()}")
    if very_confident.sum() > 0:
        obs = (y_true[very_confident] > 2.5).mean()
        pred = y_prob[very_confident].mean()
        print(f"  Avg predicted prob: {pred:.3f}")
        print(f"  Avg observed freq:  {obs:.3f}")
        print(f"  Calibration error:  {abs(pred - obs):.3f}")
    
    print(f"\nUncertain Predictions (0.4 ≤ p ≤ 0.6):")
    print(f"  Count: {uncertain.sum()}")
    if uncertain.sum() > 0:
        obs = (y_true[uncertain] > 2.5).mean()
        pred = y_prob[uncertain].mean()
        print(f"  Avg predicted prob: {pred:.3f}")
        print(f"  Avg observed freq:  {obs:.3f}")
        print(f"  Calibration error:  {abs(pred - obs):.3f}")


def generate_recommendations(preds: pd.DataFrame, y_true: pd.Series) -> None:
    """Generate specific recommendations based on diagnostics."""
    print("\n" + "="*70)
    print("RECOMMENDED FIXES")
    print("="*70)
    
    # Calculate key metrics
    alpha_mean = preds['alpha'].mean()
    ci_50_coverage = ((y_true >= preds['ci_50_lower']) & (y_true <= preds['ci_50_upper'])).mean()
    ci_80_coverage = ((y_true >= preds['ci_80_lower']) & (y_true <= preds['ci_80_upper'])).mean()
    
    ece_2_5 = np.abs(preds['prob_over_2.5'].mean() - (y_true > 2.5).mean())
    
    print("\n1. ALPHA (DISPERSION) ADJUSTMENT:")
    if alpha_mean < 1.0:
        multiplier = 2.0 / alpha_mean
        print(f"   ❌ Alpha is too low (mean: {alpha_mean:.3f})")
        print(f"   → Multiply all alpha predictions by {multiplier:.2f}")
        print(f"   → In calibration code: alpha_adjusted = alpha_pred * {multiplier:.2f}")
    elif alpha_mean < 2.0:
        multiplier = 2.5 / alpha_mean
        print(f"   ⚠️  Alpha is slightly low (mean: {alpha_mean:.3f})")
        print(f"   → Multiply all alpha predictions by {multiplier:.2f}")
    else:
        print(f"   ✓ Alpha looks reasonable (mean: {alpha_mean:.3f})")
    
    print("\n2. CALIBRATION METHOD:")
    if ci_50_coverage > 0.7:
        print("   ❌ Severe overconfidence detected")
        print("   → Switch from isotonic regression to conformal prediction")
        print("   → Use quantile-based calibration for intervals")
    elif ci_50_coverage > 0.6:
        print("   ⚠️  Moderate overconfidence")
        print("   → Try temperature scaling before isotonic regression")
        print("   → Add conformal prediction as backup")
    else:
        print("   ✓ Current calibration method may be adequate")
        print("   → Fine-tune isotonic regression parameters")
    
    print("\n3. MODEL TRAINING:")
    if ece_2_5 > 0.05:
        print("   ⚠️  Poor probability calibration")
        print("   → Add focal loss or class weights during training")
        print("   → Increase learning rate for alpha model")
    else:
        print("   ✓ Model probabilities are reasonable")
    
    print("\n4. DATA QUALITY:")
    print("   → Check for feature leakage (verify as_of timestamps)")
    print("   → Ensure test set is truly held-out")
    print("   → Validate that rolling windows don't look ahead")


def main():
    parser = argparse.ArgumentParser(description='Diagnose calibration issues')
    parser.add_argument('--model-dir', type=str, default='models/v20251021_crps',
                       help='Path to model directory')
    args = parser.parse_args()
    load_config('config/model_config.yaml')
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("="*70)
    print("NHL SOG MODEL: CALIBRATION DIAGNOSTICS")
    print("="*70)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(Path(args.model_dir))
    
    # Generate predictions
    preds = get_predictions(model, X_test)
    
    # Run diagnostics
    analyze_alpha_distribution(preds)
    analyze_calibration_by_line(preds, y_test['shots'])
    analyze_coverage(preds, y_test['shots'])
    analyze_by_prediction_confidence(preds, y_test['shots'])
    
    # Generate recommendations
    generate_recommendations(preds, y_test['shots'])
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nGenerated outputs:")
    print("  - outputs/alpha_distribution.png")
    print("  - outputs/reliability_diagrams.png")
    print("  - outputs/coverage_analysis.png")
    print("\nNext steps:")
    print("  1. Review plots to understand calibration issues")
    print("  2. Apply recommended fixes to calibration module")
    print("  3. Re-train model and re-run diagnostics")


if __name__ == '__main__':
    main()