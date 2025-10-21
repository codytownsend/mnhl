"""
Baseline models for SOG prediction.

Simple models that our main model must beat to justify its complexity.
These serve as sanity checks and provide lower bounds on performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from scipy import stats

from src.utils.config import get_config


class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    def __init__(self, name: str):
        """
        Initialize baseline model.
        
        Args:
            name: Model name for identification
        """
        self.name = name
        self.config = get_config()
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the baseline model.
        
        Args:
            train_data: Training data with player_id, shots, game_date, etc.
        """
        pass
    
    @abstractmethod
    def predict_distribution(self, player_id: int, 
                            context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict mean and dispersion for a player.
        
        Args:
            player_id: Player ID
            context: Optional context (opponent, venue, etc.)
            
        Returns:
            (mu, alpha) for Negative Binomial distribution
        """
        pass
    
    def predict_probabilities(self, player_id: int,
                             context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Predict probabilities for common lines.
        
        Args:
            player_id: Player ID
            context: Optional context
            
        Returns:
            Dict with p_over_K for each common line
        """
        mu, alpha = self.predict_distribution(player_id, context)
        
        # Convert NB parameters to probabilities
        probs = {}
        for line in self.config.model.common_lines:
            # P(X > line) for continuous line (e.g., 2.5)
            k = int(np.ceil(line))  # Need at least k shots
            p_over = 1 - stats.nbinom.cdf(k - 1, alpha, alpha / (alpha + mu))
            probs[f'p_over_{line}'] = p_over
        
        return probs


class SeasonMeanBaseline(BaselineModel):
    """
    Simplest baseline: predict season average for each player.
    
    Assumes constant SOG rate, no context awareness.
    """
    
    def __init__(self):
        super().__init__("season_mean")
        self.player_means = {}
        self.player_dispersions = {}
        self.global_mean = 2.0
        self.global_dispersion = 1.5
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Calculate season mean and variance for each player.
        
        Args:
            train_data: DataFrame with player_id, shots columns
        """
        if train_data.empty:
            return
        
        # Calculate per-player statistics
        player_stats = train_data.groupby('player_id')['shots'].agg(['mean', 'var', 'count'])
        
        for player_id, row in player_stats.iterrows():
            mean_shots = row['mean']
            var_shots = row['var']
            n_games = row['count']
            
            # Only store if sufficient games
            if n_games >= self.config.features.min_games['player_stats']:
                self.player_means[player_id] = mean_shots
                
                # Estimate dispersion parameter from variance
                # For NB: var = mu + mu^2/alpha
                # Solve for alpha: alpha = mu^2 / (var - mu)
                if var_shots > mean_shots:
                    alpha = (mean_shots ** 2) / (var_shots - mean_shots)
                    alpha = np.clip(alpha, 
                                   self.config.model.nb_min_dispersion,
                                   self.config.model.nb_max_dispersion)
                else:
                    alpha = self.config.model.nb_max_dispersion  # Low dispersion
                
                self.player_dispersions[player_id] = alpha
        
        # Calculate global defaults
        self.global_mean = train_data['shots'].mean()
        self.global_dispersion = 1.5
    
    def predict_distribution(self, player_id: int,
                            context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict using season mean.
        
        Args:
            player_id: Player ID
            context: Ignored for this baseline
            
        Returns:
            (mu, alpha) for NB distribution
        """
        mu = self.player_means.get(player_id, self.global_mean)
        alpha = self.player_dispersions.get(player_id, self.global_dispersion)
        
        return mu, alpha


class OpponentAdjustedBaseline(BaselineModel):
    """
    Season mean adjusted by opponent strength.
    
    Adjusts player's base rate by opponent's defensive rating.
    """
    
    def __init__(self):
        super().__init__("opponent_adjusted")
        self.player_means = {}
        self.player_dispersions = {}
        self.opponent_factors = {}
        self.global_mean = 2.0
        self.global_dispersion = 1.5
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Calculate player means and opponent adjustment factors.
        
        Args:
            train_data: DataFrame with player_id, shots, opponent_team_id
        """
        if train_data.empty:
            return
        
        # Calculate per-player statistics (same as SeasonMeanBaseline)
        player_stats = train_data.groupby('player_id')['shots'].agg(['mean', 'var', 'count'])
        
        for player_id, row in player_stats.iterrows():
            if row['count'] >= self.config.features.min_games['player_stats']:
                self.player_means[player_id] = row['mean']
                
                var_shots = row['var']
                mean_shots = row['mean']
                if var_shots > mean_shots:
                    alpha = (mean_shots ** 2) / (var_shots - mean_shots)
                    alpha = np.clip(alpha,
                                   self.config.model.nb_min_dispersion,
                                   self.config.model.nb_max_dispersion)
                else:
                    alpha = self.config.model.nb_max_dispersion
                
                self.player_dispersions[player_id] = alpha
        
        # Calculate opponent difficulty factors
        if 'opponent_team_id' in train_data.columns:
            # Average shots allowed per opponent
            opp_stats = train_data.groupby('opponent_team_id')['shots'].mean()
            league_avg = train_data['shots'].mean()
            
            for team_id, avg_shots in opp_stats.items():
                # Factor > 1 means easier opponent (allows more shots)
                # Factor < 1 means harder opponent (suppresses shots)
                self.opponent_factors[team_id] = avg_shots / league_avg if league_avg > 0 else 1.0
        
        self.global_mean = train_data['shots'].mean()
        self.global_dispersion = 1.5
    
    def predict_distribution(self, player_id: int,
                            context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict using season mean adjusted for opponent.
        
        Args:
            player_id: Player ID
            context: Dict with 'opponent_team_id'
            
        Returns:
            (mu, alpha) for NB distribution
        """
        base_mu = self.player_means.get(player_id, self.global_mean)
        alpha = self.player_dispersions.get(player_id, self.global_dispersion)
        
        # Apply opponent adjustment
        if context and 'opponent_team_id' in context:
            opponent_id = context['opponent_team_id']
            opponent_factor = self.opponent_factors.get(opponent_id, 1.0)
            mu = base_mu * opponent_factor
        else:
            mu = base_mu
        
        return mu, alpha


class EWMABaseline(BaselineModel):
    """
    Exponentially weighted moving average of recent performance.
    
    Gives more weight to recent games, adapts to hot/cold streaks.
    """
    
    def __init__(self, alpha: float = None):
        """
        Initialize EWMA baseline.
        
        Args:
            alpha: EWMA decay parameter (uses config default if None)
        """
        super().__init__("ewma")
        self.alpha = alpha if alpha is not None else self.config.features.ewma_alpha
        self.player_ewma = {}
        self.player_ewma_var = {}
        self.global_mean = 2.0
        self.global_dispersion = 1.5
        
        # Store training data for EWMA calculation
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Calculate EWMA for each player.
        
        Args:
            train_data: DataFrame with player_id, shots, game_date
        """
        if train_data.empty:
            return
        
        self.train_data = train_data.sort_values(['player_id', 'game_date'])
        
        # Calculate EWMA per player
        for player_id in train_data['player_id'].unique():
            player_games = train_data[train_data['player_id'] == player_id].sort_values('game_date')
            
            if len(player_games) < self.config.features.min_games['player_stats']:
                continue
            
            # Calculate EWMA
            ewma = player_games['shots'].ewm(alpha=self.alpha, adjust=False).mean()
            ewma_var = player_games['shots'].ewm(alpha=self.alpha, adjust=False).var()
            
            # Store most recent values
            self.player_ewma[player_id] = ewma.iloc[-1]
            self.player_ewma_var[player_id] = ewma_var.iloc[-1] if not pd.isna(ewma_var.iloc[-1]) else 2.0
        
        self.global_mean = train_data['shots'].mean()
        self.global_dispersion = 1.5
    
    def predict_distribution(self, player_id: int,
                            context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict using EWMA.
        
        Args:
            player_id: Player ID
            context: Ignored for this baseline
            
        Returns:
            (mu, alpha) for NB distribution
        """
        mu = self.player_ewma.get(player_id, self.global_mean)
        var = self.player_ewma_var.get(player_id, self.global_dispersion)
        
        # Convert variance to dispersion parameter
        if var > mu:
            alpha = (mu ** 2) / (var - mu)
            alpha = np.clip(alpha,
                           self.config.model.nb_min_dispersion,
                           self.config.model.nb_max_dispersion)
        else:
            alpha = self.config.model.nb_max_dispersion
        
        return mu, alpha


class TOIAdjustedEWMA(BaselineModel):
    """
    EWMA baseline adjusted for expected time-on-ice.
    
    Accounts for usage changes (e.g., promoted to first line).
    """
    
    def __init__(self, alpha: float = None):
        """
        Initialize TOI-adjusted EWMA baseline.
        
        Args:
            alpha: EWMA decay parameter
        """
        super().__init__("toi_adjusted_ewma")
        self.alpha = alpha if alpha is not None else self.config.features.ewma_alpha
        self.player_shots_per_60 = {}
        self.player_dispersion = {}
        self.global_shots_per_60 = 8.0
        self.global_dispersion = 1.5
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Calculate shots per 60 minutes using EWMA.
        
        Args:
            train_data: DataFrame with player_id, shots, toi_seconds
        """
        if train_data.empty or 'toi_seconds' not in train_data.columns:
            return
        
        self.train_data = train_data.sort_values(['player_id', 'game_date'])
        
        # Calculate shots per 60 for each player
        for player_id in train_data['player_id'].unique():
            player_games = train_data[train_data['player_id'] == player_id].sort_values('game_date')
            
            if len(player_games) < self.config.features.min_games['player_stats']:
                continue
            
            # Calculate shots per 60
            player_games = player_games.copy()
            player_games['shots_per_60'] = (player_games['shots'] / 
                                            (player_games['toi_seconds'] / 3600))
            
            # Replace inf/nan with 0
            player_games['shots_per_60'] = player_games['shots_per_60'].replace([np.inf, -np.inf], 0)
            player_games['shots_per_60'] = player_games['shots_per_60'].fillna(0)
            
            # Calculate EWMA
            ewma = player_games['shots_per_60'].ewm(alpha=self.alpha, adjust=False).mean()
            
            self.player_shots_per_60[player_id] = ewma.iloc[-1]
            
            # Estimate dispersion from residuals
            player_games['predicted_shots'] = player_games['shots_per_60'] * (player_games['toi_seconds'] / 3600)
            residuals = player_games['shots'] - player_games['predicted_shots']
            var_residuals = residuals.var()
            mean_shots = player_games['shots'].mean()
            
            if var_residuals > mean_shots:
                alpha = (mean_shots ** 2) / (var_residuals - mean_shots)
                alpha = np.clip(alpha,
                               self.config.model.nb_min_dispersion,
                               self.config.model.nb_max_dispersion)
            else:
                alpha = self.config.model.nb_max_dispersion
            
            self.player_dispersion[player_id] = alpha
        
        # Global defaults
        valid_shots_per_60 = train_data[train_data['toi_seconds'] > 0].copy()
        if not valid_shots_per_60.empty:
            valid_shots_per_60['shots_per_60'] = (valid_shots_per_60['shots'] / 
                                                   (valid_shots_per_60['toi_seconds'] / 3600))
            self.global_shots_per_60 = valid_shots_per_60['shots_per_60'].mean()
    
    def predict_distribution(self, player_id: int,
                            context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict using shots per 60 * expected TOI.
        
        Args:
            player_id: Player ID
            context: Dict with 'expected_toi_minutes'
            
        Returns:
            (mu, alpha) for NB distribution
        """
        shots_per_60 = self.player_shots_per_60.get(player_id, self.global_shots_per_60)
        alpha = self.player_dispersion.get(player_id, self.global_dispersion)
        
        # Get expected TOI from context
        if context and 'expected_toi_minutes' in context:
            expected_toi = context['expected_toi_minutes']
        else:
            # Default to 15 minutes
            expected_toi = 15.0
        
        # Calculate expected shots
        mu = shots_per_60 * (expected_toi / 60)
        
        return mu, alpha


class BaselineEnsemble:
    """
    Ensemble of baseline models.
    
    Averages predictions from multiple baselines for more robust benchmarking.
    """
    
    def __init__(self, models: List[BaselineModel]):
        """
        Initialize ensemble.
        
        Args:
            models: List of baseline models to ensemble
        """
        self.models = models
        self.name = "baseline_ensemble"
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit all baseline models."""
        for model in self.models:
            model.fit(train_data)
    
    def predict_distribution(self, player_id: int,
                            context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict by averaging model predictions.
        
        Args:
            player_id: Player ID
            context: Context dict
            
        Returns:
            (mu, alpha) averaged across models
        """
        predictions = [model.predict_distribution(player_id, context) 
                      for model in self.models]
        
        avg_mu = np.mean([pred[0] for pred in predictions])
        avg_alpha = np.mean([pred[1] for pred in predictions])
        
        return avg_mu, avg_alpha
    
    def predict_probabilities(self, player_id: int,
                             context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Predict probabilities by averaging.
        
        Args:
            player_id: Player ID
            context: Context dict
            
        Returns:
            Dict with averaged probabilities
        """
        all_probs = [model.predict_probabilities(player_id, context)
                    for model in self.models]
        
        # Average probabilities for each line
        ensemble_probs = {}
        for key in all_probs[0].keys():
            ensemble_probs[key] = np.mean([probs[key] for probs in all_probs])
        
        return ensemble_probs


def create_baseline_models() -> Dict[str, BaselineModel]:
    """
    Create all baseline models specified in configuration.
    
    Returns:
        Dict mapping baseline name to model instance
    """
    config = get_config()
    models = {}
    
    for baseline_name in config.validation.baselines:
        if baseline_name == 'season_mean':
            models[baseline_name] = SeasonMeanBaseline()
        elif baseline_name == 'ewma_l10':
            models[baseline_name] = EWMABaseline(alpha=0.3)
        elif baseline_name == 'opponent_adjusted_mean':
            models[baseline_name] = OpponentAdjustedBaseline()
        elif baseline_name == 'toi_adjusted_ewma':
            models[baseline_name] = TOIAdjustedEWMA()
    
    return models


def evaluate_baseline(model: BaselineModel, test_data: pd.DataFrame,
                     test_features: pd.DataFrame = None) -> Dict[str, float]:
    """
    Evaluate baseline model on test data.
    
    Args:
        model: Baseline model
        test_data: Test data with actual shots
        test_features: Optional features for context
        
    Returns:
        Dict of evaluation metrics
    """
    from src.validation.metrics import calculate_crps, calculate_brier_score
    
    predictions = []
    actuals = []
    
    for idx, row in test_data.iterrows():
        player_id = row['player_id']
        actual_shots = row['shots']
        
        # Get context if available
        context = None
        if test_features is not None and idx < len(test_features):
            context = {
                'opponent_team_id': test_features.iloc[idx].get('opponent_team_id'),
                'expected_toi_minutes': test_features.iloc[idx].get('toi_per_game_ewma', 15.0)
            }
        
        mu, alpha = model.predict_distribution(player_id, context)
        predictions.append((mu, alpha))
        actuals.append(actual_shots)
    
    # Calculate metrics
    crps_scores = [calculate_crps(actual, mu, alpha) 
                   for actual, (mu, alpha) in zip(actuals, predictions)]
    
    avg_crps = np.mean(crps_scores)
    
    # Calculate Brier scores for common lines
    config = get_config()
    brier_scores = {}
    
    for line in [1.5, 2.5, 3.5, 4.5]:
        brier = []
        for actual, (mu, alpha) in zip(actuals, predictions):
            k = int(np.ceil(line))
            p_over = 1 - stats.nbinom.cdf(k - 1, alpha, alpha / (alpha + mu))
            actual_over = 1 if actual >= k else 0
            brier.append((p_over - actual_over) ** 2)
        
        brier_scores[f'brier_{line}'] = np.mean(brier)
    
    metrics = {
        'crps': avg_crps,
        **brier_scores
    }
    
    return metrics