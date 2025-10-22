"""
Feature engineering for NHL SOG prediction.

Calculates player-level and contextual features from raw game data:
- Usage metrics (TOI, PP/SH time)
- Shooting tendency (iCF/60, iSF/60)
- Matchup context (opponent suppression, pace)
- Situational features (rest, travel, venue)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.utils.config import get_config


@dataclass
class PlayerFeatures:
    """Feature vector for a single player prediction."""
    player_id: int
    player_name: str
    game_id: int
    game_date: datetime
    as_of_timestamp: datetime
    
    # Usage features
    toi_per_game_l5: float
    toi_per_game_l10: float
    toi_per_game_l20: float
    toi_per_game_season: float
    toi_per_game_ewma: float
    
    ev_toi_per_game_l10: float
    pp_toi_per_game_l10: float
    sh_toi_per_game_l10: float
    
    pp_unit: int  # 0=no PP, 1=PP1, 2=PP2
    
    # Shooting tendency
    shots_per_game_l5: float
    shots_per_game_l10: float
    shots_per_game_l20: float
    shots_per_game_season: float
    shots_per_game_ewma: float
    
    icf_per_60_l10: float  # Individual Corsi For
    isf_per_60_l10: float  # Individual Shots For
    shooting_pct_l20: float
    
    # Position and role
    position: str
    is_forward: int
    is_defense: int
    line_number: Optional[int]  # 1st, 2nd, 3rd, 4th line
    
    # Opponent context
    opponent_team_id: int
    opponent_sa_per_60: float  # Shots Against per 60
    opponent_ca_per_60: float  # Corsi Against per 60
    opponent_shot_quality: float  # Average shot distance allowed
    
    # Team context
    team_pace: float  # Team's average Corsi/60
    team_shot_share: float  # Team's Corsi%
    
    # Matchup
    expected_game_pace: float  # Combined team pace
    
    # Situational
    home_away: int  # 1=home, 0=away
    rest_days: int
    back_to_back: int
    travel_distance_mi: float
    
    # Venue
    venue_name: str
    venue_sog_bias: float  # Estimated scorekeeper bias
    
    # Uncertainty flags
    games_played: int
    lineup_confidence: float  # 0-1, how certain are we about role/TOI
    projected_toi: float
    projected_toi_std: float


class FeatureEngineer:
    """
    Computes features for player SOG predictions.
    
    Maintains historical data and calculates rolling statistics,
    opponent adjustments, and contextual features.
    """
    
    def __init__(self, historical_data: pd.DataFrame = None):
        """
        Initialize feature engineer.
        
        Args:
            historical_data: DataFrame with historical player game stats
                Required columns: player_id, game_id, game_date, shots, toi_seconds,
                                  ev_toi_seconds, pp_toi_seconds, team_id, opponent_id
        """
        self.config = get_config()
        self.historical_data = historical_data if historical_data is not None else pd.DataFrame()
        
        # Precompute team-level statistics
        self.team_stats = self._compute_team_stats() if not self.historical_data.empty else {}
        self.venue_bias = self._compute_venue_bias() if not self.historical_data.empty else {}
        
    def _compute_team_stats(self) -> Dict[int, Dict[str, float]]:
        """Compute rolling team-level statistics."""
        team_stats = {}
        
        if self.historical_data.empty:
            return team_stats
        
        # Group by team and compute aggregate metrics
        for team_id in self.historical_data['team_id'].unique():
            team_games = self.historical_data[
                self.historical_data['team_id'] == team_id
            ].sort_values('game_date')
            
            if len(team_games) < self.config.features.min_games['team_stats']:
                continue
            
            # Calculate team pace (Corsi events per 60)
            # Approximate Corsi as shots * 1.5 (shots + misses + blocks)
            team_corsi = team_games['shots'].sum() * 1.5
            team_toi_minutes = team_games['toi_seconds'].sum() / 60
            pace = (team_corsi / team_toi_minutes) * 60 if team_toi_minutes > 0 else 50.0
            
            # Shot share (team shots / total shots in games)
            shot_share = 0.50  # Default to 50%
            
            team_stats[team_id] = {
                'pace': pace,
                'shot_share': shot_share,
                'avg_shots_per_game': team_games['shots'].mean()
            }
        
        return team_stats
    
    def _compute_venue_bias(self) -> Dict[str, float]:
        """
        Compute venue-specific scorekeeper bias.
        
        Some arenas are known to have generous/stingy shot recorders.
        Estimate this from historical SOG distributions.
        """
        venue_bias = {}
        
        if self.historical_data.empty or 'venue_name' not in self.historical_data.columns:
            return venue_bias
        
        # Calculate average SOG per venue relative to league average
        league_avg_sog = self.historical_data['shots'].mean()
        
        for venue in self.historical_data['venue_name'].unique():
            venue_games = self.historical_data[self.historical_data['venue_name'] == venue]
            
            if len(venue_games) < self.config.features.min_games['venue_adjustment']:
                continue
            
            venue_avg_sog = venue_games['shots'].mean()
            bias = venue_avg_sog - league_avg_sog
            
            # Cap bias at reasonable range
            venue_bias[venue] = np.clip(bias, -0.5, 0.5)
        
        return venue_bias
    
    def compute_rolling_stats(self, player_id: int, as_of_date: datetime,
                              window: int) -> Dict[str, float]:
        """
        Compute rolling window statistics for a player.
        
        Args:
            player_id: Player ID
            as_of_date: Calculate stats as of this date (no data after this)
            window: Number of games to include
            
        Returns:
            Dict of rolling statistics
        """
        if self.historical_data.empty:
            return self._get_default_rolling_stats()
        
        # Get player's games before as_of_date
        player_games = self.historical_data[
            (self.historical_data['player_id'] == player_id) &
            (self.historical_data['game_date'] < as_of_date)
        ].sort_values('game_date', ascending=False).head(window)
        
        if player_games.empty:
            return self._get_default_rolling_stats()
        
        # Calculate statistics
        stats = {
            'games_played': len(player_games),
            'avg_toi': player_games['toi_seconds'].mean() / 60,
            'avg_ev_toi': player_games['ev_toi_seconds'].mean() / 60,
            'avg_pp_toi': player_games['pp_toi_seconds'].mean() / 60,
            'avg_sh_toi': player_games['sh_toi_seconds'].mean() / 60,
            'avg_shots': player_games['shots'].mean(),
            'std_shots': player_games['shots'].std() if len(player_games) > 1 else 1.0,
            'std_toi': (player_games['toi_seconds'].std() / 60) if len(player_games) > 1 else 2.0,
        }
        
        # Calculate per-60 rates
        total_toi_minutes = player_games['toi_seconds'].sum() / 60
        if total_toi_minutes > 0:
            stats['shots_per_60'] = (player_games['shots'].sum() / total_toi_minutes) * 60
            stats['icf_per_60'] = (player_games['shots'].sum() * 1.5 / total_toi_minutes) * 60
        else:
            stats['shots_per_60'] = 0.0
            stats['icf_per_60'] = 0.0
        
        return stats
    
    def _get_default_rolling_stats(self) -> Dict[str, float]:
        """Return default statistics when no historical data available."""
        return {
            'games_played': 0,
            'avg_toi': 15.0,
            'avg_ev_toi': 13.0,
            'avg_pp_toi': 1.5,
            'avg_sh_toi': 0.5,
            'avg_shots': 2.0,
            'std_shots': 1.5,
            'std_toi': 2.0,
            'shots_per_60': 8.0,
            'icf_per_60': 12.0,
        }
    
    def compute_ewma_stats(self, player_id: int, as_of_date: datetime) -> Dict[str, float]:
        """
        Compute exponentially weighted moving average statistics.
        
        Args:
            player_id: Player ID
            as_of_date: Calculate stats as of this date
            
        Returns:
            Dict of EWMA statistics
        """
        if self.historical_data.empty:
            return {'ewma_toi': 15.0, 'ewma_shots': 2.0}
        
        player_games = self.historical_data[
            (self.historical_data['player_id'] == player_id) &
            (self.historical_data['game_date'] < as_of_date)
        ].sort_values('game_date', ascending=False)
        
        if player_games.empty:
            return {'ewma_toi': 15.0, 'ewma_shots': 2.0}
        
        alpha = self.config.features.ewma_alpha
        
        # Calculate EWMA
        ewma_toi = player_games['toi_seconds'].ewm(alpha=alpha).mean().iloc[0] / 60
        ewma_shots = player_games['shots'].ewm(alpha=alpha).mean().iloc[0]
        
        return {
            'ewma_toi': ewma_toi,
            'ewma_shots': ewma_shots
        }
    
    def get_opponent_stats(self, opponent_team_id: int, as_of_date: datetime) -> Dict[str, float]:
        """
        Get opponent defensive statistics.
        
        Args:
            opponent_team_id: Opponent team ID
            as_of_date: Calculate stats as of this date
            
        Returns:
            Dict of opponent statistics
        """
        if self.historical_data.empty:
            return {
                'sa_per_60': 30.0,  # League average
                'ca_per_60': 60.0,
                'avg_shot_distance': 35.0
            }
        
        # Get games where opponent_team_id was playing
        opp_games = self.historical_data[
            (self.historical_data['team_id'] == opponent_team_id) &
            (self.historical_data['game_date'] < as_of_date)
        ].tail(20)  # Last 20 games
        
        if opp_games.empty:
            return {
                'sa_per_60': 30.0,
                'ca_per_60': 60.0,
                'avg_shot_distance': 35.0
            }
        
        # Calculate shots against per 60 (approximate)
        total_toi = opp_games['toi_seconds'].sum() / 60
        # This is simplified - would need opponent shot data for exact calculation
        sa_per_60 = 30.0  # Placeholder
        
        return {
            'sa_per_60': sa_per_60,
            'ca_per_60': sa_per_60 * 2,  # Corsi includes misses/blocks
            'avg_shot_distance': 35.0  # Would calculate from shot location data
        }
    
    def calculate_rest_days(self, player_id: int, game_date: datetime) -> int:
        """
        Calculate days of rest before game.
        
        Args:
            player_id: Player ID
            game_date: Date of upcoming game
            
        Returns:
            Number of rest days
        """
        if self.historical_data.empty:
            return 1
        
        player_games = self.historical_data[
            (self.historical_data['player_id'] == player_id) &
            (self.historical_data['game_date'] < game_date)
        ].sort_values('game_date', ascending=False)
        
        if player_games.empty:
            return 1
        
        last_game_date = player_games.iloc[0]['game_date']
        rest_days = (game_date - last_game_date).days
        
        return max(0, rest_days)
    
    def estimate_travel_distance(self, home_venue: str, away_venue: str,
                                  is_home: bool, last_venue: Optional[str]) -> float:
        """
        Estimate travel distance (simplified city-to-city distance).
        
        Args:
            home_venue: Home team venue
            away_venue: Away team venue
            is_home: Is player's team home?
            last_venue: Last venue player played at
            
        Returns:
            Estimated travel distance in miles
        """
        # Simplified venue-to-city mapping
        # In production, would use actual venue coordinates
        venue_cities = {
            'TD Garden': 'Boston',
            'Scotiabank Arena': 'Toronto',
            'Bell Centre': 'Montreal',
            'Madison Square Garden': 'New York',
            'Rogers Arena': 'Vancouver',
            # ... would expand this
        }
        
        # Simplified distance calculation
        # In production, would use geopy or similar
        if last_venue is None:
            return 0.0
        
        # Placeholder logic
        if is_home and last_venue != home_venue:
            return 500.0  # Traveled home
        elif not is_home and last_venue != away_venue:
            return 800.0  # Traveled away
        
        return 0.0
    
    def infer_pp_unit(self, player_id: int, as_of_date: datetime) -> int:
        """
        Infer power play unit (0, 1, or 2) from historical PP TOI.
        
        Args:
            player_id: Player ID
            as_of_date: Date to infer as of
            
        Returns:
            0=no PP, 1=PP1, 2=PP2
        """
        if self.historical_data.empty:
            return 0
        
        player_games = self.historical_data[
            (self.historical_data['player_id'] == player_id) &
            (self.historical_data['game_date'] < as_of_date)
        ].tail(10)
        
        if player_games.empty:
            return 0
        
        avg_pp_toi = player_games['pp_toi_seconds'].mean() / 60
        
        if avg_pp_toi < 0.5:
            return 0
        elif avg_pp_toi >= 3.0:
            return 1  # PP1
        else:
            return 2  # PP2
    
    def estimate_line_number(self, player_id: int, as_of_date: datetime,
                             position: str) -> Optional[int]:
        """
        Estimate line number (1-4) from TOI patterns.
        
        Args:
            player_id: Player ID
            as_of_date: Date to estimate as of
            position: Player position
            
        Returns:
            Line number (1-4) or None
        """
        if position not in ['C', 'LW', 'RW', 'F']:
            return None  # Defensemen don't have "lines" in same way
        
        if self.historical_data.empty:
            return 3  # Default to 3rd line
        
        player_games = self.historical_data[
            (self.historical_data['player_id'] == player_id) &
            (self.historical_data['game_date'] < as_of_date)
        ].tail(10)
        
        if player_games.empty:
            return 3
        
        avg_ev_toi = player_games['ev_toi_seconds'].mean() / 60
        
        if avg_ev_toi >= 16:
            return 1
        elif avg_ev_toi >= 13:
            return 2
        elif avg_ev_toi >= 10:
            return 3
        else:
            return 4
    
    def build_features(self, player_id: int, player_name: str, position: str,
                       team_id: int, opponent_team_id: int, game_id: int,
                       game_date: datetime, venue_name: str, is_home: bool,
                       as_of_timestamp: Optional[datetime] = None) -> PlayerFeatures:
        """
        Build complete feature vector for a player prediction.
        
        Args:
            player_id: NHL player ID
            player_name: Player name
            position: Player position
            team_id: Player's team ID
            opponent_team_id: Opponent team ID
            game_id: Game ID
            game_date: Game date
            venue_name: Venue name
            is_home: Is player's team home?
            as_of_timestamp: Timestamp when prediction made (defaults to now)
            
        Returns:
            PlayerFeatures object
        """
        if as_of_timestamp is None:
            as_of_timestamp = datetime.now()
        
        # Rolling window statistics
        stats_l5 = self.compute_rolling_stats(player_id, game_date, 5)
        stats_l10 = self.compute_rolling_stats(player_id, game_date, 10)
        stats_l20 = self.compute_rolling_stats(player_id, game_date, 20)
        stats_season = self.compute_rolling_stats(player_id, game_date, 999)
        
        # EWMA statistics
        ewma_stats = self.compute_ewma_stats(player_id, game_date)
        
        # Opponent statistics
        opp_stats = self.get_opponent_stats(opponent_team_id, game_date)
        
        # Team context
        team_pace = self.team_stats.get(team_id, {}).get('pace', 50.0)
        team_shot_share = self.team_stats.get(team_id, {}).get('shot_share', 0.50)
        
        opponent_pace = self.team_stats.get(opponent_team_id, {}).get('pace', 50.0)
        expected_game_pace = (team_pace + opponent_pace) / 2
        
        # Situational
        rest_days = self.calculate_rest_days(player_id, game_date)
        back_to_back = 1 if rest_days == 0 else 0
        
        # Venue bias
        venue_bias = self.venue_bias.get(venue_name, 0.0)
        
        # Role inference
        pp_unit = self.infer_pp_unit(player_id, game_date)
        line_number = self.estimate_line_number(player_id, game_date, position)
        
        # Uncertainty metrics
        games_played = stats_season['games_played']
        lineup_confidence = min(1.0, games_played / 10.0)  # Full confidence after 10 games
        
        projected_toi = ewma_stats['ewma_toi']
        projected_toi_std = stats_l10['std_toi']
        
        return PlayerFeatures(
            player_id=player_id,
            player_name=player_name,
            game_id=game_id,
            game_date=game_date,
            as_of_timestamp=as_of_timestamp,
            
            # Usage
            toi_per_game_l5=stats_l5['avg_toi'],
            toi_per_game_l10=stats_l10['avg_toi'],
            toi_per_game_l20=stats_l20['avg_toi'],
            toi_per_game_season=stats_season['avg_toi'],
            toi_per_game_ewma=ewma_stats['ewma_toi'],
            
            ev_toi_per_game_l10=stats_l10['avg_ev_toi'],
            pp_toi_per_game_l10=stats_l10['avg_pp_toi'],
            sh_toi_per_game_l10=stats_l10['avg_sh_toi'],
            
            pp_unit=pp_unit,
            
            # Shooting tendency
            shots_per_game_l5=stats_l5['avg_shots'],
            shots_per_game_l10=stats_l10['avg_shots'],
            shots_per_game_l20=stats_l20['avg_shots'],
            shots_per_game_season=stats_season['avg_shots'],
            shots_per_game_ewma=ewma_stats['ewma_shots'],
            
            icf_per_60_l10=stats_l10['icf_per_60'],
            isf_per_60_l10=stats_l10['shots_per_60'],
            shooting_pct_l20=0.10,  # Would calculate from goals/shots
            
            # Position
            position=position,
            is_forward=1 if position in ['C', 'LW', 'RW', 'F'] else 0,
            is_defense=1 if position == 'D' else 0,
            line_number=line_number,
            
            # Opponent
            opponent_team_id=opponent_team_id,
            opponent_sa_per_60=opp_stats['sa_per_60'],
            opponent_ca_per_60=opp_stats['ca_per_60'],
            opponent_shot_quality=opp_stats['avg_shot_distance'],
            
            # Team context
            team_pace=team_pace,
            team_shot_share=team_shot_share,
            
            # Matchup
            expected_game_pace=expected_game_pace,
            
            # Situational
            home_away=1 if is_home else 0,
            rest_days=rest_days,
            back_to_back=back_to_back,
            travel_distance_mi=0.0,  # Would implement with venue tracking
            
            # Venue
            venue_name=venue_name,
            venue_sog_bias=venue_bias,
            
            # Uncertainty
            games_played=games_played,
            lineup_confidence=lineup_confidence,
            projected_toi=projected_toi,
            projected_toi_std=projected_toi_std,
        )
    
    def features_to_dataframe(self, features: PlayerFeatures) -> pd.DataFrame:
        """
        Convert PlayerFeatures to single-row DataFrame for model input.
        
        Args:
            features: PlayerFeatures object
            
        Returns:
            DataFrame with one row
        """
        data = {
            'player_id': features.player_id,
            'game_id': features.game_id,
            
            # Usage
            'toi_per_game_l5': features.toi_per_game_l5,
            'toi_per_game_l10': features.toi_per_game_l10,
            'toi_per_game_l20': features.toi_per_game_l20,
            'toi_per_game_season': features.toi_per_game_season,
            'toi_per_game_ewma': features.toi_per_game_ewma,
            'ev_toi_per_game_l10': features.ev_toi_per_game_l10,
            'pp_toi_per_game_l10': features.pp_toi_per_game_l10,
            'sh_toi_per_game_l10': features.sh_toi_per_game_l10,
            'pp_unit': features.pp_unit,
            
            # Shooting
            'shots_per_game_l5': features.shots_per_game_l5,
            'shots_per_game_l10': features.shots_per_game_l10,
            'shots_per_game_l20': features.shots_per_game_l20,
            'shots_per_game_season': features.shots_per_game_season,
            'shots_per_game_ewma': features.shots_per_game_ewma,
            'icf_per_60_l10': features.icf_per_60_l10,
            'isf_per_60_l10': features.isf_per_60_l10,
            'shooting_pct_l20': features.shooting_pct_l20,
            
            # Position
            'is_forward': features.is_forward,
            'is_defense': features.is_defense,
            'line_number': features.line_number if features.line_number else 3,
            
            # Opponent
            'opponent_sa_per_60': features.opponent_sa_per_60,
            'opponent_ca_per_60': features.opponent_ca_per_60,
            'opponent_shot_quality': features.opponent_shot_quality,
            
            # Context
            'team_pace': features.team_pace,
            'team_shot_share': features.team_shot_share,
            'expected_game_pace': features.expected_game_pace,
            
            # Situational
            'home_away': features.home_away,
            'rest_days': features.rest_days,
            'back_to_back': features.back_to_back,
            'venue_sog_bias': features.venue_sog_bias,
            
            # Uncertainty
            'games_played': features.games_played,
            'lineup_confidence': features.lineup_confidence,
            'projected_toi': features.projected_toi,
            'projected_toi_std': features.projected_toi_std,
        }
        
        return pd.DataFrame([data])

    def bulk_build_features(self, predictions: List[Dict]) -> pd.DataFrame:
        """
        Build features for multiple predictions using TRUE vectorization.
        
        Key insight: Pre-compute rolling stats for ALL player-games at once,
        then filter to just the ones we need.
        """
        if not predictions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Ensure unique identifier
        if 'game_id' not in pred_df.columns:
            pred_df['game_id'] = (pred_df['player_id'].astype(str) + '_' + 
                                pred_df['game_date'].astype(str))
        
        # Pre-compute ALL rolling stats for relevant players
        player_stats = self._precompute_all_player_stats(
            pred_df['player_id'].unique(),
            pred_df['game_date'].min(),
            pred_df['game_date'].max()
        )
        
        # Merge player stats
        features = pred_df.merge(
            player_stats,
            on=['player_id', 'game_date'],
            how='left'
        )
        
        # Add simple aggregated team/opponent features
        features = self._add_team_features(features)
        
        # Add static features
        features['is_defense'] = (features['position'] == 'D').astype(int)
        features['is_forward'] = (features['position'].isin(['C', 'LW', 'RW'])).astype(int)
        features['venue_encoded'] = pd.util.hash_array(
            features.get('venue_name', pd.Series(['unknown'] * len(features))).values
        ) % 100
        
        # Drop non-numeric
        features = features.drop(
            columns=['position', 'venue_name', 'player_name'], 
            errors='ignore'
        )
        
        # Fill missing
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        return features


    def _precompute_all_player_stats(self, player_ids: np.ndarray, 
                                    min_date, max_date) -> pd.DataFrame:
        """
        Pre-compute rolling stats for all players at once.
        
        This is the key optimization: compute stats for ALL games for ALL players
        in one pass, then filter to what we need.
        """
        # Get relevant history
        history = self.historical_data[
            (self.historical_data['player_id'].isin(player_ids)) &
            (self.historical_data['game_date'] <= max_date)
        ].copy()
        
        if len(history) == 0:
            return pd.DataFrame()
        
        # Sort for rolling calculations
        history = history.sort_values(['player_id', 'game_date'])
        
        # Compute rolling stats using groupby + rolling (vectorized!)
        history['shots_per_game_l10'] = (
            history.groupby('player_id')['shots']
            .rolling(10, min_periods=1).mean()
            .reset_index(0, drop=True)
        )
        
        history['shots_per_game_l20'] = (
            history.groupby('player_id')['shots']
            .rolling(20, min_periods=1).mean()
            .reset_index(0, drop=True)
        )
        
        history['shots_per_game_season'] = (
            history.groupby('player_id')['shots']
            .expanding(min_periods=1).mean()
            .reset_index(0, drop=True)
        )
        
        history['toi_per_game_l10'] = (
            history.groupby('player_id')['toi_seconds']
            .rolling(10, min_periods=1).mean()
            .reset_index(0, drop=True)
        ) / 60
        
        history['toi_per_game_l20'] = (
            history.groupby('player_id')['toi_seconds']
            .rolling(20, min_periods=1).mean()
            .reset_index(0, drop=True)
        ) / 60
        
        history['toi_per_game_season'] = (
            history.groupby('player_id')['toi_seconds']
            .expanding(min_periods=1).mean()
            .reset_index(0, drop=True)
        ) / 60
        
        history['games_played'] = (
            history.groupby('player_id').cumcount() + 1
        )
        
        # Keep only needed columns
        return history[[
            'player_id', 'game_date',
            'shots_per_game_l10', 'shots_per_game_l20', 'shots_per_game_season',
            'toi_per_game_l10', 'toi_per_game_l20', 'toi_per_game_season',
            'games_played'
        ]]


    def _add_team_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple team/opponent aggregates.
        
        Uses historical data to compute team offensive and opponent defensive stats.
        """
        # Get unique team-date combinations we need
        team_dates = pd.concat([
            features[['team_id', 'game_date']],
            features[['opponent_team_id', 'game_date']].rename(columns={'opponent_team_id': 'team_id'})
        ]).drop_duplicates()
        
        # Compute team stats for these dates
        team_stats = []
        for _, row in team_dates.iterrows():
            team_history = self.historical_data[
                (self.historical_data['team_id'] == row['team_id']) &
                (self.historical_data['game_date'] < row['game_date'])
            ].tail(10)  # Last 10 games
            
            if len(team_history) > 0:
                team_stats.append({
                    'team_id': row['team_id'],
                    'game_date': row['game_date'],
                    'team_shots_per_game': team_history.groupby('game_date')['shots'].sum().mean(),
                    'team_goals_per_game': team_history.groupby('game_date')['goals'].sum().mean()
                })
        
        team_stats_df = pd.DataFrame(team_stats)
        
        # Merge team stats
        features = features.merge(
            team_stats_df,
            on=['team_id', 'game_date'],
            how='left'
        )
        
        # Merge opponent stats (defensive)
        features = features.merge(
            team_stats_df.rename(columns={
                'team_shots_per_game': 'opp_shots_against_per_game',
                'team_goals_per_game': 'opp_goals_against_per_game'
            }),
            left_on=['opponent_team_id', 'game_date'],
            right_on=['team_id', 'game_date'],
            how='left',
            suffixes=('', '_opp')
        )
        
        # Drop duplicate team_id column
        features = features.drop(columns=['team_id_opp'], errors='ignore')
        
        return features