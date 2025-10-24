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
    
    # Trend/consistency features
    shots_trend_l5_vs_l20: float
    toi_trend_l5_vs_l20: float
    shot_consistency_l10: float
    toi_consistency_l10: float
    
    # Opponent/advantage context
    opponent_rest_days: int
    team_rest_days: int
    rest_advantage: int
    
    # Team form features
    team_shots_for_l5: float
    team_shots_against_l5: float
    team_goals_for_l5: float
    team_goals_against_l5: float
    team_shot_share_l5: float
    team_pace_l5: float
    
    # Opponent form features
    opponent_shots_against_l5: float
    opponent_goals_against_l5: float
    opponent_shot_share_allowed_l5: float
    opponent_pace_allowed_l5: float
    
    # Player-to-team relationships
    player_shot_share_team_l10: float
    player_pp_share_team_l10: float
    
    # Player vs opponent history
    player_vs_opponent_shots_l10: float
    player_vs_opponent_games_l10: int


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
        if not self.historical_data.empty:
            self.team_stats = self._compute_team_stats()
            self.venue_bias = self._compute_venue_bias()
            self.team_game_stats = self._build_team_game_stats()
            self.team_game_stats_by_team = {
                team_id: df.sort_values('game_date')
                for team_id, df in self.team_game_stats.groupby('team_id')
            }
            self.player_history = {
                player_id: df.sort_values('game_date')
                for player_id, df in self.historical_data.groupby('player_id')
            }
        else:
            self.team_stats = {}
            self.venue_bias = {}
            self.team_game_stats = pd.DataFrame()
            self.team_game_stats_by_team = {}
            self.player_history = {}
        
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
    
    def _build_team_game_stats(self) -> pd.DataFrame:
        """Aggregate team-level totals per game for advanced features."""
        if self.historical_data.empty:
            return pd.DataFrame(columns=[
                'team_id', 'game_id', 'game_date', 'opponent_team_id', 'home_away',
                'shots', 'goals', 'toi_seconds', 'pp_toi_seconds', 'sh_toi_seconds',
                'opp_shots', 'opp_goals', 'opp_pp_toi_seconds', 'opp_sh_toi_seconds',
                'shot_share', 'pace'
            ])

        grouped = (
            self.historical_data
            .groupby(['team_id', 'game_id', 'game_date', 'opponent_team_id', 'home_away'], as_index=False)
            .agg({
                'shots': 'sum',
                'goals': 'sum',
                'toi_seconds': 'sum',
                'pp_toi_seconds': 'sum',
                'sh_toi_seconds': 'sum'
            })
        )

        opponent_totals = grouped[['team_id', 'game_id', 'shots', 'goals', 'pp_toi_seconds', 'sh_toi_seconds']].rename(columns={
            'team_id': 'opponent_team_id',
            'shots': 'opp_shots',
            'goals': 'opp_goals',
            'pp_toi_seconds': 'opp_pp_toi_seconds',
            'sh_toi_seconds': 'opp_sh_toi_seconds'
        })

        merged = grouped.merge(
            opponent_totals,
            on=['game_id', 'opponent_team_id'],
            how='left'
        )

        merged['opp_shots'] = merged['opp_shots'].fillna(merged['shots'])
        merged['opp_goals'] = merged['opp_goals'].fillna(merged['goals'])
        merged['opp_pp_toi_seconds'] = merged['opp_pp_toi_seconds'].fillna(0.0)
        merged['opp_sh_toi_seconds'] = merged['opp_sh_toi_seconds'].fillna(0.0)

        total_shots = merged['shots'] + merged['opp_shots']
        merged['shot_share'] = np.where(total_shots > 0, merged['shots'] / total_shots, 0.5)
        merged['pace'] = np.where(np.isnan(total_shots), merged['shots'], total_shots)

        return merged
    
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
    
    def _calculate_team_rest_days(self, team_id: int, game_date: datetime) -> int:
        """
        Calculate rest days for a team prior to the given game date.
        
        Args:
            team_id: Team identifier
            game_date: Upcoming game date
        
        Returns:
            Rest days as integer (defaults to 1 if insufficient history)
        """
        if self.historical_data.empty:
            return 1
        
        team_games = self.historical_data[
            (self.historical_data['team_id'] == team_id) &
            (self.historical_data['game_date'] < game_date)
        ].sort_values('game_date', ascending=False)
        
        if team_games.empty:
            return 1
        
        last_game_date = team_games.iloc[0]['game_date']
        rest_days = (game_date - last_game_date).days
        
        return max(0, rest_days)

    def _get_default_team_form(self) -> Dict[str, float]:
        return {
            'shots_for': 30.0,
            'shots_against': 30.0,
            'goals_for': 3.0,
            'goals_against': 3.0,
            'shot_share': 0.5,
            'pace': 60.0,
        }

    def _get_team_recent_stats(self, team_id: int, as_of_date: datetime,
                               window: int = 5) -> Dict[str, float]:
        team_history = self.team_game_stats_by_team.get(team_id)
        if team_history is None:
            return self._get_default_team_form()
        recent = team_history[team_history['game_date'] < as_of_date].tail(window)
        if recent.empty:
            return self._get_default_team_form()
        shots_for = recent['shots'].mean()
        shots_against = recent['opp_shots'].mean()
        goals_for = recent['goals'].mean()
        goals_against = recent['opp_goals'].mean()
        shot_share = np.clip(recent['shot_share'].mean(), 0.0, 1.0)
        pace = recent['pace'].mean()
        return {
            'shots_for': float(shots_for),
            'shots_against': float(shots_against),
            'goals_for': float(goals_for),
            'goals_against': float(goals_against),
            'shot_share': float(shot_share),
            'pace': float(pace),
        }

    def _get_player_team_share(self, player_id: int, team_id: int,
                               as_of_date: datetime,
                               window: int = 10) -> Dict[str, float]:
        player_games = self.player_history.get(player_id)
        team_history = self.team_game_stats_by_team.get(team_id)
        if player_games is None or team_history is None:
            return {'shot_share': 0.0, 'pp_share': 0.0}
        recent_player = player_games[player_games['game_date'] < as_of_date].tail(window)
        if recent_player.empty:
            return {'shot_share': 0.0, 'pp_share': 0.0}
        team_totals = team_history[['game_id', 'shots', 'pp_toi_seconds']]
        merged = recent_player.merge(team_totals, on='game_id', how='left', suffixes=('', '_team'))
        merged = merged.dropna(subset=['shots_team'])
        if merged.empty:
            return {'shot_share': 0.0, 'pp_share': 0.0}
        total_team_shots = merged['shots_team']
        total_team_pp = merged['pp_toi_seconds_team']

        valid_shots = total_team_shots > 0
        if valid_shots.any():
            shot_share_vals = merged.loc[valid_shots, 'shots'] / total_team_shots.loc[valid_shots]
            shot_share = float(np.clip(shot_share_vals.mean(), 0.0, 1.0))
        else:
            shot_share = 0.0

        valid_pp = total_team_pp > 0
        if valid_pp.any():
            pp_share_vals = merged.loc[valid_pp, 'pp_toi_seconds'] / total_team_pp.loc[valid_pp]
            pp_share = float(np.clip(pp_share_vals.mean(), 0.0, 1.0))
        else:
            pp_share = 0.0

        return {'shot_share': shot_share, 'pp_share': pp_share}

    def _get_player_opponent_history(self, player_id: int, opponent_team_id: int,
                                     as_of_date: datetime, fallback: float,
                                     window: int = 10) -> Dict[str, float]:
        player_games = self.player_history.get(player_id)
        if player_games is None:
            return {'avg_shots': fallback, 'games_played': 0}
        head_to_head = player_games[
            (player_games['opponent_team_id'] == opponent_team_id) &
            (player_games['game_date'] < as_of_date)
        ].tail(window)
        if head_to_head.empty:
            return {'avg_shots': fallback, 'games_played': 0}
        avg_shots = head_to_head['shots'].mean()
        games = len(head_to_head)
        return {'avg_shots': float(avg_shots), 'games_played': int(games)}
    
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
        
        # Ensure we never peek past the prediction timestamp
        cutoff_timestamp = min(game_date, as_of_timestamp)
        
        # Rolling window statistics
        stats_l5 = self.compute_rolling_stats(player_id, cutoff_timestamp, 5)
        stats_l10 = self.compute_rolling_stats(player_id, cutoff_timestamp, 10)
        stats_l20 = self.compute_rolling_stats(player_id, cutoff_timestamp, 20)
        stats_season = self.compute_rolling_stats(player_id, cutoff_timestamp, 999)
        
        # EWMA statistics
        ewma_stats = self.compute_ewma_stats(player_id, cutoff_timestamp)
        
        # Opponent statistics
        opp_stats = self.get_opponent_stats(opponent_team_id, cutoff_timestamp)
        
        # Team context
        team_pace = self.team_stats.get(team_id, {}).get('pace', 50.0)
        team_shot_share = self.team_stats.get(team_id, {}).get('shot_share', 0.50)
        team_form = self._get_team_recent_stats(team_id, cutoff_timestamp, window=5)
        opponent_form = self._get_team_recent_stats(opponent_team_id, cutoff_timestamp, window=5)
        player_team_share = self._get_player_team_share(player_id, team_id, cutoff_timestamp, window=10)
        opponent_history = self._get_player_opponent_history(
            player_id,
            opponent_team_id,
            cutoff_timestamp,
            fallback=stats_season['avg_shots'],
            window=10
        )
        
        opponent_pace = opponent_form['pace'] if opponent_form else self.team_stats.get(opponent_team_id, {}).get('pace', 50.0)
        expected_game_pace = (team_form['pace'] + opponent_form['pace']) / 2 if team_form and opponent_form else (team_pace + opponent_pace) / 2
        
        # Situational
        rest_days = self.calculate_rest_days(player_id, game_date)
        team_rest_days = self._calculate_team_rest_days(team_id, game_date)
        opponent_rest_days = self._calculate_team_rest_days(opponent_team_id, game_date)
        rest_advantage = team_rest_days - opponent_rest_days
        back_to_back = 1 if rest_days == 0 else 0
        
        # Venue bias
        venue_bias = self.venue_bias.get(venue_name, 0.0)
        
        # Role inference
        pp_unit = self.infer_pp_unit(player_id, cutoff_timestamp)
        line_number = self.estimate_line_number(player_id, cutoff_timestamp, position)
        
        # Uncertainty metrics
        games_played = stats_season['games_played']
        lineup_confidence = min(1.0, games_played / 10.0)  # Full confidence after 10 games
        
        projected_toi = ewma_stats['ewma_toi']
        projected_toi_std = stats_l10['std_toi']
        
        # Trend features capture recent momentum
        shots_trend = stats_l5['avg_shots'] - stats_l20['avg_shots']
        toi_trend = stats_l5['avg_toi'] - stats_l20['avg_toi']
        
        shot_consistency = stats_l10['std_shots']
        toi_consistency = stats_l10['std_toi']
        
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
            
            shots_trend_l5_vs_l20=shots_trend,
            toi_trend_l5_vs_l20=toi_trend,
            shot_consistency_l10=shot_consistency,
            toi_consistency_l10=toi_consistency,
            opponent_rest_days=opponent_rest_days,
            team_rest_days=team_rest_days,
            rest_advantage=rest_advantage,
            team_shots_for_l5=team_form['shots_for'],
            team_shots_against_l5=team_form['shots_against'],
            team_goals_for_l5=team_form['goals_for'],
            team_goals_against_l5=team_form['goals_against'],
            team_shot_share_l5=team_form['shot_share'],
            team_pace_l5=team_form['pace'],
            opponent_shots_against_l5=opponent_form['shots_against'],
            opponent_goals_against_l5=opponent_form['goals_against'],
            opponent_shot_share_allowed_l5=1 - opponent_form['shot_share'],
            opponent_pace_allowed_l5=opponent_form['pace'],
            player_shot_share_team_l10=player_team_share['shot_share'],
            player_pp_share_team_l10=player_team_share['pp_share'],
            player_vs_opponent_shots_l10=opponent_history['avg_shots'],
            player_vs_opponent_games_l10=opponent_history['games_played'],
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
            
            # Trend/consistency
            'shots_trend_l5_vs_l20': features.shots_trend_l5_vs_l20,
            'toi_trend_l5_vs_l20': features.toi_trend_l5_vs_l20,
            'shot_consistency_l10': features.shot_consistency_l10,
            'toi_consistency_l10': features.toi_consistency_l10,
            
            # Rest context
            'opponent_rest_days': features.opponent_rest_days,
            'team_rest_days': features.team_rest_days,
            'rest_advantage': features.rest_advantage,
            
            # Team form
            'team_shots_for_l5': features.team_shots_for_l5,
            'team_shots_against_l5': features.team_shots_against_l5,
            'team_goals_for_l5': features.team_goals_for_l5,
            'team_goals_against_l5': features.team_goals_against_l5,
            'team_shot_share_l5': features.team_shot_share_l5,
            'team_pace_l5': features.team_pace_l5,
            
            # Opponent form
            'opponent_shots_against_l5': features.opponent_shots_against_l5,
            'opponent_goals_against_l5': features.opponent_goals_against_l5,
            'opponent_shot_share_allowed_l5': features.opponent_shot_share_allowed_l5,
            'opponent_pace_allowed_l5': features.opponent_pace_allowed_l5,
            
            # Player/team relationships
            'player_shot_share_team_l10': features.player_shot_share_team_l10,
            'player_pp_share_team_l10': features.player_pp_share_team_l10,
            
            # Player vs opponent history
            'player_vs_opponent_shots_l10': features.player_vs_opponent_shots_l10,
            'player_vs_opponent_games_l10': features.player_vs_opponent_games_l10,
        }
        
        return pd.DataFrame([data])

    def bulk_build_features(self, predictions: List[Dict]) -> pd.DataFrame:
        """
        Build features for multiple predictions.
        
        Processes each prediction independently to ensure rolling windows
        respect the provided as_of timestamp and never leak future data.
        """
        if not predictions:
            return pd.DataFrame()
        
        pred_df = pd.DataFrame(predictions)
        feature_frames: List[pd.DataFrame] = []
        
        # Iterate row-wise to ensure each prediction uses history strictly
        # before its game_date. This avoids leakage and the zero-feature bug
        # we observed when future dates were not present in the training history.
        for row in pred_df.itertuples(index=False):
            as_of_ts = getattr(row, 'as_of_timestamp', None)
            venue_name = getattr(row, 'venue_name', 'unknown')
            is_home = getattr(row, 'is_home', False)
            
            player_features = self.build_features(
                player_id=row.player_id,
                player_name=row.player_name,
                position=row.position,
                team_id=row.team_id,
                opponent_team_id=row.opponent_team_id,
                game_id=row.game_id,
                game_date=row.game_date,
                venue_name=venue_name,
                is_home=is_home,
                as_of_timestamp=as_of_ts
            )
            
            feature_df = self.features_to_dataframe(player_features)
            
            # Re-attach contextual identifiers used downstream
            feature_df['team_id'] = row.team_id
            feature_df['opponent_team_id'] = row.opponent_team_id
            feature_df['game_date'] = row.game_date
            feature_df['position'] = row.position
            feature_df['player_name'] = row.player_name
            feature_df['venue_name'] = venue_name
            
            if as_of_ts is not None:
                feature_df['as_of_timestamp'] = as_of_ts
            
            feature_frames.append(feature_df)
        
        features = pd.concat(feature_frames, ignore_index=True)
        
        # Hash venue for categorical representation
        if 'venue_name' in features.columns:
            features['venue_encoded'] = pd.util.hash_array(
                features['venue_name'].fillna('unknown').astype(str).values
            ) % 100
        
        # Add aggregated team/opponent context derived from historical data
        features = self._add_team_features(features)
        
        # Drop non-numeric identifiers before modeling
        features = features.drop(
            columns=['position', 'venue_name', 'player_name'],
            errors='ignore'
        )
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        return features


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
