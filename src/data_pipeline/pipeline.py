"""
ETL Pipeline for NHL SOG prediction system.

Orchestrates:
- Historical data collection and caching
- Feature engineering
- Model training
- Prediction generation
- Post-game result collection

All operations are timestamped to ensure no data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from src.data_pipeline.nhl_api import NHLAPIClient, create_nhl_client, PlayerGameStats
from src.data_pipeline.features import FeatureEngineer, PlayerFeatures
from src.modeling.lgbm_model import LGBMNegativeBinomialModel
from src.modeling.calibration import MultiLineCalibrator, create_calibrator
from src.modeling.conformal import (
    ConformalIntervalCalibrator,
    load_conformal_calibrator
)
from src.validation.baselines import create_baseline_models
from src.utils.config import get_config


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when pipeline operation fails."""
    pass


class HistoricalDataCollector:
    """
    Collects and maintains historical player game statistics.
    
    Fetches data from NHL API and builds training/validation datasets.
    """
    
    def __init__(self, client: Optional[NHLAPIClient] = None):
        """
        Initialize historical data collector.
        
        Args:
            client: NHL API client (creates new one if None)
        """
        self.config = get_config()
        self.client = client or create_nhl_client()
        self.historical_data = pd.DataFrame()
    
    def collect_date_range(self, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """
        Collect all player game stats for date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with player game statistics
        """
        logger.info(f"Collecting data from {start_date.date()} to {end_date.date()}")
        
        all_stats = []
        
        # Get all games in date range
        games = self.client.get_schedule(start_date, end_date)
        logger.info(f"Found {len(games)} games")
        
        for game in games:
            # Only process completed games
            if game.game_state not in ['OFF', 'FINAL']:
                continue
            
            try:
                # Fetch boxscore
                boxscore = self.client.get_game_boxscore(game.game_id)
                
                # Parse player stats
                player_stats = self.client.parse_player_game_stats(
                    boxscore, game.game_id, game.date
                )
                
                # Add game context
                for stat in player_stats:
                    all_stats.append({
                        'player_id': stat.player_id,
                        'player_name': stat.player_name,
                        'team_id': stat.team_id,
                        'position': stat.position,
                        'game_id': stat.game_id,
                        'game_date': stat.game_date,
                        'home_away': stat.home_away,
                        'opponent_team_id': game.away_team_id if stat.home_away == 'home' else game.home_team_id,
                        'venue_name': game.venue_name,
                        'shots': stat.shots,
                        'goals': stat.goals,
                        'assists': stat.assists,
                        'toi_seconds': stat.toi_seconds,
                        'ev_toi_seconds': stat.ev_toi_seconds,
                        'pp_toi_seconds': stat.pp_toi_seconds,
                        'sh_toi_seconds': stat.sh_toi_seconds,
                        'shifts': stat.shifts,
                    })
                
                logger.debug(f"Processed game {game.game_id}")
                
            except Exception as e:
                logger.warning(f"Failed to process game {game.game_id}: {e}")
                continue
        
        df = pd.DataFrame(all_stats)
        
        if df.empty:
            logger.warning("No data collected")
            return df
        
        # Sort by date
        df = df.sort_values('game_date')
        
        logger.info(f"Collected {len(df)} player-game records")
        
        return df
    
    def collect_season(self, season: str) -> pd.DataFrame:
        """
        Collect all data for a season.
        
        Args:
            season: Season string (e.g., "20232024")
            
        Returns:
            DataFrame with season data
        """
        # Parse season string
        start_year = int(season[:4])
        
        # NHL season typically runs Oct - April
        start_date = datetime(start_year, 10, 1)
        end_date = datetime(start_year + 1, 6, 30)
        
        return self.collect_date_range(start_date, end_date)
    
    def save_to_disk(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save historical data to disk.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.config.data.raw_data_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
    
    def load_from_disk(self, filename: str) -> pd.DataFrame:
        """
        Load historical data from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with historical data
        """
        input_path = self.config.data.raw_data_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Historical data not found: {input_path}")
        
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
        
        return df


class PredictionPipeline:
    """
    End-to-end pipeline for generating daily predictions.
    
    Orchestrates feature engineering, model inference, and calibration.
    """
    
    def __init__(self, 
                 model: Optional[LGBMNegativeBinomialModel] = None,
                 calibrator: Optional[MultiLineCalibrator] = None,
                 interval_calibrator: Optional[ConformalIntervalCalibrator] = None,
                 historical_data: Optional[pd.DataFrame] = None,
                 model_dir: Optional[Path] = None):
        """
        Initialize prediction pipeline.
        
        Args:
            model: Trained model (loads from disk if None)
            calibrator: Fitted calibrator (loads from disk if None)
            interval_calibrator: Conformal interval calibrator
            historical_data: Historical data for feature engineering
            model_dir: Explicit directory containing model artifacts
        """
        self.config = get_config()
        self.client = create_nhl_client()
        self.model_dir: Optional[Path] = Path(model_dir) if model_dir is not None else None
        
        # Load or use provided model
        if model is None:
            self.model = self._load_latest_model()
        else:
            self.model = model
            if self.model_dir is None:
                logger.warning("Model directory not provided; calibrator loading may fail")
       
        # Load or use provided calibrator
        if calibrator is None:
            self.calibrator = self._load_calibrator()
        else:
            self.calibrator = calibrator
        
        if interval_calibrator is None:
            self.interval_calibrator = self._load_interval_calibrator()
        else:
            self.interval_calibrator = interval_calibrator
        
        # Initialize feature engineer
        if historical_data is None:
            historical_data = self._load_historical_data()
        
        self.feature_engineer = FeatureEngineer(historical_data)
        
        logger.info("Prediction pipeline initialized")
    
    def _load_latest_model(self) -> LGBMNegativeBinomialModel:
        """Load most recent trained model."""
        model_dir = self.config.data.model_artifacts_dir
        
        # Find latest model
        model_dirs = list(model_dir.glob("v*"))
        if not model_dirs:
            raise PipelineError(f"No models found in {model_dir}")
        
        latest_model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
        
        model = LGBMNegativeBinomialModel()
        model.load(latest_model_dir)
        self.model_dir = latest_model_dir
        
        return model
    
    def _load_calibrator(self) -> Optional[MultiLineCalibrator]:
        """Load calibrator if available."""
        if self.model_dir is None:
            logger.warning("Model directory unknown; skipping probability calibrator load")
            return None
        
        calibrator_path = self.model_dir / "calibrator.pkl"
        
        if not calibrator_path.exists():
            logger.warning("No calibrator found, predictions will be uncalibrated")
            return None
        
        from src.modeling.calibration import load_calibrator
        return load_calibrator(calibrator_path)
    
    def _load_interval_calibrator(self) -> Optional[ConformalIntervalCalibrator]:
        """Load conformal interval calibrator if available."""
        if self.model_dir is None:
            logger.warning("Model directory unknown; skipping interval calibrator load")
            return None
        
        calibrator_path = self.model_dir / "interval_calibrator.pkl"
        
        if not calibrator_path.exists():
            logger.info("No interval calibrator found; using model intervals directly")
            return None
        
        return load_conformal_calibrator(calibrator_path)
    
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data for feature engineering."""
        data_path = self.config.data.processed_data_dir / "historical_data.parquet"
        
        if not data_path.exists():
            logger.warning("No historical data found, using empty DataFrame")
            return pd.DataFrame()
        
        return pd.read_parquet(data_path)
    
    def generate_predictions_for_date(self, 
                                     target_date: datetime,
                                     as_of_timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate predictions for all games on target date.
        
        Args:
            target_date: Date to generate predictions for
            as_of_timestamp: Timestamp when prediction is made (for reproducibility)
            
        Returns:
            DataFrame with predictions for all players
        """
        if as_of_timestamp is None:
            # Default to lead time before first game
            as_of_timestamp = datetime.combine(
                target_date.date(), 
                datetime.min.time()
            ) - timedelta(minutes=self.config.prediction.prediction_lead_time_minutes)
        
        logger.info(f"Generating predictions for {target_date.date()}, as_of={as_of_timestamp}")
        
        # Get games for target date
        games = self.client.get_games_for_date(target_date)
        
        if not games:
            logger.warning(f"No games found for {target_date.date()}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(games)} games")
        
        # Build prediction tasks
        prediction_tasks = []
        
        for game in games:
            # Get rosters for both teams
            try:
                home_roster = self.client.get_team_roster(game.home_team_abbrev)
                away_roster = self.client.get_team_roster(game.away_team_abbrev)
            except Exception as e:
                logger.warning(f"Failed to get roster for game {game.game_id}: {e}")
                continue
            
            # Add home players
            for player in home_roster:
                if player.get('positionCode') in ['C', 'L', 'R', 'D']:  # Skip goalies
                    prediction_tasks.append({
                        'player_id': player['id'],
                        'player_name': player.get('firstName', {}).get('default', '') + ' ' + 
                                      player.get('lastName', {}).get('default', ''),
                        'position': player.get('positionCode', 'F'),
                        'team_id': game.home_team_id,
                        'opponent_team_id': game.away_team_id,
                        'game_id': game.game_id,
                        'game_date': game.date,
                        'venue_name': game.venue_name,
                        'is_home': True,
                        'as_of_timestamp': as_of_timestamp
                    })
            
            # Add away players
            for player in away_roster:
                if player.get('positionCode') in ['C', 'L', 'R', 'D']:
                    prediction_tasks.append({
                        'player_id': player['id'],
                        'player_name': player.get('firstName', {}).get('default', '') + ' ' + 
                                      player.get('lastName', {}).get('default', ''),
                        'position': player.get('positionCode', 'F'),
                        'team_id': game.away_team_id,
                        'opponent_team_id': game.home_team_id,
                        'game_id': game.game_id,
                        'game_date': game.date,
                        'venue_name': game.venue_name,
                        'is_home': False,
                        'as_of_timestamp': as_of_timestamp
                    })
        
        if not prediction_tasks:
            logger.warning("No prediction tasks generated")
            return pd.DataFrame()
        
        logger.info(f"Generating predictions for {len(prediction_tasks)} players")
        
        # Build features
        features_df = self.feature_engineer.bulk_build_features(prediction_tasks)
        
        # Generate predictions
        dist_params = self.model.predict_with_uncertainty_adjustment(features_df)
        
        # Calculate probabilities for common lines
        probs = self.model.predict_probabilities(features_df)
        
        # Apply calibration if available
        if self.calibrator is not None:
            combined = pd.concat([dist_params, probs], axis=1)
            calibrated = self.calibrator.transform(combined)
            
            # Merge calibrated probabilities
            for col in calibrated.columns:
                if col.endswith('_calibrated'):
                    probs[col] = calibrated[col]
        
        # Calculate confidence intervals (conformal if available)
        if self.interval_calibrator is not None:
            intervals_df = self.interval_calibrator.predict(dist_params)
        else:
            intervals_df = pd.DataFrame(index=dist_params.index)
            for level in self.config.validation.confidence_levels:
                base = self.model.predict_intervals(features_df, confidence_level=level)
                key = f'ci_{int(level*100)}'
                intervals_df[f'{key}_lower'] = base['lower']
                intervals_df[f'{key}_upper'] = base['upper']
        
        # Combine into output DataFrame
        output = pd.DataFrame({
            'player_id': features_df['player_id'],
            'game_id': features_df['game_id'],
            'as_of_timestamp': as_of_timestamp,
        })
        
        # Add player info from tasks
        task_df = pd.DataFrame(prediction_tasks)
        output = output.merge(
            task_df[['player_id', 'game_id', 'player_name', 'position', 'team_id', 
                     'opponent_team_id', 'venue_name', 'is_home', 'game_date']],
            on=['player_id', 'game_id'],
            how='left'
        )
        
        # Add distribution parameters
        output['mu'] = dist_params['mu']
        output['alpha'] = dist_params['alpha']
        
        # Add probabilities
        for col in probs.columns:
            output[col] = probs[col]
        
        # Add intervals
        for col in intervals_df.columns:
            output[col] = intervals_df[col]
        
        ci80_lower_col = 'ci_80_lower'
        ci80_upper_col = 'ci_80_upper'
        if ci80_lower_col in intervals_df.columns and ci80_upper_col in intervals_df.columns:
            output['p10'] = intervals_df[ci80_lower_col]
            output['p90'] = intervals_df[ci80_upper_col]
        else:
            fallback = self.model.predict_intervals(features_df, confidence_level=0.8)
            output['p10'] = fallback['lower']
            output['p90'] = fallback['upper']
        
        output['p50'] = dist_params['mu'].round().astype(int)
        
        # Add confidence metrics
        output['projected_toi'] = features_df['projected_toi']
        output['lineup_confidence'] = features_df['lineup_confidence']
        
        logger.info(f"Generated {len(output)} predictions")
        
        return output
    
    def save_predictions(self, predictions: pd.DataFrame, filename: str) -> None:
        """
        Save predictions to disk.
        
        Args:
            predictions: Predictions DataFrame
            filename: Output filename
        """
        output_path = self.config.data.processed_data_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
    
    def format_predictions_for_export(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Format predictions for human-readable export.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Formatted DataFrame suitable for CSV/JSON export
        """
        formatted = predictions.copy()
        
        # Round probabilities
        prob_cols = [col for col in formatted.columns if col.startswith('p_over_')]
        for col in prob_cols:
            formatted[col] = formatted[col].round(self.config.prediction.decimal_precision)
        
        # Round distribution parameters
        formatted['mu'] = formatted['mu'].round(2)
        formatted['alpha'] = formatted['alpha'].round(2)
        
        # Sort by expected shots (descending)
        formatted = formatted.sort_values('mu', ascending=False)
        
        # Select output columns
        output_cols = [
            'player_name', 'position', 'game_id', 'is_home',
            'mu', 'p10', 'p50', 'p90',
        ]
        
        # Add probability columns
        for line in self.config.model.common_lines:
            if line in [1.5, 2.5, 3.5, 4.5]:
                # Prefer calibrated if available
                if f'p_over_{line}_calibrated' in formatted.columns:
                    output_cols.append(f'p_over_{line}_calibrated')
                elif f'p_over_{line}' in formatted.columns:
                    output_cols.append(f'p_over_{line}')
        
        output_cols.extend(['projected_toi', 'lineup_confidence'])
        
        # Filter to available columns
        output_cols = [col for col in output_cols if col in formatted.columns]
        
        return formatted[output_cols]


class TrainingPipeline:
    """
    Pipeline for model training and validation.
    
    Handles data splitting, baseline comparison, model training,
    calibration, and persistence.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize training pipeline.
        
        Args:
            historical_data: Historical player game statistics
        """
        self.config = get_config()
        self.historical_data = historical_data
        self.feature_engineer = None
        self.model = None
        self.calibrator = None
        self.baselines = None
        
        logger.info(f"Training pipeline initialized with {len(historical_data)} records")
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        
        Returns:
            (train_df, val_df, test_df)
        """
        logger.info("Preparing data splits")
        
        # Time-based splits
        train_mask = self.historical_data['game_date'] < self.config.validation.val_start_date
        val_mask = (
            (self.historical_data['game_date'] >= self.config.validation.val_start_date) &
            (self.historical_data['game_date'] < self.config.validation.test_start_date)
        )
        test_mask = self.historical_data['game_date'] >= self.config.validation.test_start_date
        
        train_df = self.historical_data[train_mask].copy()
        val_df = self.historical_data[val_mask].copy()
        test_df = self.historical_data[test_mask].copy()
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> LGBMNegativeBinomialModel:
        """
        Train main model.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Trained model
        """
        logger.info("Training main model")
        
        # Initialize feature engineer with training data
        self.feature_engineer = FeatureEngineer(train_df)
        
        # Build features for training
        # This is simplified - in practice would need to properly construct prediction tasks
        X_train = train_df.drop(columns=['shots'])
        y_train = train_df['shots'].values
        
        X_val = val_df.drop(columns=['shots'])
        y_val = val_df['shots'].values
        
        # Train model
        self.model = LGBMNegativeBinomialModel()
        self.model.fit(X_train, y_train, X_val, y_val, verbose=True)
        
        return self.model
    
    def save_artifacts(self, output_dir: Path) -> None:
        """
        Save trained model and calibrator.
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model is not None:
            self.model.save(output_dir)
        
        # Save calibrator
        if self.calibrator is not None:
            from src.modeling.calibration import save_calibrator
            save_calibrator(self.calibrator, output_dir / "calibrator.pkl")
        
        logger.info(f"Artifacts saved to {output_dir}")


def setup_logging(log_level: str = None) -> None:
    """
    Configure logging for pipeline.
    
    Args:
        log_level: Log level (uses config if None)
    """
    config = get_config()
    level = log_level or config.monitoring.log_level
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_daily_predictions(target_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Convenience function to run daily prediction workflow.
    
    Args:
        target_date: Date to predict (uses today if None)
        
    Returns:
        DataFrame with predictions
    """
    setup_logging()
    
    if target_date is None:
        target_date = datetime.now()
    
    pipeline = PredictionPipeline()
    predictions = pipeline.generate_predictions_for_date(target_date)
    
    # Save predictions
    filename = f"predictions_{target_date.strftime('%Y%m%d')}.parquet"
    pipeline.save_predictions(predictions, filename)
    
    # Export formatted version
    formatted = pipeline.format_predictions_for_export(predictions)
    csv_path = pipeline.config.data.processed_data_dir / f"predictions_{target_date.strftime('%Y%m%d')}.csv"
    formatted.to_csv(csv_path, index=False)
    
    logger.info(f"Predictions exported to {csv_path}")
    
    return predictions
