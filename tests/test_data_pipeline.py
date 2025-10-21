"""
Unit tests for data pipeline components.

Tests NHL API client, feature engineering, and ETL pipeline.

Run with:
    pytest tests/test_data_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.data_pipeline.nhl_api import (
    NHLAPIClient, GameInfo, PlayerGameStats, NHLAPIError
)
from src.data_pipeline.features import FeatureEngineer, PlayerFeatures
from src.data_pipeline.pipeline import (
    HistoricalDataCollector, PredictionPipeline, setup_logging
)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def setup_test_config():
    """Load configuration for tests."""
    load_config("config/model_config.yaml")


@pytest.fixture
def mock_api_response():
    """Mock NHL API responses."""
    return {
        'gameWeek': [{
            'games': [{
                'id': 2024020123,
                'season': '20242025',
                'gameType': 'REG',
                'gameDate': '2024-11-15T19:00:00Z',
                'homeTeam': {
                    'id': 22,
                    'abbrev': 'EDM'
                },
                'awayTeam': {
                    'id': 6,
                    'abbrev': 'BOS'
                },
                'venue': {
                    'default': 'Rogers Place'
                },
                'gameState': 'FINAL'
            }]
        }]
    }


@pytest.fixture
def mock_boxscore():
    """Mock boxscore data."""
    return {
        'homeTeam': {
            'id': 22
        },
        'awayTeam': {
            'id': 6
        },
        'playerByGameStats': {
            'homeTeam': {
                'forwards': [{
                    'playerId': 8478402,
                    'name': {'default': 'Connor McDavid'},
                    'position': 'C',
                    'goals': 2,
                    'assists': 1,
                    'shots': 5,
                    'toi': '21:35',
                    'evenStrengthToi': '17:20',
                    'powerPlayToi': '3:45',
                    'shorthandedToi': '0:30',
                    'shifts': 28
                }],
                'defense': []
            },
            'awayTeam': {
                'forwards': [],
                'defense': []
            }
        }
    }


@pytest.fixture
def sample_historical_data():
    """Create sample historical data."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-10-01', '2024-11-30', freq='D')
    player_ids = [8478402, 8479318, 8480012]
    
    data = []
    for player_id in player_ids:
        for date in dates:
            data.append({
                'player_id': player_id,
                'player_name': f'Player_{player_id}',
                'team_id': 22,
                'position': 'C',
                'game_id': int(date.strftime('%Y%m%d')),
                'game_date': date,
                'home_away': np.random.choice(['home', 'away']),
                'opponent_team_id': np.random.choice([1, 2, 3, 4, 5]),
                'venue_name': 'Test Arena',
                'shots': np.random.poisson(3),
                'goals': np.random.poisson(0.3),
                'assists': np.random.poisson(0.5),
                'toi_seconds': np.random.normal(1200, 180),
                'ev_toi_seconds': np.random.normal(1000, 150),
                'pp_toi_seconds': np.random.normal(120, 30),
                'sh_toi_seconds': np.random.normal(30, 10),
                'shifts': np.random.poisson(25),
            })
    
    return pd.DataFrame(data)


class TestNHLAPIClient:
    """Tests for NHL API client."""
    
    def test_initialization(self, setup_test_config):
        """Test client initializes correctly."""
        client = NHLAPIClient()
        
        assert client.base_url is not None
        assert client.rate_limit > 0
        assert client.timeout > 0
        assert client.session is not None
    
    def test_cache_key_generation(self, setup_test_config):
        """Test cache key generation."""
        client = NHLAPIClient()
        
        key1 = client._get_cache_key('/endpoint', {'param': 'value'})
        key2 = client._get_cache_key('/endpoint', {'param': 'value'})
        key3 = client._get_cache_key('/endpoint', {'param': 'different'})
        
        assert key1 == key2  # Same params should give same key
        assert key1 != key3  # Different params should give different key
    
    def test_toi_conversion(self, setup_test_config):
        """Test TOI string to seconds conversion."""
        client = NHLAPIClient()
        
        assert client._toi_to_seconds('21:35') == 1295
        assert client._toi_to_seconds('0:30') == 30
        assert client._toi_to_seconds('0:00') == 0
        assert client._toi_to_seconds('') == 0
        assert client._toi_to_seconds(None) == 0
    
    @patch('src.data_pipeline.nhl_api.requests.Session.get')
    def test_successful_request(self, mock_get, setup_test_config, mock_api_response):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = NHLAPIClient(cache_enabled=False)
        result = client._request('/test-endpoint')
        
        assert result == mock_api_response
        mock_get.assert_called_once()
    
    @patch('src.data_pipeline.nhl_api.requests.Session.get')
    def test_request_failure(self, mock_get, setup_test_config):
        """Test API request failure handling."""
        mock_get.side_effect = Exception("Network error")
        
        client = NHLAPIClient(cache_enabled=False)
        
        with pytest.raises(NHLAPIError):
            client._request('/test-endpoint')
    
    def test_parse_player_game_stats(self, setup_test_config, mock_boxscore):
        """Test parsing boxscore into PlayerGameStats."""
        client = NHLAPIClient()
        game_id = 2024020123
        game_date = datetime(2024, 11, 15)
        
        stats = client.parse_player_game_stats(mock_boxscore, game_id, game_date)
        
        assert len(stats) == 1
        assert isinstance(stats[0], PlayerGameStats)
        assert stats[0].player_id == 8478402
        assert stats[0].shots == 5
        assert stats[0].goals == 2
        assert stats[0].toi_seconds == 1295
    
    def test_extract_shots_from_pbp(self, setup_test_config):
        """Test extracting shot events from play-by-play."""
        client = NHLAPIClient()
        
        pbp_data = {
            'plays': [
                {
                    'eventId': 1,
                    'typeDescKey': 'shot-on-goal',
                    'periodDescriptor': {'number': 1},
                    'timeInPeriod': '5:23',
                    'situationCode': '5v5',
                    'details': {
                        'shotType': 'Wrist',
                        'shootingPlayerId': 8478402,
                        'xCoord': 75,
                        'yCoord': 10
                    }
                },
                {
                    'eventId': 2,
                    'typeDescKey': 'faceoff',  # Not a shot
                }
            ]
        }
        
        shots = client.extract_shots_from_pbp(pbp_data)
        
        assert len(shots) == 1
        assert shots[0]['event_type'] == 'shot-on-goal'
        assert shots[0]['shooter_player_id'] == 8478402


class TestFeatureEngineer:
    """Tests for feature engineering."""
    
    def test_initialization(self, setup_test_config, sample_historical_data):
        """Test feature engineer initializes."""
        engineer = FeatureEngineer(sample_historical_data)
        
        assert engineer.historical_data is not None
        assert len(engineer.historical_data) > 0
        assert isinstance(engineer.team_stats, dict)
        assert isinstance(engineer.venue_bias, dict)
    
    def test_compute_rolling_stats_with_data(self, setup_test_config, sample_historical_data):
        """Test rolling stats with sufficient data."""
        engineer = FeatureEngineer(sample_historical_data)
        
        player_id = 8478402
        as_of_date = datetime(2024, 11, 15)
        
        stats = engineer.compute_rolling_stats(player_id, as_of_date, window=10)
        
        assert 'games_played' in stats
        assert stats['games_played'] > 0
        assert stats['games_played'] <= 10
        assert stats['avg_toi'] > 0
        assert stats['avg_shots'] >= 0
        assert stats['shots_per_60'] >= 0
    
    def test_compute_rolling_stats_no_data(self, setup_test_config):
        """Test rolling stats with no historical data."""
        engineer = FeatureEngineer(pd.DataFrame())
        
        stats = engineer.compute_rolling_stats(9999999, datetime.now(), window=10)
        
        # Should return defaults
        assert stats['games_played'] == 0
        assert stats['avg_toi'] == 15.0  # Default
    
    def test_ewma_calculation(self, setup_test_config, sample_historical_data):
        """Test EWMA statistics."""
        engineer = FeatureEngineer(sample_historical_data)
        
        ewma = engineer.compute_ewma_stats(8478402, datetime(2024, 11, 15))
        
        assert 'ewma_toi' in ewma
        assert 'ewma_shots' in ewma
        assert ewma['ewma_toi'] > 0
        assert ewma['ewma_shots'] >= 0
    
    def test_rest_days_calculation(self, setup_test_config, sample_historical_data):
        """Test rest days calculation."""
        engineer = FeatureEngineer(sample_historical_data)
        
        # Player has games in consecutive days
        rest_days = engineer.calculate_rest_days(8478402, datetime(2024, 11, 15))
        
        assert isinstance(rest_days, int)
        assert rest_days >= 0
    
    def test_pp_unit_inference(self, setup_test_config, sample_historical_data):
        """Test PP unit inference."""
        engineer = FeatureEngineer(sample_historical_data)
        
        pp_unit = engineer.infer_pp_unit(8478402, datetime(2024, 11, 15))
        
        assert pp_unit in [0, 1, 2]
    
    def test_build_complete_features(self, setup_test_config, sample_historical_data):
        """Test building complete feature vector."""
        engineer = FeatureEngineer(sample_historical_data)
        
        features = engineer.build_features(
            player_id=8478402,
            player_name='Connor McDavid',
            position='C',
            team_id=22,
            opponent_team_id=6,
            game_id=2024111501,
            game_date=datetime(2024, 11, 15),
            venue_name='Rogers Arena',
            is_home=True
        )
        
        assert isinstance(features, PlayerFeatures)
        assert features.player_id == 8478402
        assert features.position == 'C'
        
        # Check usage features exist and are reasonable
        assert features.toi_per_game_l10 > 0
        assert features.shots_per_game_l10 >= 0
        
        # Check uncertainty metrics
        assert 0 <= features.lineup_confidence <= 1
        assert features.projected_toi > 0
    
    def test_features_to_dataframe(self, setup_test_config, sample_historical_data):
        """Test conversion to DataFrame."""
        engineer = FeatureEngineer(sample_historical_data)
        
        features = engineer.build_features(
            player_id=8478402,
            player_name='Test Player',
            position='C',
            team_id=22,
            opponent_team_id=6,
            game_id=2024111501,
            game_date=datetime(2024, 11, 15),
            venue_name='Test Arena',
            is_home=True
        )
        
        df = engineer.features_to_dataframe(features)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'player_id' in df.columns
        assert 'toi_per_game_l10' in df.columns
    
    def test_bulk_build_features(self, setup_test_config, sample_historical_data):
        """Test bulk feature building."""
        engineer = FeatureEngineer(sample_historical_data)
        
        tasks = [
            {
                'player_id': 8478402,
                'player_name': 'Player 1',
                'position': 'C',
                'team_id': 22,
                'opponent_team_id': 6,
                'game_id': 2024111501,
                'game_date': datetime(2024, 11, 15),
                'venue_name': 'Test Arena',
                'is_home': True,
            },
            {
                'player_id': 8479318,
                'player_name': 'Player 2',
                'position': 'RW',
                'team_id': 22,
                'opponent_team_id': 6,
                'game_id': 2024111501,
                'game_date': datetime(2024, 11, 15),
                'venue_name': 'Test Arena',
                'is_home': True,
            },
        ]
        
        features_df = engineer.bulk_build_features(tasks)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 2
        assert all(features_df['player_id'].isin([8478402, 8479318]))
    
    def test_venue_bias_reasonable(self, setup_test_config, sample_historical_data):
        """Test venue bias is in reasonable range."""
        engineer = FeatureEngineer(sample_historical_data)
        
        for venue, bias in engineer.venue_bias.items():
            assert -1.0 <= bias <= 1.0, f"Venue bias out of range: {venue}={bias}"
    
    def test_team_stats_structure(self, setup_test_config, sample_historical_data):
        """Test team stats have correct structure."""
        engineer = FeatureEngineer(sample_historical_data)
        
        for team_id, stats in engineer.team_stats.items():
            assert 'pace' in stats
            assert 'shot_share' in stats
            assert stats['pace'] > 0
            assert 0 <= stats['shot_share'] <= 1


class TestHistoricalDataCollector:
    """Tests for historical data collection."""
    
    @patch('src.data_pipeline.pipeline.create_nhl_client')
    def test_initialization(self, mock_client, setup_test_config):
        """Test collector initializes."""
        mock_client.return_value = Mock()
        collector = HistoricalDataCollector()
        
        assert collector.client is not None
        assert isinstance(collector.historical_data, pd.DataFrame)
    
    @patch('src.data_pipeline.pipeline.create_nhl_client')
    def test_save_and_load(self, mock_client, setup_test_config, sample_historical_data, tmp_path):
        """Test saving and loading data."""
        mock_client.return_value = Mock()
        collector = HistoricalDataCollector()
        
        # Override config path for test
        collector.config.data.raw_data_dir = tmp_path
        
        # Save
        filename = 'test_data.parquet'
        collector.save_to_disk(sample_historical_data, filename)
        
        # Check file exists
        assert (tmp_path / filename).exists()
        
        # Load
        loaded_data = collector.load_from_disk(filename)
        
        assert len(loaded_data) == len(sample_historical_data)
        assert list(loaded_data.columns) == list(sample_historical_data.columns)


class TestPredictionPipeline:
    """Tests for prediction pipeline."""
    
    @patch('src.data_pipeline.pipeline.create_nhl_client')
    @patch('src.data_pipeline.pipeline.PredictionPipeline._load_latest_model')
    @patch('src.data_pipeline.pipeline.PredictionPipeline._load_calibrator')
    @patch('src.data_pipeline.pipeline.PredictionPipeline._load_historical_data')
    def test_initialization(self, mock_hist, mock_cal, mock_model, mock_client, 
                          setup_test_config, sample_historical_data):
        """Test pipeline initializes."""
        mock_client.return_value = Mock()
        mock_model.return_value = Mock()
        mock_cal.return_value = None
        mock_hist.return_value = sample_historical_data
        
        pipeline = PredictionPipeline()
        
        assert pipeline.client is not None
        assert pipeline.feature_engineer is not None
    
    def test_format_predictions_for_export(self, setup_test_config):
        """Test prediction formatting."""
        # Create mock predictions
        predictions = pd.DataFrame({
            'player_name': ['Player 1', 'Player 2'],
            'position': ['C', 'RW'],
            'game_id': [123, 123],
            'is_home': [True, False],
            'mu': [3.2, 2.8],
            'alpha': [1.5, 1.8],
            'p_over_2.5': [0.5234, 0.4123],
            'p_over_2.5_calibrated': [0.5100, 0.4000],
            'p10': [1, 1],
            'p50': [3, 3],
            'p90': [6, 5],
            'projected_toi': [21.5, 18.2],
            'lineup_confidence': [0.95, 0.88],
        })
        
        # Create pipeline with mocks (simplified)
        with patch.multiple(
            'src.data_pipeline.pipeline.PredictionPipeline',
            _load_latest_model=Mock(return_value=Mock()),
            _load_calibrator=Mock(return_value=None),
            _load_historical_data=Mock(return_value=pd.DataFrame())
        ):
            pipeline = PredictionPipeline(
                model=Mock(),
                calibrator=None,
                historical_data=pd.DataFrame()
            )
            
            formatted = pipeline.format_predictions_for_export(predictions)
            
            assert isinstance(formatted, pd.DataFrame)
            assert 'player_name' in formatted.columns
            assert 'mu' in formatted.columns


class TestDataIntegrity:
    """Tests for data integrity and validation."""
    
    def test_no_future_data_leakage(self, sample_historical_data):
        """Test that features don't use future data."""
        engineer = FeatureEngineer(sample_historical_data)
        
        # Build features for a date
        target_date = datetime(2024, 11, 1)
        
        features = engineer.build_features(
            player_id=8478402,
            player_name='Test',
            position='C',
            team_id=22,
            opponent_team_id=6,
            game_id=123,
            game_date=target_date,
            venue_name='Test Arena',
            is_home=True
        )
        
        # Check that historical data used is before target date
        player_data = sample_historical_data[
            (sample_historical_data['player_id'] == 8478402) &
            (sample_historical_data['game_date'] >= target_date)
        ]
        
        # If there's data on/after target date, ensure features don't reflect it
        if not player_data.empty:
            # Features should be based on data BEFORE target_date
            assert features.games_played <= len(
                sample_historical_data[
                    (sample_historical_data['player_id'] == 8478402) &
                    (sample_historical_data['game_date'] < target_date)
                ]
            )
    
    def test_feature_consistency(self, sample_historical_data):
        """Test that same inputs produce same features."""
        engineer = FeatureEngineer(sample_historical_data)
        
        # Build features twice with same inputs
        features1 = engineer.build_features(
            player_id=8478402,
            player_name='Test',
            position='C',
            team_id=22,
            opponent_team_id=6,
            game_id=123,
            game_date=datetime(2024, 11, 15),
            venue_name='Test Arena',
            is_home=True
        )
        
        features2 = engineer.build_features(
            player_id=8478402,
            player_name='Test',
            position='C',
            team_id=22,
            opponent_team_id=6,
            game_id=123,
            game_date=datetime(2024, 11, 15),
            venue_name='Test Arena',
            is_home=True
        )
        
        # Compare key numeric features
        assert features1.toi_per_game_l10 == features2.toi_per_game_l10
        assert features1.shots_per_game_l10 == features2.shots_per_game_l10
        assert features1.projected_toi == features2.projected_toi
    
    def test_no_nan_in_features(self, sample_historical_data):
        """Test that features don't contain NaN values."""
        engineer = FeatureEngineer(sample_historical_data)
        
        features = engineer.build_features(
            player_id=8478402,
            player_name='Test',
            position='C',
            team_id=22,
            opponent_team_id=6,
            game_id=123,
            game_date=datetime(2024, 11, 15),
            venue_name='Test Arena',
            is_home=True
        )
        
        df = engineer.features_to_dataframe(features)
        
        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        assert len(nan_cols) == 0, f"NaN values found in columns: {nan_cols}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])