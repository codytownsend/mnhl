"""
NHL API client for fetching game, player, and team data.

Handles rate limiting, caching, retries, and provides clean interfaces
for all required data endpoints.
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import get_config


class NHLAPIError(Exception):
    """Raised when NHL API request fails."""
    pass


@dataclass
class GameInfo:
    """Basic game information."""
    game_id: int
    season: str
    game_type: str
    date: datetime
    home_team_id: int
    away_team_id: int
    home_team_abbrev: str
    away_team_abbrev: str
    venue_name: str
    game_state: str
    start_time: datetime


@dataclass
class PlayerGameStats:
    """Player statistics for a single game."""
    player_id: int
    player_name: str
    team_id: int
    position: str
    
    goals: int
    assists: int
    shots: int
    toi_seconds: int
    
    ev_toi_seconds: int
    pp_toi_seconds: int
    sh_toi_seconds: int
    
    shifts: int
    faceoff_pct: Optional[float]
    hits: int
    blocked: int
    
    game_id: int
    game_date: datetime
    home_away: str


class NHLAPIClient:
    """
    Client for NHL public API.
    
    Handles:
    - Rate limiting
    - Automatic retries with exponential backoff
    - Response caching
    - Clean data extraction
    """
    
    def __init__(self, cache_enabled: bool = None, cache_dir: Path = None):
        """
        Initialize NHL API client.
        
        Args:
            cache_enabled: Enable response caching (uses config default if None)
            cache_dir: Cache directory (uses config default if None)
        """
        config = get_config()
        
        self.base_url = config.data.nhl_api_base_url
        self.rate_limit = config.data.nhl_api_rate_limit
        self.timeout = config.data.nhl_api_timeout
        
        self.cache_enabled = cache_enabled if cache_enabled is not None else config.data.cache_enabled
        self.cache_dir = cache_dir if cache_dir is not None else config.data.cache_dir
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last request time for rate limiting
        self._last_request_time = 0
        
        # Setup session with retry logic
        self.session = self._create_session(config.data.nhl_api_retries)
    
    def _create_session(self, max_retries: int) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from endpoint and parameters."""
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Retrieve response from cache if available and fresh."""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is still valid
        config = get_config()
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age > config.data.cache_ttl_hours * 3600:
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save response to cache."""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            # Cache write failures are non-fatal
            pass
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make HTTP request to NHL API with caching and rate limiting.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            
        Returns:
            JSON response as dict
            
        Raises:
            NHLAPIError: If request fails after retries
        """
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            return cached_response
        
        # Rate limit
        self._rate_limit()
        
        # Make request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Cache successful response
            self._save_to_cache(cache_key, data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise NHLAPIError(f"Failed to fetch {endpoint}: {e}")
        except json.JSONDecodeError as e:
            raise NHLAPIError(f"Invalid JSON response from {endpoint}: {e}")
    
    def get_schedule(self, start_date: datetime, end_date: datetime) -> List[GameInfo]:
        """
        Fetch schedule for date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of GameInfo objects
        """
        games = []
        
        # Make sure start/end dates are timezone-naive for comparison
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        
        # NHL API uses YYYY-MM-DD format
        date_str = start_date.strftime("%Y-%m-%d")
        
        # Fetch schedule
        try:
            data = self._request(f"schedule/{date_str}")
        except Exception as e:
            logger.warning(f"Failed to fetch schedule for {date_str}: {e}")
            return games
        
        if 'gameWeek' not in data:
            return games
        
        for game_week in data['gameWeek']:
            for game_data in game_week.get('games', []):
                try:
                    # Handle different date field names
                    game_date_str = game_data.get('gameDate') or game_data.get('startTimeUTC')
                    if not game_date_str:
                        continue
                    
                    game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    # Remove timezone info for comparison
                    game_date_naive = game_date.replace(tzinfo=None)
                    
                    # Filter by date range
                    if not (start_date <= game_date_naive <= end_date):
                        continue
                    
                    game = GameInfo(
                        game_id=game_data['id'],
                        season=game_data.get('season', ''),
                        game_type=game_data.get('gameType', ''),
                        date=game_date_naive,
                        home_team_id=game_data['homeTeam']['id'],
                        away_team_id=game_data['awayTeam']['id'],
                        home_team_abbrev=game_data['homeTeam'].get('abbrev', ''),
                        away_team_abbrev=game_data['awayTeam'].get('abbrev', ''),
                        venue_name=game_data.get('venue', {}).get('default', ''),
                        game_state=game_data.get('gameState', ''),
                        start_time=game_date_naive
                    )
                    games.append(game)
                except (KeyError, ValueError) as e:
                    logger.debug(f"Failed to parse game data: {e}")
                    continue
        
        return games
    
    def get_game_boxscore(self, game_id: int) -> Dict[str, Any]:
        """
        Fetch boxscore for a game.
        
        Args:
            game_id: NHL game ID
            
        Returns:
            Boxscore data including player stats
        """
        return self._request(f"gamecenter/{game_id}/boxscore")
    
    def get_game_play_by_play(self, game_id: int) -> Dict[str, Any]:
        """
        Fetch play-by-play data for a game.
        
        Args:
            game_id: NHL game ID
            
        Returns:
            Play-by-play data including all events, shots, goals, etc.
        """
        return self._request(f"gamecenter/{game_id}/play-by-play")
    
    def get_player_stats(self, player_id: int, season: str = None) -> Dict[str, Any]:
        """
        Fetch player career/season statistics.
        
        Args:
            player_id: NHL player ID
            season: Season string (e.g., "20232024") - uses current season if None
            
        Returns:
            Player stats data
        """
        endpoint = f"player/{player_id}/landing"
        return self._request(endpoint)
    
    def get_team_roster(self, team_abbrev: str, season: str = None) -> List[Dict[str, Any]]:
        """
        Fetch team roster.
        
        Args:
            team_abbrev: Team abbreviation (e.g., "TOR", "BOS")
            season: Season string - uses current season if None
            
        Returns:
            List of player data
        """
        data = self._request(f"roster/{team_abbrev}/current")
        
        roster = []
        for position_group in ['forwards', 'defensemen', 'goalies']:
            if position_group in data:
                roster.extend(data[position_group])
        
        return roster
    
    def get_shifts(self, game_id: int) -> Dict[str, List[Dict]]:
        """
        Fetch shift data for a game.
        
        Note: Shift data may not be available for all games in real-time.
        
        Args:
            game_id: NHL game ID
            
        Returns:
            Dict with 'home' and 'away' shift lists
        """
        # NHL's shift data is often on a different endpoint
        # This is a placeholder - actual implementation may need adjustment
        try:
            data = self._request(f"gamecenter/{game_id}/play-by-play")
            # Extract shift information from play-by-play if available
            return self._extract_shifts_from_pbp(data)
        except NHLAPIError:
            return {'home': [], 'away': []}
    
    def _extract_shifts_from_pbp(self, pbp_data: Dict) -> Dict[str, List[Dict]]:
        """Extract shift information from play-by-play data."""
        # This is a simplified extraction - actual implementation would be more complex
        shifts = {'home': [], 'away': []}
        
        # The actual shift extraction logic would go here
        # For now, return empty lists
        return shifts
    
    def parse_player_game_stats(self, boxscore: Dict, game_id: int, 
                                 game_date: datetime) -> List[PlayerGameStats]:
        """
        Parse boxscore data into PlayerGameStats objects.
        
        Args:
            boxscore: Boxscore data from get_game_boxscore()
            game_id: Game ID
            game_date: Game date
            
        Returns:
            List of PlayerGameStats for all players in the game
        """
        stats = []
        
        for team_side in ['homeTeam', 'awayTeam']:
            if team_side not in boxscore:
                continue
            
            team_data = boxscore[team_side]
            team_id = team_data.get('id')
            home_away = 'home' if team_side == 'homeTeam' else 'away'
            
            # Parse forwards and defense
            for position_group in ['forwards', 'defense']:
                if position_group not in boxscore['playerByGameStats'][team_side]:
                    continue
                
                for player_data in boxscore['playerByGameStats'][team_side][position_group]:
                    stats.append(self._parse_single_player_stats(
                        player_data, team_id, game_id, game_date, home_away
                    ))
        
        return stats
    
    def _parse_single_player_stats(self, player_data: Dict, team_id: int,
                                     game_id: int, game_date: datetime,
                                     home_away: str) -> PlayerGameStats:
        """Parse single player's stats from boxscore."""
        # Handle TOI - it might not be present in all games
        toi = player_data.get('toi', '0:00')
        ev_toi = player_data.get('evenStrengthToi', '0:00')
        pp_toi = player_data.get('powerPlayToi', '0:00')
        sh_toi = player_data.get('shorthandedToi', '0:00')
        
        # API uses 'sog' for shots on goal
        shots = player_data.get('sog', player_data.get('shots', 0))
        
        return PlayerGameStats(
            player_id=player_data['playerId'],
            player_name=player_data.get('name', {}).get('default', ''),
            team_id=team_id,
            position=player_data.get('position', ''),
            goals=player_data.get('goals', 0),
            assists=player_data.get('assists', 0),
            shots=shots,
            toi_seconds=self._toi_to_seconds(toi),
            ev_toi_seconds=self._toi_to_seconds(ev_toi),
            pp_toi_seconds=self._toi_to_seconds(pp_toi),
            sh_toi_seconds=self._toi_to_seconds(sh_toi),
            shifts=player_data.get('shifts', 0),
            faceoff_pct=player_data.get('faceoffWinningPctg'),
            hits=player_data.get('hits', 0),
            blocked=player_data.get('blockedShots', 0),
            game_id=game_id,
            game_date=game_date,
            home_away=home_away
        )
    
    @staticmethod
    def _toi_to_seconds(toi_str: str) -> int:
        """Convert TOI string (MM:SS) to seconds."""
        if not toi_str or toi_str == '0:00':
            return 0
        
        try:
            parts = toi_str.split(':')
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                return minutes * 60 + seconds
            return 0
        except (ValueError, AttributeError):
            return 0
    
    def get_standings(self, season: str = None, date: datetime = None) -> Dict[str, Any]:
        """
        Fetch league standings.
        
        Args:
            season: Season string (uses current if None)
            date: Get standings as of specific date (uses current if None)
            
        Returns:
            Standings data
        """
        date_str = date.strftime("%Y-%m-%d") if date else "now"
        return self._request(f"standings/{date_str}")
    
    def get_team_stats(self, team_abbrev: str, season: str = None) -> Dict[str, Any]:
        """
        Fetch team statistics.
        
        Args:
            team_abbrev: Team abbreviation
            season: Season string (uses current if None)
            
        Returns:
            Team stats data
        """
        # Team stats are often embedded in other endpoints
        # This is a placeholder for the actual implementation
        return self._request(f"club-stats/{team_abbrev}/now")
    
    def get_game_story(self, game_id: int) -> Dict[str, Any]:
        """
        Fetch game story/summary.
        
        Args:
            game_id: NHL game ID
            
        Returns:
            Game story data
        """
        return self._request(f"gamecenter/{game_id}/landing")
    
    def extract_shots_from_pbp(self, pbp_data: Dict) -> List[Dict[str, Any]]:
        """
        Extract all shot events from play-by-play data.
        
        Args:
            pbp_data: Play-by-play data from get_game_play_by_play()
            
        Returns:
            List of shot events with metadata
        """
        shots = []
        
        if 'plays' not in pbp_data:
            return shots
        
        for play in pbp_data['plays']:
            # Filter for shot events
            event_type = play.get('typeDescKey', '')
            if event_type not in ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']:
                continue
            
            shot_info = {
                'event_id': play.get('eventId'),
                'period': play.get('periodDescriptor', {}).get('number'),
                'time_in_period': play.get('timeInPeriod'),
                'time_remaining': play.get('timeRemaining'),
                'situation_code': play.get('situationCode'),
                'event_type': event_type,
                'shot_type': play.get('details', {}).get('shotType'),
                'shooter_player_id': play.get('details', {}).get('shootingPlayerId'),
                'goalie_player_id': play.get('details', {}).get('goalieInNetId'),
                'x_coord': play.get('details', {}).get('xCoord'),
                'y_coord': play.get('details', {}).get('yCoord'),
                'zone_code': play.get('details', {}).get('zoneCode'),
                'is_goal': event_type == 'goal',
                'is_shot_on_goal': event_type in ['shot-on-goal', 'goal'],
                'home_team_defending_side': play.get('homeTeamDefendingSide'),
            }
            
            shots.append(shot_info)
        
        return shots
    
    def get_todays_games(self) -> List[GameInfo]:
        """
        Convenience method to get today's games.
        
        Returns:
            List of GameInfo for today
        """
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())
        return self.get_schedule(start, end)
    
    def get_games_for_date(self, date: datetime) -> List[GameInfo]:
        """
        Get all games for a specific date.
        
        Args:
            date: Date to fetch games for
            
        Returns:
            List of GameInfo
        """
        start = datetime.combine(date.date(), datetime.min.time())
        end = datetime.combine(date.date(), datetime.max.time())
        return self.get_schedule(start, end)
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        if not self.cache_enabled:
            return
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError:
                pass


# Convenience function for creating a configured client
def create_nhl_client() -> NHLAPIClient:
    """
    Create NHL API client using global configuration.
    
    Returns:
        Configured NHLAPIClient instance
    """
    return NHLAPIClient()