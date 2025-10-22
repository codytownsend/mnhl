"""
Configuration management for NHL SOG prediction system.

Loads and validates configuration from YAML file, provides type-safe access
to all configuration parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass
class DataConfig:
    """Data source and storage configuration."""
    collection_start_date: str
    nhl_api_base_url: str
    nhl_api_rate_limit: float
    nhl_api_timeout: int
    nhl_api_retries: int
    
    cache_enabled: bool
    cache_dir: Path
    cache_ttl_hours: int
    
    raw_data_dir: Path
    processed_data_dir: Path
    model_artifacts_dir: Path


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    rolling_windows: Dict[str, Optional[int]]
    min_games: Dict[str, int]
    toi_bins: List[float]
    strength_states: List[str]
    shot_distance_bins: List[float]
    ewma_alpha: float
    
    include_rest_days: bool
    include_travel: bool
    include_home_away: bool
    include_venue_bias: bool
    include_b2b_flag: bool
    include_score_effects: bool


@dataclass
class LGBMModelConfig:
    """LightGBM model hyperparameters."""
    objective: str
    boosting_type: str
    num_leaves: int
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    random_state: int
    n_jobs: int
    verbose: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LightGBM."""
        return {
            'objective': self.objective,
            'boosting_type': self.boosting_type,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
        }


@dataclass
class ModelConfig:
    """Model configuration and hyperparameters."""
    type: str
    target: str
    include_ot_shots: bool
    common_lines: List[float]
    
    nb_distribution: str
    nb_min_dispersion: float
    nb_max_dispersion: float
    
    mu_model: LGBMModelConfig
    alpha_model: LGBMModelConfig


@dataclass
class CalibrationConfig:
    """Calibration parameters."""
    method: str
    n_bins: int
    min_samples_per_bin: int
    recalibration_threshold: float
    
    inflate_uncertainty_enabled: bool
    inflation_conditions: List[str]
    inflation_factor: float


@dataclass
class ValidationConfig:
    """Validation and evaluation configuration."""
    train_pct: float
    val_pct: float
    test_pct: float
    
    min_train_samples: int
    min_val_samples: int
    min_test_samples: int
    
    purge_days: int
    
    metrics: List[str]
    baselines: List[str]
    
    cv_n_splits: int
    cv_test_size_days: int
    cv_gap_days: int
    
    calibration_bins: int
    confidence_levels: List[float]


@dataclass
class PredictionConfig:
    """Prediction pipeline configuration."""
    prediction_lead_time_minutes: int
    toi_projection_method: str
    
    use_lineup_projection: bool
    lineup_confidence_threshold: float
    fallback_to_season_avg: bool
    
    min_prediction_confidence: float
    
    output_format: str
    include_full_pmf: bool
    max_pmf_value: int
    include_percentiles: List[int]
    decimal_precision: int


@dataclass
class TrainingConfig:
    """Training configuration."""
    retrain_frequency_days: int
    retrain_day_of_week: str
    
    incremental_training_enabled: bool
    incremental_lookback_days: int
    
    feature_selection_enabled: bool
    feature_selection_method: str
    feature_selection_n_repeats: int
    feature_selection_threshold: int
    
    early_stopping_enabled: bool
    early_stopping_rounds: int
    early_stopping_metric: str


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    rolling_window_days: int
    
    crps_degradation_threshold: float
    calibration_error_threshold: float
    coverage_deviation_threshold: float
    
    log_level: str
    log_predictions: bool
    log_features: bool
    
    save_model_artifacts: bool
    model_version_format: str


@dataclass
class ProductionConfig:
    """Production settings."""
    api_enabled: bool
    api_host: str
    api_port: int
    
    batch_max_players: int
    batch_parallel_workers: int
    
    graceful_degradation: bool
    fallback_to_baseline: bool


@dataclass
class Config:
    """Root configuration object."""
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    calibration: CalibrationConfig
    validation: ValidationConfig
    prediction: PredictionConfig
    training: TrainingConfig
    monitoring: MonitoringConfig
    production: ProductionConfig
    
    random_seed: int
    deterministic: bool
    
    _config_path: Optional[Path] = field(default=None, repr=False)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object
            
        Raises:
            ConfigError: If file not found or invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML: {e}")
        
        # Parse and validate each section
        try:
            data_config = cls._parse_data_config(raw_config['data'])
            features_config = cls._parse_features_config(raw_config['features'])
            model_config = cls._parse_model_config(raw_config['model'])
            calibration_config = cls._parse_calibration_config(raw_config['calibration'])
            validation_config = cls._parse_validation_config(raw_config['validation'])
            prediction_config = cls._parse_prediction_config(raw_config['prediction'])
            training_config = cls._parse_training_config(raw_config['training'])
            monitoring_config = cls._parse_monitoring_config(raw_config['monitoring'])
            production_config = cls._parse_production_config(raw_config['production'])
            
            config = cls(
                data=data_config,
                features=features_config,
                model=model_config,
                calibration=calibration_config,
                validation=validation_config,
                prediction=prediction_config,
                training=training_config,
                monitoring=monitoring_config,
                production=production_config,
                random_seed=raw_config['random_seed'],
                deterministic=raw_config['deterministic'],
                _config_path=config_path
            )
            
            # Create directories if they don't exist
            config._ensure_directories()
            
            return config
            
        except KeyError as e:
            raise ConfigError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ConfigError(f"Error parsing configuration: {e}")
    
    @staticmethod
    def _parse_data_config(data: Dict) -> DataConfig:
        """Parse data configuration section."""
        return DataConfig(
            collection_start_date=data['collection_start_date'],
            nhl_api_base_url=data['nhl_api']['base_url'],
            nhl_api_rate_limit=data['nhl_api']['rate_limit_seconds'],
            nhl_api_timeout=data['nhl_api']['timeout_seconds'],
            nhl_api_retries=data['nhl_api']['retry_attempts'],
            cache_enabled=data['cache']['enabled'],
            cache_dir=Path(data['cache']['directory']),
            cache_ttl_hours=data['cache']['ttl_hours'],
            raw_data_dir=Path(data['storage']['raw_data_dir']),
            processed_data_dir=Path(data['storage']['processed_data_dir']),
            model_artifacts_dir=Path(data['storage']['model_artifacts_dir']),
        )
    
    @staticmethod
    def _parse_features_config(features: Dict) -> FeatureConfig:
        """Parse features configuration section."""
        return FeatureConfig(
            rolling_windows=features['rolling_windows'],
            min_games=features['min_games'],
            toi_bins=features['toi_bins'],
            strength_states=features['strength_states'],
            shot_distance_bins=features['shot_distance_bins'],
            ewma_alpha=features['ewma_alpha'],
            include_rest_days=features['include_rest_days'],
            include_travel=features['include_travel'],
            include_home_away=features['include_home_away'],
            include_venue_bias=features['include_venue_bias'],
            include_b2b_flag=features['include_b2b_flag'],
            include_score_effects=features['include_score_effects'],
        )
    
    @staticmethod
    def _parse_model_config(model: Dict) -> ModelConfig:
        """Parse model configuration section."""
        mu_params = model['lightgbm']['mu_model']
        alpha_params = model['lightgbm']['alpha_model']
        
        mu_model = LGBMModelConfig(
            objective=mu_params['objective'],
            boosting_type=mu_params['boosting_type'],
            num_leaves=mu_params['num_leaves'],
            learning_rate=mu_params['learning_rate'],
            n_estimators=mu_params['n_estimators'],
            max_depth=mu_params['max_depth'],
            min_child_samples=mu_params['min_child_samples'],
            subsample=mu_params['subsample'],
            colsample_bytree=mu_params['colsample_bytree'],
            reg_alpha=mu_params['reg_alpha'],
            reg_lambda=mu_params['reg_lambda'],
            random_state=mu_params['random_state'],
            n_jobs=mu_params['n_jobs'],
            verbose=mu_params['verbose'],
        )
        
        alpha_model = LGBMModelConfig(
            objective=alpha_params['objective'],
            boosting_type=alpha_params['boosting_type'],
            num_leaves=alpha_params['num_leaves'],
            learning_rate=alpha_params['learning_rate'],
            n_estimators=alpha_params['n_estimators'],
            max_depth=alpha_params['max_depth'],
            min_child_samples=alpha_params['min_child_samples'],
            subsample=alpha_params['subsample'],
            colsample_bytree=alpha_params['colsample_bytree'],
            reg_alpha=alpha_params['reg_alpha'],
            reg_lambda=alpha_params['reg_lambda'],
            random_state=alpha_params['random_state'],
            n_jobs=alpha_params['n_jobs'],
            verbose=alpha_params['verbose'],
        )
        
        return ModelConfig(
            type=model['type'],
            target=model['target'],
            include_ot_shots=model['include_ot_shots'],
            common_lines=model['common_lines'],
            nb_distribution=model['negative_binomial']['distribution'],
            nb_min_dispersion=model['negative_binomial']['nb_min_dispersion'],
            nb_max_dispersion=model['negative_binomial']['nb_max_dispersion'],
            mu_model=mu_model,
            alpha_model=alpha_model,
        )
    
    @staticmethod
    def _parse_calibration_config(calibration: Dict) -> CalibrationConfig:
        """Parse calibration configuration section."""
        return CalibrationConfig(
            method=calibration['method'],
            n_bins=calibration['n_bins'],
            min_samples_per_bin=calibration['min_samples_per_bin'],
            recalibration_threshold=calibration['recalibration_threshold'],
            inflate_uncertainty_enabled=calibration['inflate_uncertainty']['enabled'],
            inflation_conditions=calibration['inflate_uncertainty']['conditions'],
            inflation_factor=calibration['inflate_uncertainty']['inflation_factor'],
        )
    
    @staticmethod
    def _parse_validation_config(validation: Dict) -> ValidationConfig:
        """Parse validation configuration section."""
        return ValidationConfig(
            train_pct=validation['train_pct'],
            val_pct=validation['val_pct'],
            test_pct=validation['test_pct'],
            min_train_samples=validation['min_train_samples'],
            min_val_samples=validation['min_val_samples'],
            min_test_samples=validation['min_test_samples'],
            purge_days=validation['purge_days'],
            metrics=validation['metrics'],
            baselines=validation['baselines'],
            cv_n_splits=validation['time_series_cv']['n_splits'],
            cv_test_size_days=validation['time_series_cv']['test_size_days'],
            cv_gap_days=validation['time_series_cv']['gap_days'],
            calibration_bins=validation['calibration_bins'],
            confidence_levels=validation['confidence_levels'],
        )
    
    @staticmethod
    def _parse_prediction_config(prediction: Dict) -> PredictionConfig:
        """Parse prediction configuration section."""
        return PredictionConfig(
            prediction_lead_time_minutes=prediction['prediction_lead_time_minutes'],
            toi_projection_method=prediction['toi_projection_method'],
            use_lineup_projection=prediction['lineup_uncertainty']['use_projection'],
            lineup_confidence_threshold=prediction['lineup_uncertainty']['confidence_threshold'],
            fallback_to_season_avg=prediction['lineup_uncertainty']['fallback_to_season_avg'],
            min_prediction_confidence=prediction['min_prediction_confidence'],
            output_format=prediction['output']['format'],
            include_full_pmf=prediction['output']['include_full_pmf'],
            max_pmf_value=prediction['output']['max_pmf_value'],
            include_percentiles=prediction['output']['include_percentiles'],
            decimal_precision=prediction['output']['decimal_precision'],
        )
    
    @staticmethod
    def _parse_training_config(training: Dict) -> TrainingConfig:
        """Parse training configuration section."""
        return TrainingConfig(
            retrain_frequency_days=training['retrain_frequency_days'],
            retrain_day_of_week=training['retrain_day_of_week'],
            incremental_training_enabled=training['incremental_training']['enabled'],
            incremental_lookback_days=training['incremental_training']['lookback_days'],
            feature_selection_enabled=training['feature_selection']['enabled'],
            feature_selection_method=training['feature_selection']['method'],
            feature_selection_n_repeats=training['feature_selection']['n_repeats'],
            feature_selection_threshold=training['feature_selection']['threshold_percentile'],
            early_stopping_enabled=training['early_stopping']['enabled'],
            early_stopping_rounds=training['early_stopping']['rounds'],
            early_stopping_metric=training['early_stopping']['metric'],
        )
    
    @staticmethod
    def _parse_monitoring_config(monitoring: Dict) -> MonitoringConfig:
        """Parse monitoring configuration section."""
        return MonitoringConfig(
            rolling_window_days=monitoring['rolling_window_days'],
            crps_degradation_threshold=monitoring['alerts']['crps_degradation_threshold'],
            calibration_error_threshold=monitoring['alerts']['calibration_error_threshold'],
            coverage_deviation_threshold=monitoring['alerts']['coverage_deviation_threshold'],
            log_level=monitoring['log_level'],
            log_predictions=monitoring['log_predictions'],
            log_features=monitoring['log_features'],
            save_model_artifacts=monitoring['save_model_artifacts'],
            model_version_format=monitoring['model_version_format'],
        )
    
    @staticmethod
    def _parse_production_config(production: Dict) -> ProductionConfig:
        """Parse production configuration section."""
        return ProductionConfig(
            api_enabled=production['api']['enabled'],
            api_host=production['api']['host'],
            api_port=production['api']['port'],
            batch_max_players=production['batch']['max_players_per_batch'],
            batch_parallel_workers=production['batch']['parallel_workers'],
            graceful_degradation=production['graceful_degradation'],
            fallback_to_baseline=production['fallback_to_baseline'],
        )
    
    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.model_artifacts_dir,
            self.data.cache_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ConfigError: If any parameters are invalid
        """
        # Validate split percentages sum to 1.0
        total_pct = self.validation.train_pct + self.validation.val_pct + self.validation.test_pct
        if not (0.99 <= total_pct <= 1.01):  # Allow small floating point error
            raise ConfigError(f"Train/val/test percentages must sum to 1.0 (got {total_pct})")
        
        # Validate numeric ranges
        if self.features.ewma_alpha <= 0 or self.features.ewma_alpha >= 1:
            raise ConfigError("ewma_alpha must be between 0 and 1")
        
        if self.model.nb_min_dispersion >= self.model.nb_max_dispersion:
            raise ConfigError("nb_min_dispersion must be less than nb_max_dispersion")
        
        # Validate common lines are sorted
        if self.model.common_lines != sorted(self.model.common_lines):
            raise ConfigError("common_lines must be sorted in ascending order")
        
        # Validate model type
        valid_model_types = ['lightgbm_nb', 'statsmodels_nb', 'lgbm_poisson']
        if self.model.type not in valid_model_types:
            raise ConfigError(f"model.type must be one of {valid_model_types}")
        
        # Validate calibration method
        valid_calibration = ['isotonic', 'platt', 'beta']
        if self.calibration.method not in valid_calibration:
            raise ConfigError(f"calibration.method must be one of {valid_calibration}")
    
    def get_model_version_string(self) -> str:
        """
        Generate model version string based on configuration.
        
        Returns:
            Version string
        """
        date_str = datetime.now().strftime("%Y%m%d")
        return self.monitoring.model_version_format.format(
            date=date_str,
            metric=self.training.early_stopping_metric
        )
    
    def __repr__(self) -> str:
        """String representation showing config file path."""
        if self._config_path:
            return f"Config(loaded_from='{self._config_path}')"
        return "Config(no_file)"


# Singleton pattern for global config access
_global_config: Optional[Config] = None


def load_config(config_path: Union[str, Path] = "config/model_config.yaml") -> Config:
    """
    Load configuration from YAML file and set as global config.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    global _global_config
    _global_config = Config.from_yaml(config_path)
    _global_config.validate()
    return _global_config


def get_config() -> Config:
    """
    Get global configuration object.
    
    Returns:
        Config object
        
    Raises:
        ConfigError: If config not loaded yet
    """
    if _global_config is None:
        raise ConfigError(
            "Configuration not loaded. Call load_config() first."
        )
    return _global_config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Reload configuration, optionally from a different file.
    
    Args:
        config_path: Path to configuration file (uses previous path if None)
        
    Returns:
        Config object
    """
    global _global_config
    
    if config_path is None and _global_config is not None:
        config_path = _global_config._config_path
    elif config_path is None:
        config_path = "config/model_config.yaml"
    
    return load_config(config_path)