"""
Structured logging configuration for NHL SOG prediction system.

Provides:
- Consistent log formatting across all modules
- File and console handlers with rotation
- Contextual logging (prediction_id, model_version, etc.)
- Performance monitoring hooks
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """
    JSON-structured log formatter for easier parsing and monitoring.
    
    Outputs logs in JSON format with consistent fields.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add any extra context fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual fields to all log messages.
    
    Useful for tracking prediction batches, model versions, etc.
    """
    
    def process(self, msg, kwargs):
        """Add context to log message."""
        # Merge context into extra fields
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        
        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    use_json: bool = False
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to logs/)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        use_json: Use JSON formatter (useful for production)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Define formatters
    if use_json:
        formatter = StructuredFormatter()
    else:
        # Human-readable format for console/development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        if log_dir is None:
            log_dir = Path('logs')
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = log_dir / f"nhl_sog_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('lightgbm').setLevel(logging.WARNING)
    
    root_logger.info(f"Logging configured: level={log_level}, file={log_to_file}, console={log_to_console}")


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (usually __name__)
        context: Optional context dict to add to all log messages
        
    Returns:
        Logger or ContextAdapter if context provided
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, context)
    
    return logger


class PerformanceLogger:
    """
    Context manager for logging performance metrics.
    
    Usage:
        with PerformanceLogger('feature_engineering'):
            # ... code to time ...
            pass
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize performance logger.
        
        Args:
            operation_name: Name of operation being timed
            logger: Logger to use (creates one if None)
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation_name} (took {duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation_name} (took {duration:.2f}s)")
        
        # Don't suppress exceptions
        return False


def log_prediction_batch(
    logger: logging.Logger,
    game_date: datetime,
    n_predictions: int,
    model_version: str,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log prediction batch with structured metadata.
    
    Args:
        logger: Logger instance
        game_date: Date predictions are for
        n_predictions: Number of predictions generated
        model_version: Model version used
        metrics: Optional metrics dict
    """
    log_data = {
        'game_date': game_date.isoformat(),
        'n_predictions': n_predictions,
        'model_version': model_version,
    }
    
    if metrics:
        log_data['metrics'] = metrics
    
    logger.info(
        f"Generated {n_predictions} predictions for {game_date.date()}",
        extra={'extra_fields': log_data}
    )


def log_model_training(
    logger: logging.Logger,
    train_samples: int,
    val_samples: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    model_version: str
) -> None:
    """
    Log model training summary.
    
    Args:
        logger: Logger instance
        train_samples: Number of training samples
        val_samples: Number of validation samples
        train_metrics: Training metrics
        val_metrics: Validation metrics
        model_version: Model version
    """
    log_data = {
        'train_samples': train_samples,
        'val_samples': val_samples,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_version': model_version,
    }
    
    logger.info(
        f"Model training complete: {model_version}",
        extra={'extra_fields': log_data}
    )


def log_calibration_results(
    logger: logging.Logger,
    method: str,
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float]
) -> None:
    """
    Log calibration improvement.
    
    Args:
        logger: Logger instance
        method: Calibration method used
        before_metrics: Metrics before calibration
        after_metrics: Metrics after calibration
    """
    log_data = {
        'calibration_method': method,
        'before': before_metrics,
        'after': after_metrics,
    }
    
    # Calculate improvements
    improvements = {}
    for key in before_metrics:
        if key in after_metrics:
            improvement = before_metrics[key] - after_metrics[key]
            improvements[key] = improvement
    
    log_data['improvements'] = improvements
    
    logger.info(
        f"Calibration complete using {method}",
        extra={'extra_fields': log_data}
    )


def log_api_request(
    logger: logging.Logger,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    cached: bool = False
) -> None:
    """
    Log API request for monitoring.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint called
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        cached: Whether result was cached
    """
    log_data = {
        'endpoint': endpoint,
        'status_code': status_code,
        'duration_ms': duration_ms,
        'cached': cached,
    }
    
    level = logging.DEBUG if status_code < 400 else logging.WARNING
    
    logger.log(
        level,
        f"API request: {endpoint} [{status_code}] ({duration_ms:.1f}ms)",
        extra={'extra_fields': log_data}
    )


def log_data_quality_issue(
    logger: logging.Logger,
    issue_type: str,
    description: str,
    affected_records: int,
    severity: str = "WARNING"
) -> None:
    """
    Log data quality issues.
    
    Args:
        logger: Logger instance
        issue_type: Type of issue (missing_data, outlier, etc.)
        description: Description of issue
        affected_records: Number of records affected
        severity: Severity level
    """
    log_data = {
        'issue_type': issue_type,
        'description': description,
        'affected_records': affected_records,
        'severity': severity,
    }
    
    level = getattr(logging, severity.upper(), logging.WARNING)
    
    logger.log(
        level,
        f"Data quality issue: {issue_type} - {description}",
        extra={'extra_fields': log_data}
    )


class MetricsLogger:
    """
    Helper class for logging model evaluation metrics.
    
    Formats metrics in a consistent, readable way.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Log metrics in a formatted table.
        
        Args:
            metrics: Dict of metric name -> value
            prefix: Optional prefix for metric names
        """
        self.logger.info(f"{prefix}Metrics:")
        
        # Group metrics by type
        main_metrics = ['crps', 'mae', 'rmse', 'calibration_error']
        brier_metrics = [k for k in metrics if k.startswith('brier_')]
        coverage_metrics = [k for k in metrics if k.startswith('coverage_')]
        other_metrics = [k for k in metrics if k not in main_metrics + brier_metrics + coverage_metrics]
        
        # Log main metrics
        for metric in main_metrics:
            if metric in metrics:
                self.logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        # Log Brier scores
        if brier_metrics:
            self.logger.info("  Brier scores:")
            for metric in sorted(brier_metrics):
                line = metric.replace('brier_', '')
                self.logger.info(f"    {line}: {metrics[metric]:.4f}")
        
        # Log coverage
        if coverage_metrics:
            self.logger.info("  Coverage:")
            for metric in sorted(coverage_metrics):
                level = metric.replace('coverage_', '')
                self.logger.info(f"    {level}%: {metrics[metric]:.3f}")
        
        # Log other metrics
        for metric in other_metrics:
            self.logger.info(f"  {metric}: {metrics[metric]:.4f}")
    
    def log_comparison(self, baseline_metrics: Dict[str, float],
                      model_metrics: Dict[str, float],
                      baseline_name: str = "Baseline",
                      model_name: str = "Model") -> None:
        """
        Log side-by-side comparison of metrics.
        
        Args:
            baseline_metrics: Baseline metrics
            model_metrics: Model metrics
            baseline_name: Name for baseline
            model_name: Name for model
        """
        self.logger.info(f"\nMetric Comparison:")
        self.logger.info(f"{'Metric':<20} {baseline_name:<12} {model_name:<12} {'Improvement':<12}")
        self.logger.info("-" * 60)
        
        for metric in sorted(baseline_metrics.keys()):
            if metric in model_metrics:
                baseline_val = baseline_metrics[metric]
                model_val = model_metrics[metric]
                
                # Calculate improvement (negative for metrics where lower is better)
                if metric in ['crps', 'mae', 'rmse', 'calibration_error'] or metric.startswith('brier_'):
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement = model_val - baseline_val
                    improvement_str = f"{improvement:+.3f}"
                
                self.logger.info(
                    f"{metric:<20} {baseline_val:>11.4f} {model_val:>11.4f} {improvement_str:>11}"
                )


# Convenience function for quick setup
def quick_setup(level: str = "INFO") -> logging.Logger:
    """
    Quick logging setup for scripts.
    
    Args:
        level: Log level
        
    Returns:
        Root logger
    """
    setup_logging(log_level=level, use_json=False)
    return logging.getLogger()