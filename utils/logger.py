"""Logger Module.

Provides logging utilities for RL-LLM Agent.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class LoggerConfig:
    """Configuration for logging.

    Attributes:
        level: Logging level
        format: Log format string
        log_dir: Directory for log files
        console_output: Whether to output to console
        file_output: Whether to output to file
        tensorboard_dir: Directory for TensorBoard logs
        wandb_enabled: Whether to enable Weights & Biases
        wandb_project: W&B project name
    """
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    log_dir: str = "./logs"
    console_output: bool = True
    file_output: bool = True
    tensorboard_dir: Optional[str] = None
    wandb_enabled: bool = False
    wandb_project: str = "rl-llm-agent"


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record

        Returns:
            Formatted log message
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted log message
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class LoggerManager:
    """Manages logging configuration and instances."""

    _loggers: Dict[str, logging.Logger] = {}
    _config: Optional[LoggerConfig] = None

    @classmethod
    def configure(cls, config: Optional[LoggerConfig] = None) -> None:
        """Configure logging system.

        Args:
            config: Logger configuration
        """
        cls._config = config or LoggerConfig()

        # Create log directory
        if cls._config.file_output:
            os.makedirs(cls._config.log_dir, exist_ok=True)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)

        # Set level
        level = getattr(logging, cls._config.level.upper(), logging.INFO)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers = []

        # Console handler
        if cls._config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            # Use colored formatter for console
            console_formatter = ColoredFormatter(cls._config.format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if cls._config.file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(
                cls._config.log_dir,
                f"{name}_{timestamp}.log"
            )

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)

            # Use standard formatter for files
            file_formatter = logging.Formatter(cls._config.format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Also write JSON logs
            json_file = os.path.join(
                cls._config.log_dir,
                f"{name}_{timestamp}.jsonl"
            )
            json_handler = logging.FileHandler(json_file, encoding="utf-8")
            json_handler.setLevel(level)
            json_handler.setFormatter(JsonFormatter())
            logger.addHandler(json_handler)

        cls._loggers[name] = logger
        return logger


# Global logger instance
_default_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "rl-llm-agent",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Setup and get a logger.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Whether to output to console
        file: Whether to output to file

    Returns:
        Logger instance
    """
    config = LoggerConfig(
        level=level,
        log_dir=log_dir or "./logs",
        console_output=console,
        file_output=file,
    )

    LoggerManager.configure(config)
    return LoggerManager.get_logger(name)


def get_logger(name: str = "rl-llm-agent") -> logging.Logger:
    """Get an existing logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return LoggerManager.get_logger(name)


def get_default_logger() -> logging.Logger:
    """Get the default logger.

    Returns:
        Default logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


# Convenience logging functions
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message."""
    get_default_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log info message."""
    get_default_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message."""
    get_default_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log error message."""
    get_default_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message."""
    get_default_logger().critical(msg, *args, **kwargs)


class TrainingLogger:
    """Specialized logger for training metrics."""

    def __init__(
        self,
        log_dir: str = "./logs",
        project_name: str = "rl-llm-agent"
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory for logs
            project_name: Project name
        """
        self.log_dir = log_dir
        self.project_name = project_name
        self._logger = setup_logger("training", log_dir=log_dir)
        self._metrics_history: Dict[str, List[float]] = {}
        self._writer = None

        # Try to import TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._writer = SummaryWriter(f"{log_dir}/tensorboard/{timestamp}")
            self._logger.info("TensorBoard logging enabled")
        except ImportError:
            self._logger.warning("TensorBoard not available")

        # Try to import wandb
        self._wandb_enabled = False
        try:
            import wandb
            wandb.init(project=project_name)
            self._wandb_enabled = True
            self._logger.info("W&B logging enabled")
        except ImportError:
            pass

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        # Log to file
        self._logger.info(f"[Step {step}] {name}: {value:.6f}")

        # Store in history
        if name not in self._metrics_history:
            self._metrics_history[name] = []
        self._metrics_history[name].append(value)

        # Log to TensorBoard
        if self._writer:
            self._writer.add_scalar(name, value, step)

        # Log to wandb
        if self._wandb_enabled:
            import wandb
            wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def log_histogram(self, name: str, values: Union[list, Any], step: int) -> None:
        """Log a histogram.

        Args:
            name: Metric name
            values: Values to histogram
            step: Training step
        """
        if self._writer:
            try:
                import torch
                if isinstance(values, list):
                    values = torch.tensor(values)
                self._writer.add_histogram(name, values, step)
            except Exception as e:
                self._logger.warning(f"Failed to log histogram {name}: {e}")

    def log_image(self, name: str, image: Any, step: int) -> None:
        """Log an image.

        Args:
            name: Image name
            image: Image tensor/array
            step: Training step
        """
        if self._writer:
            try:
                import torch
                self._writer.add_image(name, image, step)
            except Exception as e:
                self._logger.warning(f"Failed to log image {name}: {e}")

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text.

        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        if self._writer:
            try:
                self._writer.add_text(name, text, step)
            except Exception as e:
                self._logger.warning(f"Failed to log text {name}: {e}")

    def get_metric_history(self, name: str) -> List[float]:
        """Get history for a metric.

        Args:
            name: Metric name

        Returns:
            List of metric values
        """
        return self._metrics_history.get(name, [])

    def get_all_history(self) -> Dict[str, List[float]]:
        """Get all metric history.

        Returns:
            Dictionary of metric histories
        """
        return self._metrics_history.copy()

    def close(self) -> None:
        """Close the logger."""
        if self._writer:
            self._writer.close()
        if self._wandb_enabled:
            import wandb
            wandb.finish()


class EpisodeLogger:
    """Logger for episode-level statistics."""

    def __init__(self):
        """Initialize episode logger."""
        self._episodes: List[Dict[str, Any]] = []
        self._current_episode: Dict[str, Any] = {}

    def start_episode(self, episode_id: int) -> None:
        """Start a new episode.

        Args:
            episode_id: Episode ID
        """
        self._current_episode = {
            "episode_id": episode_id,
            "start_time": datetime.now().isoformat(),
            "steps": [],
        }

    def log_step(
        self,
        observation: str,
        action: str,
        reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a step within an episode.

        Args:
            observation: Environment observation
            action: Agent action
            reward: Reward received
            info: Optional info dictionary
        """
        step_data = {
            "observation": observation[:200],  # Truncate long observations
            "action": action,
            "reward": reward,
            "info": info or {},
        }
        self._current_episode["steps"].append(step_data)

    def end_episode(
        self,
        success: bool,
        total_reward: float,
        length: int,
        info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """End the current episode.

        Args:
            success: Whether episode was successful
            total_reward: Total episode reward
            length: Episode length
            info: Optional info dictionary

        Returns:
            Episode summary
        """
        self._current_episode["end_time"] = datetime.now().isoformat()
        self._current_episode["success"] = success
        self._current_episode["total_reward"] = total_reward
        self._current_episode["length"] = length
        self._current_episode["info"] = info or {}

        # Compute summary
        summary = {
            "episode_id": self._current_episode["episode_id"],
            "success": success,
            "total_reward": total_reward,
            "length": length,
            "num_steps": len(self._current_episode["steps"]),
        }

        self._episodes.append(self._current_episode.copy())
        self._current_episode = {}

        return summary

    def get_episode(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific episode.

        Args:
            episode_id: Episode ID

        Returns:
            Episode data or None
        """
        for ep in self._episodes:
            if ep.get("episode_id") == episode_id:
                return ep
        return None

    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodes.

        Args:
            n: Number of episodes

        Returns:
            List of episodes
        """
        return self._episodes[-n:]

    def clear(self) -> None:
        """Clear all episodes."""
        self._episodes = []
