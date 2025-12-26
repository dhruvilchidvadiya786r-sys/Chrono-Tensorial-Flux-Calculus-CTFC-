"""
ctfc.utils.logging
==================

Structured logging utilities for Chrono-Tensorial Flux Calculus (CTFC).

This module defines a unified logging interface for:
- experiment runs,
- configuration tracking,
- operator-level diagnostics,
- reproducibility metadata.

Design goals
------------
- Deterministic output
- Human-readable logs
- Machine-parsable structure
- Zero impact on core math

CTFC logging is *observational*, never causal.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

# ============================================================
# Logger configuration
# ============================================================

DEFAULT_LOGGER_NAME = "ctfc"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] "
    "[%(name)s] %(message)s"
)


# ============================================================
# Logger creation
# ============================================================


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create or retrieve a CTFC logger.

    Parameters
    ----------
    name : str, default="ctfc"
        Logger name.

    level : int, default=logging.INFO
        Logging level.

    log_file : str or None
        Optional path to a log file.
        If provided, logs are written to both console and file.

    Returns
    -------
    logger : logging.Logger

    Notes
    -----
    - Safe to call multiple times
    - Handlers are added only once
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ============================================================
# Structured logging helpers
# ============================================================


def log_config(
    logger: logging.Logger,
    config: Dict[str, Any],
) -> None:
    """
    Log CTFC configuration parameters in structured form.

    Parameters
    ----------
    logger : logging.Logger
        CTFC logger.

    config : dict
        Configuration dictionary.
    """

    logger.info("CTFC configuration:")
    for key, value in sorted(config.items()):
        logger.info(f"  {key}: {value}")


def log_experiment_signature(
    logger: logging.Logger,
    signature: Dict[str, str],
) -> None:
    """
    Log experiment reproducibility signature.

    Parameters
    ----------
    logger : logging.Logger
        CTFC logger.

    signature : dict
        Output of experiment_signature().
    """

    logger.info("Experiment signature:")
    for key, value in signature.items():
        logger.info(f"  {key}: {value}")


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    *,
    prefix: str = "Metric",
) -> None:
    """
    Log scalar metrics in a standardized format.

    Parameters
    ----------
    logger : logging.Logger
        CTFC logger.

    metrics : dict[str, float]
        Metrics dictionary.

    prefix : str, default="Metric"
        Prefix for log entries.
    """

    for key, value in sorted(metrics.items()):
        logger.info(f"{prefix} | {key}: {value:.6f}")


# ============================================================
# JSON logging (machine-readable)
# ============================================================


def write_json_log(
    path: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    signature: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a structured JSON log file.

    Parameters
    ----------
    path : str
        Output file path.

    config : dict or None
        CTFC configuration.

    metrics : dict or None
        Evaluation metrics.

    signature : dict or None
        Experiment signature.

    extra : dict or None
        Additional metadata.

    Notes
    -----
    - JSON logs are immutable snapshots
    - Intended for experiment tracking and archiving
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "metrics": metrics,
        "signature": signature,
        "extra": extra,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# ============================================================
# Context-style experiment logging
# ============================================================


class CTFCLogger:
    """
    Context manager for structured CTFC experiment logging.

    Example
    -------
    >>> logger = CTFCLogger("logs/run.log")
    >>> with logger as log:
    >>>     log.info("Starting CTFC run")
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        *,
        level: int = DEFAULT_LOG_LEVEL,
    ) -> None:
        self.logger = get_logger(
            level=level,
            log_file=log_file,
        )

    def __enter__(self) -> logging.Logger:
        self.logger.info("CTFC run started")
        return self.logger

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.logger.error(
                "CTFC run terminated with exception",
                exc_info=(exc_type, exc, tb),
            )
        else:
            self.logger.info("CTFC run completed successfully")

