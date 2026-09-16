"""
utils/logger.py

Centralized logging system for PSD-HP.

Every module in the framework obtains its logger by calling
get_module_logger(module_name, logs_dir) instead of configuring its
own logging.Logger. This guarantees:

  1. One consistent log message format across the entire framework.
  2. One dedicated log file per module, e.g. logs/validator.log,
     logs/scanner.log, logs/quality.log — exactly as required.
  3. One shared logs/pipeline.log containing every module's messages
     in chronological order, giving a single end-to-end trace of a
     full harmonization run without needing to stitch files together.

No other PSD-HP module should ever call logging.basicConfig() or
logging.getLogger() directly — this file is the only place logging
handlers are attached.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tracks which module names have already had handlers attached, so
# calling get_module_logger() twice for the same module (e.g. once in
# main.py, once inside a stage) never duplicates log lines.
_configured_loggers: set[str] = set()


def get_module_logger(module_name: str, logs_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Return a logger dedicated to one PSD-HP module.

    Parameters
    ----------
    module_name : str
        Short identifier for the calling module, e.g. "validator",
        "scanner". Used as the logger's name (prefixed with
        "psd_hp.") and as the log file's stem: logs/<module_name>.log.
    logs_dir : Path
        Directory where log files are written. Created if it does
        not already exist.
    level : str
        Logging level name, e.g. "INFO" or "DEBUG". Defaults to INFO.

    Returns
    -------
    logging.Logger
        A logger that writes to the console, to
        logs/<module_name>.log, and to the shared logs/pipeline.log.
    """
    logger = logging.getLogger(f"psd_hp.{module_name}")

    if module_name in _configured_loggers:
        # Already fully configured earlier in this process — return
        # the existing logger as-is rather than attaching duplicate
        # handlers.
        return logger

    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # don't also send messages to the root logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 1) Module-specific log file, e.g. logs/validator.log
    module_handler = logging.FileHandler(
        logs_dir / f"{module_name}.log", mode="a", encoding="utf-8"
    )
    module_handler.setFormatter(formatter)
    logger.addHandler(module_handler)

    # 2) Shared master trace across every module, in run order
    master_handler = logging.FileHandler(
        logs_dir / "pipeline.log", mode="a", encoding="utf-8"
    )
    master_handler.setFormatter(formatter)
    logger.addHandler(master_handler)

    # 3) Console output so `python main.py` shows live progress
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _configured_loggers.add(module_name)
    return logger
