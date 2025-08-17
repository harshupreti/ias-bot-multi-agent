# logger_config.py

import logging
from pathlib import Path
from config import LOG_DIR

def setup_logger(name="IASPipeline") -> logging.Logger:
    """
    Sets up a logger that outputs to both console and a log file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "pipeline.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if re-imported
    if logger.hasHandlers():
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
