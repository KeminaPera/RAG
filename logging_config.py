import os
import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "app.log") -> logging.Logger:
    log_dir = Path(os.path.dirname(__file__)) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)