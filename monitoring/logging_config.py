"""
Logging configuration for WIZ Video Intelligence Pipeline
"""

import logging
import logging.handlers
from pathlib import Path

def setup_pipeline_logging(log_dir: str = "logs", level: int = logging.INFO):
    """
    Setup centralized logging for the entire pipeline
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (INFO, DEBUG, WARNING, ERROR)
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Detailed file handler (with rotation)
    detailed_handler = logging.handlers.RotatingFileHandler(
        log_path / 'pipeline_detailed.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(detailed_handler)
    
    # Simple file handler (with rotation)
    simple_handler = logging.handlers.RotatingFileHandler(
        log_path / 'pipeline_simple.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    simple_handler.setLevel(logging.INFO)
    simple_handler.setFormatter(simple_formatter)
    root_logger.addHandler(simple_handler)
    
    # Error-only file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / 'pipeline_errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    logging.info("Pipeline logging configured successfully")
    
    return root_logger