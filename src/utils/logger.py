import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
from logging.handlers import RotatingFileHandler
import threading

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and structured format"""
    
    # Color codes
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[34;20m"
    white = "\x1b[37;20m"
    green = "\x1b[32;20m"
    
    # Format string
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'file': record.filename,
            'line': record.lineno,
            'thread': threading.current_thread().name,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        return json.dumps(log_data)

class Logger:
    """Custom logger class with both console and file output"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('STRV-SimilaritySearch')
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers multiple times
        if self.logger.handlers:
            return
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter())
        
        # File handler with JSON formatting (regular logs)
        regular_file_handler = RotatingFileHandler(
            filename=log_dir / 'app.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        regular_file_handler.setLevel(logging.DEBUG)
        regular_file_handler.setFormatter(JSONFormatter())
        
        # Separate file handler for errors
        error_file_handler = RotatingFileHandler(
            filename=log_dir / 'error.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(JSONFormatter())
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(regular_file_handler)
        self.logger.addHandler(error_file_handler)
    
    def add_extra_fields(self, **kwargs):
        """Add extra fields to the log output"""
        extra = {'extra_fields': kwargs}
        return logging.LoggerAdapter(self.logger, extra)
    
    @property
    def get_logger(self):
        """Get the configured logger instance"""
        return self.logger

# Create global logger instance
logger = Logger().get_logger

# Example usage function
def log_example():
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Log with extra fields
    custom_logger = Logger().add_extra_fields(user_id="123", action="example")
    custom_logger.info("This is a message with extra fields")
    
    # Log an exception
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.exception("An error occurred")