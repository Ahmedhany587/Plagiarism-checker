"""
Centralized logging configuration for the Smart Document Analyzer.

This module provides structured logging with proper levels, formatting,
and production-ready configurations.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'file_path'):
            log_entry['file_path'] = record.file_path
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
            
        # Ensure proper Unicode handling for Arabic text in JSON
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))


class ProductionLogger:
    """Production-ready logger configuration."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 structured_logging: bool = True):
        """
        Initialize production logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            max_file_size: Maximum size of each log file in bytes
            backup_count: Number of backup files to keep
            enable_console: Whether to log to console
            enable_file: Whether to log to files
            structured_logging: Whether to use structured JSON logging
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.structured_logging = structured_logging
        
        # Create logs directory
        if self.enable_file:
            self.log_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        root_logger.setLevel(self.log_level)
        
        # Setup formatters
        if self.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file:
            # Main application log
            app_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "app.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'  # Ensure UTF-8 encoding for Arabic text
            )
            app_handler.setLevel(self.log_level)
            app_handler.setFormatter(formatter)
            root_logger.addHandler(app_handler)
            
            # Error log (WARNING and above)
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "errors.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'  # Ensure UTF-8 encoding for Arabic text
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance with the given name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger
    
    def log_operation(self, operation: str, **kwargs):
        """Log an operation with additional context."""
        extra = {'operation': operation}
        extra.update(kwargs)
        return OperationLogger(self.logger, extra)


class OperationLogger:
    """Context manager for logging operations with timing."""
    
    def __init__(self, logger: logging.Logger, extra: dict):
        self.logger = logger
        self.extra = extra
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting operation: {self.extra.get('operation', 'unknown')}", extra=self.extra)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        self.extra['duration'] = duration
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.extra.get('operation', 'unknown')}", extra=self.extra)
        else:
            self.logger.error(f"Failed operation: {self.extra.get('operation', 'unknown')}", 
                            extra=self.extra, exc_info=True)


# Global logger instance
_production_logger: Optional[ProductionLogger] = None


def setup_logging(log_level: str = "INFO", 
                 log_dir: str = "logs",
                 structured_logging: bool = True,
                 **kwargs) -> ProductionLogger:
    """
    Setup global logging configuration.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        structured_logging: Whether to use structured JSON logging
        **kwargs: Additional arguments for ProductionLogger
        
    Returns:
        ProductionLogger instance
    """
    global _production_logger
    _production_logger = ProductionLogger(
        log_level=log_level,
        log_dir=log_dir,
        structured_logging=structured_logging,
        **kwargs
    )
    return _production_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    if _production_logger is None:
        setup_logging()
    return _production_logger.get_logger(name) 