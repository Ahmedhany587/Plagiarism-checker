"""
Input validation and error handling framework for the Smart Document Analyzer.

This module provides comprehensive validation utilities, custom exceptions,
and error handling patterns for production use.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union, Dict, Callable
import mimetypes
from functools import wraps


# Custom Exception Classes
class ValidationError(Exception):
    """Base class for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class FileValidationError(ValidationError):
    """Exception raised for file-related validation errors."""
    pass


class DirectoryValidationError(ValidationError):
    """Exception raised for directory-related validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Exception raised for parameter validation errors."""
    pass


class SecurityValidationError(ValidationError):
    """Exception raised for security-related validation errors."""
    pass


# Validation Utilities
class FileValidator:
    """Comprehensive file validation utilities."""
    
    ALLOWED_PDF_EXTENSIONS = {'.pdf'}
    ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    MAX_FILE_SIZE_MB = 100  # 100MB default limit
    MAX_FILENAME_LENGTH = 255
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], 
                          must_exist: bool = True,
                          allowed_extensions: Optional[set] = None,
                          max_size_mb: Optional[float] = None) -> Path:
        """
        Simple file path validation.
        
        Args:
            file_path: Path to the file
            must_exist: Whether the file must exist
            allowed_extensions: Set of allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            Path object
            
        Raises:
            FileValidationError: If validation fails
        """
        if not file_path:
            raise FileValidationError("File path cannot be empty", field="file_path", value=file_path)
        
        path = Path(file_path)
        
        # Check if file exists (if required)
        if must_exist and not path.exists():
            raise FileValidationError(f"File does not exist: {file_path}", field="file_path", value=file_path)
        
        # Check if it's actually a file (not a directory)
        if must_exist and path.exists() and not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}", field="file_path", value=file_path)
        
        # Check file extension
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            raise FileValidationError(
                f"File extension not allowed. Allowed: {allowed_extensions}, got: {path.suffix}",
                field="file_path",
                value=file_path
            )
        
        # Check file size
        if must_exist and path.exists() and max_size_mb:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise FileValidationError(
                    f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)",
                    field="file_path",
                    value=file_path
                )
        
        return path
    
    @staticmethod
    def validate_pdf_file(file_path: Union[str, Path]) -> Path:
        """Validate PDF file specifically."""
        return FileValidator.validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=FileValidator.ALLOWED_PDF_EXTENSIONS,
            max_size_mb=FileValidator.MAX_FILE_SIZE_MB
        )
    
    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> Path:
        """Validate image file specifically."""
        return FileValidator.validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=FileValidator.ALLOWED_IMAGE_EXTENSIONS,
            max_size_mb=10  # Smaller limit for images
        )


class DirectoryValidator:
    """Directory validation utilities."""
    
    @staticmethod
    def validate_directory_path(dir_path: Union[str, Path], 
                               must_exist: bool = True,
                               must_be_readable: bool = True,
                               must_be_writable: bool = False) -> Path:
        """
        Simple directory path validation.
        
        Args:
            dir_path: Path to the directory
            must_exist: Whether the directory must exist
            must_be_readable: Whether the directory must be readable
            must_be_writable: Whether the directory must be writable
            
        Returns:
            Path object
            
        Raises:
            DirectoryValidationError: If validation fails
        """
        if not dir_path:
            raise DirectoryValidationError("Directory path cannot be empty", field="dir_path", value=dir_path)
        
        path = Path(dir_path)
        
        # Check if directory exists (if required)
        if must_exist and not path.exists():
            raise DirectoryValidationError(f"Directory does not exist: {dir_path}", field="dir_path", value=dir_path)
        
        # Check if it's actually a directory
        if must_exist and path.exists() and not path.is_dir():
            raise DirectoryValidationError(f"Path is not a directory: {dir_path}", field="dir_path", value=dir_path)
        
        return path


class ParameterValidator:
    """Parameter validation utilities."""
    
    @staticmethod
    def validate_positive_integer(value: Any, field: str, min_value: int = 1, max_value: Optional[int] = None) -> int:
        """Validate positive integer parameter."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ParameterValidationError(
                    f"{field} must be an integer, got {type(value).__name__}",
                    field=field,
                    value=value
                )
        
        if value < min_value:
            raise ParameterValidationError(
                f"{field} must be >= {min_value}, got {value}",
                field=field,
                value=value
            )
        
        if max_value is not None and value > max_value:
            raise ParameterValidationError(
                f"{field} must be <= {max_value}, got {value}",
                field=field,
                value=value
            )
        
        return value
    
    @staticmethod
    def validate_positive_float(value: Any, field: str, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
        """Validate positive float parameter."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ParameterValidationError(
                    f"{field} must be a number, got {type(value).__name__}",
                    field=field,
                    value=value
                )
        
        if value < min_value:
            raise ParameterValidationError(
                f"{field} must be >= {min_value}, got {value}",
                field=field,
                value=value
            )
        
        if max_value is not None and value > max_value:
            raise ParameterValidationError(
                f"{field} must be <= {max_value}, got {value}",
                field=field,
                value=value
            )
        
        return float(value)
    
    @staticmethod
    def validate_string(value: Any, field: str, min_length: int = 0, max_length: Optional[int] = None, 
                       pattern: Optional[str] = None) -> str:
        """Validate string parameter."""
        if not isinstance(value, str):
            raise ParameterValidationError(
                f"{field} must be a string, got {type(value).__name__}",
                field=field,
                value=value
            )
        
        if len(value) < min_length:
            raise ParameterValidationError(
                f"{field} must be at least {min_length} characters, got {len(value)}",
                field=field,
                value=value
            )
        
        if max_length is not None and len(value) > max_length:
            raise ParameterValidationError(
                f"{field} must be at most {max_length} characters, got {len(value)}",
                field=field,
                value=value
            )
        
        if pattern and not re.match(pattern, value):
            raise ParameterValidationError(
                f"{field} does not match required pattern",
                field=field,
                value=value
            )
        
        return value


# Decorators for validation
def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dict mapping parameter names to validation functions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except ValidationError:
                        raise
                    except Exception as e:
                        raise ParameterValidationError(
                            f"Validation failed for {param_name}: {str(e)}",
                            field=param_name,
                            value=value
                        )
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def handle_exceptions(default_return=None, reraise_types=None):
    """
    Decorator to handle exceptions gracefully.
    
    Args:
        default_return: Default value to return on exception
        reraise_types: List of exception types to re-raise
    """
    if reraise_types is None:
        reraise_types = [ValidationError, SecurityValidationError]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tuple(reraise_types):
                raise
            except Exception as e:
                # Log the exception
                import logging
                logger = logging.getLogger(func.__module__)
                logger.error(f"Unhandled exception in {func.__name__}: {str(e)}", exc_info=True)
                
                if default_return is not None:
                    return default_return
                raise
        return wrapper
    return decorator 