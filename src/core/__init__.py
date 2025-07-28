"""
Core functionality for document analysis and similarity detection.

This package contains the main algorithms and processing logic:
- PDF handling and text extraction
- AI embedding generation
- Semantic similarity analysis  
- Sequence similarity analysis
- Exact match detection
"""

from .pdf_handler import PDFHandler
from .embedding_generator import EmbeddingGenerator
from .semantic_similarity import PDFSimilarityCalculator
from .sequence_similarity import SequenceSimilarityCalculator
from .exact_match import ExactMatchDetector
from .logging_config import setup_logging, get_logger, LoggerMixin, ProductionLogger
from .validation import (
    ValidationError, FileValidationError, DirectoryValidationError,
    ParameterValidationError, SecurityValidationError,
    FileValidator, DirectoryValidator, ParameterValidator,
    validate_inputs, handle_exceptions
)

__all__ = [
    'PDFHandler',
    'EmbeddingGenerator', 
    'PDFSimilarityCalculator',
    'SequenceSimilarityCalculator',
    'ExactMatchDetector',
    'setup_logging',
    'get_logger',
    'LoggerMixin',
    'ProductionLogger',
    'ValidationError',
    'FileValidationError',
    'DirectoryValidationError',
    'ParameterValidationError',
    'SecurityValidationError',
    'FileValidator',
    'DirectoryValidator',
    'ParameterValidator',
    'validate_inputs',
    'handle_exceptions'
] 