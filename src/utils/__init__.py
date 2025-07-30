"""
Utility functions and helpers for document analysis.

This package contains supporting utilities:
- Image extraction (pdf_img_extractor)
- Image duplicate detection (image_duplication_detector)
- Helper functions and tools
"""

from .pdf_img_extractor import PDFImageExtractor
from .image_duplication_detector import (
    ImageDuplicationDetector,
    build_dataset_from_results, 
    index_and_report_cross_pdf_duplicates
)

__all__ = [
    'PDFImageExtractor',
    'ImageDuplicationDetector',
    'build_dataset_from_results', 
    'index_and_report_cross_pdf_duplicates'
] 