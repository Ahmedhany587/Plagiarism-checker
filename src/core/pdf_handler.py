import os
import fitz  # PyMuPDF
import unicodedata
import locale
from typing import List, Dict
from pathlib import Path
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.document.chunking.document import DocumentChunking
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging_config import LoggerMixin
from .validation import (
    DirectoryValidator, FileValidator, ParameterValidator,
    DirectoryValidationError, FileValidationError, ParameterValidationError,
    validate_inputs, handle_exceptions
)


def normalize_filename(filename: str) -> str:
    """
    Normalize filename for consistent handling of Unicode characters including Arabic.
    
    Args:
        filename: Original filename string
        
    Returns:
        Normalized filename string
    """
    # Normalize Unicode to NFC (Canonical Decomposition, followed by Canonical Composition)
    # This ensures consistent representation of Arabic characters
    normalized = unicodedata.normalize('NFC', filename)
    return normalized


def safe_filename_encode(filename: str) -> str:
    """
    Safely encode filename for logging and display purposes.
    
    Args:
        filename: Original filename
        
    Returns:
        Safely encoded filename string
    """
    try:
        # Normalize the filename first
        normalized = normalize_filename(filename)
        # Ensure it can be encoded/decoded properly
        return normalized.encode('utf-8', errors='replace').decode('utf-8')
    except (UnicodeError, AttributeError):
        # Fallback to ASCII representation if there are encoding issues
        return repr(filename)


@handle_exceptions(default_return=[])
def find_pdf_files(directory_path: str) -> List[str]:
    """
    Recursively walk a directory tree and return full paths to all .pdf files.
    Enhanced to properly handle Arabic and Unicode filenames.
    
    Args:
        directory_path: Path to directory to search
        
    Returns:
        List of PDF file paths with proper Unicode handling
        
    Raises:
        DirectoryValidationError: If directory is invalid or inaccessible
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Set UTF-8 locale for proper Unicode handling
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            # Fallback - continue without locale setting
            logger.debug("Could not set UTF-8 locale, continuing with default")
    
    # Validate directory path
    validated_path = DirectoryValidator.validate_directory_path(
        directory_path, 
        must_exist=True, 
        must_be_readable=True
    )
    
    safe_path = safe_filename_encode(str(validated_path))
    logger.info(f"Searching for PDF files in directory: {safe_path}")
    
    pdf_files = []
    try:
        for root, _, files in os.walk(validated_path):
            for file in files:
                try:
                    # Normalize the filename for consistent handling
                    normalized_file = normalize_filename(file)
                    
                    if normalized_file.lower().endswith('.pdf'):
                        full_path = os.path.join(root, file)  # Use original filename for path
                        try:
                            # Validate each PDF file
                            FileValidator.validate_pdf_file(full_path)
                            pdf_files.append(full_path)
                            
                            safe_path_display = safe_filename_encode(full_path)
                            logger.debug(f"Found valid PDF: {safe_path_display}")
                        except FileValidationError as e:
                            safe_path_display = safe_filename_encode(full_path)
                            logger.warning(f"Skipping invalid PDF {safe_path_display}: {e.message}")
                            continue
                except (UnicodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Unicode error processing filename: {repr(file)}: {str(e)}")
                    continue
    except OSError as e:
        safe_path_display = safe_filename_encode(str(validated_path))
        logger.error(f"Error accessing directory {safe_path_display}: {str(e)}")
        raise DirectoryValidationError(
            f"Cannot access directory: {str(e)}", 
            field="directory_path", 
            value=directory_path
        )
    
    logger.info(f"Found {len(pdf_files)} valid PDF files")
    return pdf_files


class PDFHandler(LoggerMixin):
    """
    A class to handle PDF documents in a directory by extracting and chunking
    each PDF page-wise (1 page == 1 chunk).
    Enhanced to properly handle Arabic and Unicode filenames.
    """

    @validate_inputs(
        dir_path=lambda x: DirectoryValidator.validate_directory_path(x, must_exist=True, must_be_readable=True)
    )
    def __init__(self, dir_path: str):
        """
        Initializes the PDFHandler with the directory containing PDF files.
        Enhanced with Unicode filename support.

        Args:
            dir_path: Path to the directory with PDF files.
            
        Raises:
            DirectoryValidationError: If directory is invalid or inaccessible
        """
        safe_dir_path = safe_filename_encode(str(dir_path))
        with self.log_operation("pdf_handler_init", directory=safe_dir_path):
            self.dir_path = Path(dir_path)
            self.logger.info(f"Initializing PDFHandler with directory: {safe_dir_path}")
            
            # Gather all PDF files in the directory and subdirectories
            self.pdf_files = find_pdf_files(str(self.dir_path))
            
            if not self.pdf_files:
                self.logger.warning(f"No PDF files found in directory: {safe_dir_path}")
            else:
                self.logger.info(f"Successfully initialized PDFHandler with {len(self.pdf_files)} PDF files")

    def get_pdf_count(self) -> int:
        """
        Returns the count of PDF files in the directory.

        Returns:
            Number of PDF files.
        """
        count = len(self.pdf_files)
        self.logger.debug(f"PDF count requested: {count}")
        return count

    @handle_exceptions(default_return={})
    def get_page_counts(self) -> Dict[str, int]:
        """
        Returns the number of pages in each PDF file.
        Enhanced to properly handle Arabic filenames in the returned dictionary.

        Returns:
            A dictionary mapping PDF file names (normalized) to their page counts.
            
        Raises:
            FileValidationError: If a PDF file cannot be accessed
        """
        with self.log_operation("get_page_counts"):
            page_counts = {}
            
            for pdf_path in self.pdf_files:
                try:
                    with fitz.open(pdf_path) as doc:
                        original_filename = os.path.basename(pdf_path)
                        # Normalize filename for consistent dictionary keys
                        normalized_filename = normalize_filename(original_filename)
                        count = doc.page_count
                        page_counts[normalized_filename] = count
                        
                        safe_filename = safe_filename_encode(normalized_filename)
                        self.logger.debug(f"PDF {safe_filename} has {count} pages")
                        
                except Exception as e:
                    safe_path = safe_filename_encode(pdf_path)
                    self.logger.error(f"Failed to get page count for {safe_path}: {str(e)}")
                    raise FileValidationError(
                        f"Cannot access PDF file: {str(e)}", 
                        field="pdf_path", 
                        value=pdf_path
                    )
            
            self.logger.info(f"Retrieved page counts for {len(page_counts)} PDF files")
            return page_counts

    def _extract_chunks_from_pdf(self, pdf_path: str, chunk_size: int) -> tuple:
        """
        Extract text chunks from a single PDF file.
        Enhanced to properly handle Arabic filenames.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks to create
            
        Returns:
            Tuple of (normalized_filename, list_of_chunks)
        """
        original_filename = os.path.basename(pdf_path)
        normalized_filename = normalize_filename(original_filename)
        safe_filename = safe_filename_encode(normalized_filename)
        
        try:
            # Validate PDF file before processing
            FileValidator.validate_pdf_file(pdf_path)
            
            self.logger.debug(f"Extracting chunks from {safe_filename} with chunk size {chunk_size}")
            
            kb = PDFKnowledgeBase(
                path=pdf_path,
                reader=PDFReader(),
                chunking_strategy=DocumentChunking(chunk_size=chunk_size),
            )
            
            documents = kb.reader.read(kb.path)
            file_chunks = []
            
            for doc in documents:
                for chunk in kb.chunking_strategy.chunk(doc):
                    if chunk.content and chunk.content.strip():
                        file_chunks.append(chunk.content.strip())
            
            self.logger.info(f"Successfully extracted {len(file_chunks)} chunks from {safe_filename}")
            return normalized_filename, file_chunks
            
        except FileValidationError:
            self.logger.error(f"File validation failed for {safe_filename}")
            raise
        except Exception as e:
            safe_path = safe_filename_encode(pdf_path)
            self.logger.error(f"Error processing {safe_filename}: {str(e)}", 
                            extra={'file_path': safe_path})
            # Return empty chunks instead of failing completely
            return normalized_filename, []

    @validate_inputs(
        chunk_size=lambda x: ParameterValidator.validate_positive_integer(x, "chunk_size", min_value=100, max_value=50000)
    )
    @handle_exceptions(default_return={})
    def extract_page_chunks(self, chunk_size: int = 5000) -> Dict[str, List[str]]:
        """
        Extracts text chunks from each PDF using DocumentChunking from phidata, in parallel.
        Enhanced to properly handle Arabic filenames in the returned dictionary.
        
        Args:
            chunk_size: Size of text chunks to create (100-50000 characters)
            
        Returns:
            Dictionary mapping normalized PDF filenames to lists of text chunks
            
        Raises:
            ParameterValidationError: If chunk_size is invalid
            FileValidationError: If PDF files cannot be processed
        """
        with self.log_operation("extract_page_chunks", chunk_size=chunk_size, file_count=len(self.pdf_files)):
            if not self.pdf_files:
                self.logger.warning("No PDF files to process")
                return {}
            
            chunks = {}
            successful_files = 0
            failed_files = 0
            
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._extract_chunks_from_pdf, pdf_path, chunk_size) 
                    for pdf_path in self.pdf_files
                ]
                
                for future in as_completed(futures):
                    try:
                        normalized_filename, file_chunks = future.result()
                        chunks[normalized_filename] = file_chunks
                        
                        safe_filename = safe_filename_encode(normalized_filename)
                        if file_chunks:
                            successful_files += 1
                            self.logger.debug(f"Successfully processed {safe_filename}: {len(file_chunks)} chunks")
                        else:
                            failed_files += 1
                            self.logger.warning(f"No chunks extracted from {safe_filename}")
                            
                    except Exception as e:
                        failed_files += 1
                        self.logger.error(f"Failed to process PDF: {str(e)}")
            
            self.logger.info(f"Chunk extraction completed: {successful_files} successful, {failed_files} failed")
            return chunks

    def get_pdf_filenames(self) -> List[str]:
        """
        Get a list of normalized PDF filenames for display purposes.
        
        Returns:
            List of normalized PDF filenames
        """
        filenames = []
        for pdf_path in self.pdf_files:
            original_filename = os.path.basename(pdf_path)
            normalized_filename = normalize_filename(original_filename)
            filenames.append(normalized_filename)
        
        self.logger.debug(f"Retrieved {len(filenames)} PDF filenames")
        return filenames

    def get_pdf_info(self) -> Dict[str, Dict[str, any]]:
        """
        Get comprehensive information about all PDF files.
        
        Returns:
            Dictionary with PDF info including original path, normalized name, etc.
        """
        pdf_info = {}
        
        for pdf_path in self.pdf_files:
            original_filename = os.path.basename(pdf_path)
            normalized_filename = normalize_filename(original_filename)
            
            pdf_info[normalized_filename] = {
                'original_path': pdf_path,
                'original_filename': original_filename,
                'normalized_filename': normalized_filename,
                'safe_display_name': safe_filename_encode(normalized_filename)
            }
        
        return pdf_info
