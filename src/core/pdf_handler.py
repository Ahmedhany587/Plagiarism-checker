import os
import fitz  # PyMuPDF
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


@handle_exceptions(default_return=[])
def find_pdf_files(directory_path: str) -> List[str]:
    """
    Recursively walk a directory tree and return full paths to all .pdf files.
    
    Args:
        directory_path: Path to directory to search
        
    Returns:
        List of PDF file paths
        
    Raises:
        DirectoryValidationError: If directory is invalid or inaccessible
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate directory path
    validated_path = DirectoryValidator.validate_directory_path(
        directory_path, 
        must_exist=True, 
        must_be_readable=True
    )
    
    logger.info(f"Searching for PDF files in directory: {validated_path}")
    
    pdf_files = []
    try:
        for root, _, files in os.walk(validated_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    try:
                        # Validate each PDF file
                        FileValidator.validate_pdf_file(full_path)
                        pdf_files.append(full_path)
                        logger.debug(f"Found valid PDF: {full_path}")
                    except FileValidationError as e:
                        logger.warning(f"Skipping invalid PDF {full_path}: {e.message}")
                        continue
    except OSError as e:
        logger.error(f"Error accessing directory {validated_path}: {str(e)}")
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
    """

    @validate_inputs(
        dir_path=lambda x: DirectoryValidator.validate_directory_path(x, must_exist=True, must_be_readable=True)
    )
    def __init__(self, dir_path: str):
        """
        Initializes the PDFHandler with the directory containing PDF files.

        Args:
            dir_path: Path to the directory with PDF files.
            
        Raises:
            DirectoryValidationError: If directory is invalid or inaccessible
        """
        with self.log_operation("pdf_handler_init", directory=str(dir_path)):
            self.dir_path = Path(dir_path)
            self.logger.info(f"Initializing PDFHandler with directory: {self.dir_path}")
            
            # Gather all PDF files in the directory and subdirectories
            self.pdf_files = find_pdf_files(str(self.dir_path))
            
            if not self.pdf_files:
                self.logger.warning(f"No PDF files found in directory: {self.dir_path}")
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

        Returns:
            A dictionary mapping PDF file names to their page counts.
            
        Raises:
            FileValidationError: If a PDF file cannot be accessed
        """
        with self.log_operation("get_page_counts"):
            page_counts = {}
            
            for pdf_path in self.pdf_files:
                try:
                    with fitz.open(pdf_path) as doc:
                        filename = os.path.basename(pdf_path)
                        count = doc.page_count
                        page_counts[filename] = count
                        self.logger.debug(f"PDF {filename} has {count} pages")
                        
                except Exception as e:
                    self.logger.error(f"Failed to get page count for {pdf_path}: {str(e)}")
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
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks to create
            
        Returns:
            Tuple of (filename, list_of_chunks)
        """
        filename = os.path.basename(pdf_path)
        
        try:
            # Validate PDF file before processing
            FileValidator.validate_pdf_file(pdf_path)
            
            self.logger.debug(f"Extracting chunks from {filename} with chunk size {chunk_size}")
            
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
            
            self.logger.info(f"Successfully extracted {len(file_chunks)} chunks from {filename}")
            return filename, file_chunks
            
        except FileValidationError:
            self.logger.error(f"File validation failed for {filename}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing {filename}: {str(e)}", 
                            extra={'file_path': pdf_path})
            # Return empty chunks instead of failing completely
            return filename, []

    @validate_inputs(
        chunk_size=lambda x: ParameterValidator.validate_positive_integer(x, "chunk_size", min_value=100, max_value=50000)
    )
    @handle_exceptions(default_return={})
    def extract_page_chunks(self, chunk_size: int = 5000) -> Dict[str, List[str]]:
        """
        Extracts text chunks from each PDF using DocumentChunking from phidata, in parallel.
        
        Args:
            chunk_size: Size of text chunks to create (100-50000 characters)
            
        Returns:
            Dictionary mapping PDF filenames to lists of text chunks
            
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
                        pdf_name, file_chunks = future.result()
                        chunks[pdf_name] = file_chunks
                        
                        if file_chunks:
                            successful_files += 1
                            self.logger.debug(f"Successfully processed {pdf_name}: {len(file_chunks)} chunks")
                        else:
                            failed_files += 1
                            self.logger.warning(f"No chunks extracted from {pdf_name}")
                            
                    except Exception as e:
                        failed_files += 1
                        self.logger.error(f"Failed to process PDF: {str(e)}")
            
            self.logger.info(f"Chunk extraction completed: {successful_files} successful, {failed_files} failed")
            return chunks
