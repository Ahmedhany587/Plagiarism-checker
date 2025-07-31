import torch
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from tqdm import tqdm
import multiprocessing
import hashlib
import pickle
import time
import psutil
from typing import Dict, List, Optional, Any
from pathlib import Path

from .logging_config import LoggerMixin
from .validation import (
    ParameterValidator, FileValidationError, ParameterValidationError,
    validate_inputs, handle_exceptions
)

CACHE_DIR = 'embedding_cache'

# Global model for worker processes
_worker_model = None

def _init_worker(model_name: str, device: str, hf_token: Optional[str]) -> None:
    """Initialize worker process with SentenceTransformer model."""
    global _worker_model
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure model cache directory is set for worker processes
        model_cache_dir = os.path.join(os.getcwd(), "model_cache", "sentence_transformers")
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)
        
        logger.info(f"Initializing worker with model {model_name} on device {device}")
        _worker_model = SentenceTransformer(model_name, device=device, token=hf_token)
        logger.info("Worker initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize worker: {str(e)}")
        raise

def _hash_pdf(pdf_name: str, pages: List[str]) -> str:
    """
    Generate hash for PDF content based on filename and page content.
    
    Args:
        pdf_name: Name of the PDF file
        pages: List of page content strings
        
    Returns:
        SHA256 hash string
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        m = hashlib.sha256()
        m.update(pdf_name.encode('utf-8'))
        
        # Hash page content with error handling
        for i, page in enumerate(pages):
            try:
                if isinstance(page, str):
                    m.update(page.encode('utf-8'))
                else:
                    logger.warning(f"Non-string page content at index {i} in {pdf_name}")
                    m.update(str(page).encode('utf-8'))
            except UnicodeEncodeError as e:
                logger.warning(f"Unicode encoding error for page {i} in {pdf_name}: {str(e)}")
                m.update(page.encode('utf-8', errors='ignore'))
        
        hash_value = m.hexdigest()
        logger.debug(f"Generated hash for {pdf_name}: {hash_value[:16]}...")
        return hash_value
        
    except Exception as e:
        logger.error(f"Failed to hash PDF {pdf_name}: {str(e)}")
        # Fallback to simple hash
        return hashlib.sha256(f"{pdf_name}_{len(pages)}".encode('utf-8')).hexdigest()

def _embed_pdf_worker(args: tuple) -> tuple:
    """
    Worker function to generate embeddings for a single PDF.
    
    Args:
        args: Tuple of (pdf_name, pages, batch_size)
        
    Returns:
        Tuple of (pdf_name, pdf_embeddings)
    """
    pdf_name, pages, batch_size = args
    global _worker_model
    
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        if _worker_model is None:
            raise RuntimeError("Worker model not initialized")
        
        logger.info(f"Starting embedding generation for {pdf_name} ({len(pages)} pages)")
        
        pdf_embeddings = []
        for i in tqdm(range(0, len(pages), batch_size), desc=f"Embedding {pdf_name}", leave=False):
            batch = pages[i:i+batch_size]
            
            # Validate batch content
            valid_batch = []
            for j, page in enumerate(batch):
                if isinstance(page, str) and page.strip():
                    valid_batch.append(page.strip())
                else:
                    logger.warning(f"Skipping invalid page content in {pdf_name} at batch {i}, page {j}")
            
            if valid_batch:
                try:
                    batch_emb = _worker_model.encode(
                        valid_batch, 
                        convert_to_tensor=True, 
                        batch_size=min(batch_size, len(valid_batch))
                    )
                    pdf_embeddings.extend(batch_emb)
                except Exception as e:
                    logger.error(f"Failed to embed batch {i} for {pdf_name}: {str(e)}")
                    continue
        
        logger.info(f"Successfully generated {len(pdf_embeddings)} embeddings for {pdf_name}")
        return pdf_name, pdf_embeddings
        
    except Exception as e:
        logger.error(f"Worker failed for {pdf_name}: {str(e)}")
        return pdf_name, []

class EmbeddingGenerator(LoggerMixin):
    """
    A class to load a SentenceTransformer model and generate embeddings
    for text chunks (1 chunk = 1 page), with caching and efficient multiprocessing.
    """

    @validate_inputs(
        model_name=lambda x: ParameterValidator.validate_string(x, "model_name", min_length=1, max_length=200),
        batch_size=lambda x: ParameterValidator.validate_positive_integer(x, "batch_size", min_value=1, max_value=2000) if x is not None else None
    )
    def __init__(self, 
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
                 batch_size: Optional[int] = None):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            batch_size: Batch size for processing (auto-detected if None)
            
        Raises:
            ParameterValidationError: If parameters are invalid
            RuntimeError: If model loading fails
        """
        with self.log_operation("embedding_generator_init", model_name=model_name):
            # Load environment variables
            load_dotenv()
            hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
                self.logger.info("HuggingFace token loaded from environment")
            else:
                self.logger.warning("No HuggingFace token found in environment")
            
            # Setup custom model cache directory for faster loading
            model_cache_dir = os.path.join(os.getcwd(), "model_cache", "sentence_transformers")
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
            os.makedirs(model_cache_dir, exist_ok=True)
            self.logger.info(f"Sentence Transformers cache directory: {model_cache_dir}")
            
            # Setup device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {self.device}")
            
            self.model_name = model_name
            self.hf_token = hf_token
            
            # Setup batch size
            if batch_size is not None:
                self.batch_size = batch_size
            else:
                # Dynamically set batch size based on available memory
                try:
                    mem_gb = psutil.virtual_memory().available / 1e9
                    self.batch_size = min(1000, max(8, int(mem_gb * 32)))
                    self.logger.info(f"Auto-detected batch size: {self.batch_size} (based on {mem_gb:.1f}GB available memory)")
                except Exception as e:
                    self.logger.warning(f"Failed to auto-detect batch size: {str(e)}. Using default: 32")
                    self.batch_size = 32
            
            # Setup cache directory
            self.cache_dir = Path(CACHE_DIR)
            try:
                self.cache_dir.mkdir(exist_ok=True)
                self.logger.info(f"Cache directory ready: {self.cache_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create cache directory: {str(e)}")
                raise RuntimeError(f"Cannot create cache directory: {str(e)}")
            
            # Load model for main process
            try:
                self.logger.info(f"Loading model: {model_name}")
                self.model = SentenceTransformer(model_name, device=self.device, token=hf_token)
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise RuntimeError(f"Cannot load model: {str(e)}")

    def _cache_path(self, pdf_name: str, pages: List[str]) -> Path:
        """Generate cache file path for a PDF."""
        hash_val = _hash_pdf(pdf_name, pages)
        return self.cache_dir / f"{hash_val}.pkl"

    @handle_exceptions(default_return=None)
    def _load_from_cache(self, pdf_name: str, pages: List[str]) -> Optional[Any]:
        """
        Load embeddings from cache if available.
        
        Args:
            pdf_name: Name of the PDF file
            pages: List of page content
            
        Returns:
            Cached embeddings or None if not found
        """
        try:
            path = self._cache_path(pdf_name, pages)
            if path.exists():
                with open(path, 'rb') as f:
                    embeddings = pickle.load(f)
                self.logger.debug(f"Loaded embeddings from cache for {pdf_name}")
                return embeddings
            else:
                self.logger.debug(f"No cache found for {pdf_name}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to load cache for {pdf_name}: {str(e)}")
            return None

    @handle_exceptions(default_return=False)
    def _save_to_cache(self, pdf_name: str, pages: List[str], embeddings: Any) -> bool:
        """
        Save embeddings to cache.
        
        Args:
            pdf_name: Name of the PDF file
            pages: List of page content
            embeddings: Embeddings to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = self._cache_path(pdf_name, pages)
            path.parent.mkdir(exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            self.logger.debug(f"Saved embeddings to cache for {pdf_name}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {pdf_name}: {str(e)}")
            return False

    @handle_exceptions(default_return=[])
    def _embed_pdf_single(self, pdf_name: str, pages: List[str]) -> List[Any]:
        """
        Generate embeddings for a single PDF using the main process model.
        
        Args:
            pdf_name: Name of the PDF file
            pages: List of page content strings
            
        Returns:
            List of embeddings
        """
        with self.log_operation("embed_pdf_single", pdf_name=pdf_name, page_count=len(pages)):
            pdf_embeddings = []
            
            for i in tqdm(range(0, len(pages), self.batch_size), desc=f"Embedding {pdf_name}", leave=False):
                batch = pages[i:i+self.batch_size]
                
                # Validate and clean batch
                valid_batch = []
                for j, page in enumerate(batch):
                    if isinstance(page, str) and page.strip():
                        valid_batch.append(page.strip())
                    else:
                        self.logger.warning(f"Skipping invalid page content in {pdf_name} at batch {i}, page {j}")
                
                if valid_batch:
                    try:
                        batch_emb = self.model.encode(
                            valid_batch, 
                            convert_to_tensor=True, 
                            batch_size=min(self.batch_size, len(valid_batch))
                        )
                        pdf_embeddings.extend(batch_emb)
                    except Exception as e:
                        self.logger.error(f"Failed to embed batch {i} for {pdf_name}: {str(e)}")
                        continue
            
            self.logger.info(f"Generated {len(pdf_embeddings)} embeddings for {pdf_name}")
            return pdf_embeddings

    @validate_inputs(
        chunks=lambda x: x if isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, list) for k, v in x.items()) else None
    )
    @handle_exceptions(default_return={})
    def generate_embeddings(self, chunks: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        """
        Generates embeddings for each text chunk (each page) from each PDF, with caching and progress tracking.
        
        Args:
            chunks: Dictionary mapping PDF names to lists of page content
            
        Returns:
            Dictionary mapping PDF names to lists of embeddings
            
        Raises:
            ParameterValidationError: If chunks parameter is invalid
        """
        with self.log_operation("generate_embeddings", total_pdfs=len(chunks)):
            if not chunks:
                self.logger.warning("No chunks provided for embedding generation")
                return {}
            
            embeddings = {}
            to_process = []
            cache_hits = 0
            
            # Check cache first
            self.logger.info("Checking cache for existing embeddings...")
            for pdf_name, pages in chunks.items():
                try:
                    cached = self._load_from_cache(pdf_name, pages)
                    if cached is not None:
                        embeddings[pdf_name] = cached
                        cache_hits += 1
                        self.logger.debug(f"Cache hit for {pdf_name}")
                    else:
                        to_process.append((pdf_name, pages, self.batch_size))
                        self.logger.debug(f"Cache miss for {pdf_name}")
                except Exception as e:
                    self.logger.warning(f"Cache check failed for {pdf_name}: {str(e)}")
                    to_process.append((pdf_name, pages, self.batch_size))
            
            self.logger.info(f"Cache summary: {cache_hits} hits, {len(to_process)} files to process")
            
            if not to_process:
                self.logger.info("All embeddings loaded from cache")
                return embeddings
            
            # Determine processing strategy
            num_pdfs = len(to_process)
            num_cpus = os.cpu_count() or 1
            num_processes = min(num_cpus, num_pdfs, 4)  # Limit to 4 processes max
            
            self.logger.info(f"Processing {num_pdfs} PDFs using {num_processes} processes")
            
            if num_processes > 1:
                # Multi-process approach
                try:
                    with multiprocessing.Pool(
                        processes=num_processes,
                        initializer=_init_worker,
                        initargs=(self.model_name, self.device, self.hf_token)
                    ) as pool:
                        successful_embeddings = 0
                        failed_embeddings = 0
                        
                        for pdf_name, pdf_embeddings in tqdm(
                            pool.imap_unordered(_embed_pdf_worker, to_process), 
                            total=num_pdfs, 
                            desc="Generating Embeddings", 
                            unit="pdf"
                        ):
                            if pdf_embeddings:
                                embeddings[pdf_name] = pdf_embeddings
                                successful_embeddings += 1
                                
                                # Save to cache
                                try:
                                    original_pages = chunks[pdf_name]
                                    self._save_to_cache(pdf_name, original_pages, pdf_embeddings)
                                except Exception as e:
                                    self.logger.warning(f"Failed to cache embeddings for {pdf_name}: {str(e)}")
                            else:
                                failed_embeddings += 1
                                self.logger.error(f"Failed to generate embeddings for {pdf_name}")
                        
                        self.logger.info(f"Multi-process embedding completed: {successful_embeddings} successful, {failed_embeddings} failed")
                        
                except Exception as e:
                    self.logger.error(f"Multi-process embedding failed: {str(e)}. Falling back to single process.")
                    # Fallback to single process
                    for pdf_name, pages, _ in tqdm(to_process, desc="Generating Embeddings (fallback)", unit="pdf"):
                        try:
                            pdf_embeddings = self._embed_pdf_single(pdf_name, pages)
                            if pdf_embeddings:
                                embeddings[pdf_name] = pdf_embeddings
                                self._save_to_cache(pdf_name, pages, pdf_embeddings)
                        except Exception as pdf_error:
                            self.logger.error(f"Failed to generate embeddings for {pdf_name}: {str(pdf_error)}")
            else:
                # Single process approach
                self.logger.info("Using single-process embedding generation")
                successful_embeddings = 0
                failed_embeddings = 0
                
                for pdf_name, pages, _ in tqdm(to_process, desc="Generating Embeddings", unit="pdf"):
                    try:
                        pdf_embeddings = self._embed_pdf_single(pdf_name, pages)
                        if pdf_embeddings:
                            embeddings[pdf_name] = pdf_embeddings
                            successful_embeddings += 1
                            self._save_to_cache(pdf_name, pages, pdf_embeddings)
                        else:
                            failed_embeddings += 1
                    except Exception as e:
                        failed_embeddings += 1
                        self.logger.error(f"Failed to generate embeddings for {pdf_name}: {str(e)}")
                
                self.logger.info(f"Single-process embedding completed: {successful_embeddings} successful, {failed_embeddings} failed")
            
            total_embeddings = sum(len(emb) for emb in embeddings.values())
            self.logger.info(f"Embedding generation complete: {len(embeddings)} PDFs, {total_embeddings} total embeddings")
            
            return embeddings
