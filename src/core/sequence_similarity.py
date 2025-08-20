from typing import Dict, Tuple, List, Any, Optional
import itertools
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging_config import LoggerMixin
from .validation import (
    ParameterValidator, ParameterValidationError,
    validate_inputs, handle_exceptions
)

class SequenceSimilarityCalculator(LoggerMixin):
    """
    A class to compute sequence similarity between different PDFs
    using fuzzy string matching and text pattern analysis.
    """

    @validate_inputs(
        chunks=lambda x: x if isinstance(x, dict) and len(x) >= 2 else None
    )
    def __init__(self, chunks: Dict[str, List[str]]):
        """
        Initialize SequenceSimilarityCalculator.
        
        Args:
            chunks: Dictionary mapping PDF file names to lists of text chunks
            
        Raises:
            ParameterValidationError: If parameters are invalid
        """
        with self.log_operation("sequence_similarity_calculator_init", pdf_count=len(chunks)):
            if len(chunks) < 2:
                raise ParameterValidationError(
                    "At least 2 PDFs required for similarity calculation",
                    field="chunks",
                    value=len(chunks)
                )
            
            self.chunks = chunks
            self.logger.info(f"Initialized sequence similarity calculator for {len(chunks)} PDFs")

    @handle_exceptions(default_return=(0.0, 0.0, 0.0))
    def compute_pairwise_similarity(self, chunks1: List[str], chunks2: List[str]) -> Tuple[float, float, float]:
        """
        Computes the max, min, and mean sequence similarity between all chunks of two PDFs.

        Args:
            chunks1: List of text chunks for PDF A
            chunks2: List of text chunks for PDF B
            
        Returns:
            Tuple of (max_similarity, min_similarity, mean_similarity)
            
        Raises:
            RuntimeError: If similarity computation fails
        """
        try:
            from rapidfuzz import fuzz
            
            similarities = []
            
            # Compare each chunk from PDF1 with each chunk from PDF2
            for chunk1 in chunks1:
                chunk_similarities = []
                for chunk2 in chunks2:
                    # Use token sort ratio for better sequence matching
                    similarity = fuzz.token_sort_ratio(chunk1, chunk2) / 100.0
                    chunk_similarities.append(similarity)
                
                if chunk_similarities:
                    similarities.extend(chunk_similarities)
            
            if not similarities:
                return (0.0, 0.0, 0.0)
            
            max_sim = max(similarities)
            min_sim = min(similarities)
            mean_sim = sum(similarities) / len(similarities)

            self.logger.debug(f"Computed sequence similarities - max: {max_sim:.3f}, min: {min_sim:.3f}, mean: {mean_sim:.3f}")
            return max_sim, min_sim, mean_sim
            
        except Exception as e:
            self.logger.error(f"Failed to compute sequence similarity: {str(e)}")
            raise RuntimeError(f"Sequence similarity computation failed: {str(e)}")

    @handle_exceptions(default_return={})
    def compute_all_pdf_similarities_parallel(self, max_workers: int = 4) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """
        Enhanced parallel computation of many-to-many PDF sequence similarity with improved performance.

        Args:
            max_workers: Maximum number of parallel workers for similarity computation
            
        Returns:
            Dictionary with key (pdfA, pdfB) and value as tuple of (max_sim, min_sim, mean_sim)
        """
        with self.log_operation("compute_all_sequence_similarities_parallel", max_workers=max_workers):
            similarity_scores = {}
            pdf_pairs = list(itertools.combinations(self.chunks.keys(), 2))
            
            if not pdf_pairs:
                self.logger.warning("No PDF pairs found for sequence similarity computation")
                return {}
            
            # Validate and adjust max_workers
            max_workers = min(max_workers, len(pdf_pairs), os.cpu_count() or 1)
            self.logger.info(f"Computing sequence similarities for {len(pdf_pairs)} PDF pairs using {max_workers} workers")
            
            successful_pairs = 0
            failed_pairs = 0
            
            # Use ThreadPoolExecutor for parallel similarity computation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all similarity computation tasks
                future_to_pair = {
                    executor.submit(self._compute_sequence_pair_similarity_worker, pdf_pair): pdf_pair 
                    for pdf_pair in pdf_pairs
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_pair):
                    pdf_pair = future_to_pair[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            similarity_scores[pdf_pair] = result
                            successful_pairs += 1
                            self.logger.debug(f"Computed sequence similarity for {pdf_pair}: mean={result[2]:.3f}")
                        else:
                            failed_pairs += 1
                            self.logger.warning(f"Failed to compute sequence similarity for pair {pdf_pair}")
                            
                    except Exception as e:
                        failed_pairs += 1
                        self.logger.error(f"Exception computing sequence similarity for pair {pdf_pair}: {str(e)}")
                        continue
            
            self.logger.info(f"Parallel sequence similarity computation completed: {successful_pairs} successful, {failed_pairs} failed")
            return similarity_scores

    def _compute_sequence_pair_similarity_worker(self, pdf_pair: Tuple[str, str]) -> Optional[Tuple[float, float, float]]:
        """
        Worker function for computing sequence similarity between a single PDF pair.
        
        Args:
            pdf_pair: Tuple of (pdfA, pdfB) to compare
            
        Returns:
            Tuple of (max_sim, min_sim, mean_sim) or None if computation fails
        """
        try:
            pdfA, pdfB = pdf_pair
            chunks1 = self.chunks.get(pdfA, [])
            chunks2 = self.chunks.get(pdfB, [])
            
            # Skip if either PDF has no chunks
            if not chunks1 or not chunks2:
                self.logger.warning(f"Skipping sequence pair ({pdfA}, {pdfB}): missing chunks")
                return None
            
            max_sim, min_sim, mean_sim = self.compute_pairwise_similarity(chunks1, chunks2)
            return (max_sim, min_sim, mean_sim)
            
        except Exception as e:
            self.logger.error(f"Failed to compute sequence similarity for pair {pdf_pair}: {str(e)}")
            return None

    @handle_exceptions(default_return={})
    def compute_all_pdf_similarities(self, chunks: Dict[str, List[str]]) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """
        Compute sequence similarity between all unique PDF pairs in parallel.

        :param chunks: Dictionary {pdf_name: list of page texts}
        :return: Dictionary {(pdfA, pdfB): (max_similarity, min_similarity, mean_similarity)}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        cleaned_chunks = self._preprocess_chunks(chunks)
        # If fewer than 2 PDFs have text, return empty result gracefully
        if len(cleaned_chunks) < 2:
            return {}
        scores = {}
        pairs = list(itertools.combinations(cleaned_chunks.keys(), 2))
        args = [(pdfA, pdfB, cleaned_chunks) for pdfA, pdfB in pairs]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._pair_similarity, arg) for arg in args]
            for future in as_completed(futures):
                pair, result = future.result()
                scores[pair] = result
        return scores
