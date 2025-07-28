from typing import Dict, Tuple, List, Any
import torch
import torch.nn.functional as F
import itertools
from tqdm import tqdm

from .logging_config import LoggerMixin
from .validation import (
    ParameterValidator, ParameterValidationError,
    validate_inputs, handle_exceptions
)

class PDFSimilarityCalculator(LoggerMixin):
    """
    A class to compute semantic similarity between different PDFs
    using cosine similarity of page embeddings.
    """

    @validate_inputs(
        embeddings=lambda x: x if isinstance(x, dict) and len(x) >= 2 else None,
        batch_size=lambda x: ParameterValidator.validate_positive_integer(x, "batch_size", min_value=1, max_value=2048)
    )
    def __init__(self, embeddings: Dict[str, List[Any]], batch_size: int = 256):
        """
        Initialize PDFSimilarityCalculator.
        
        Args:
            embeddings: Dictionary mapping PDF file names to lists of embeddings
            batch_size: Batch size for similarity computation
            
        Raises:
            ParameterValidationError: If parameters are invalid
        """
        with self.log_operation("similarity_calculator_init", pdf_count=len(embeddings)):
            if len(embeddings) < 2:
                raise ParameterValidationError(
                    "At least 2 PDFs required for similarity calculation",
                    field="embeddings",
                    value=len(embeddings)
                )
            
            self.embeddings = embeddings
            self.batch_size = batch_size
            
            # Validate embeddings format
            for pdf_name, emb_list in embeddings.items():
                if not emb_list:
                    self.logger.warning(f"Empty embeddings for PDF: {pdf_name}")
                    continue
                
                # Convert to tensors if needed
                if isinstance(emb_list, list) and len(emb_list) > 0:
                    if not isinstance(emb_list[0], torch.Tensor):
                        self.logger.debug(f"Converting embeddings to tensors for {pdf_name}")
                        try:
                            self.embeddings[pdf_name] = [torch.as_tensor(emb) for emb in emb_list]
                        except Exception as e:
                            self.logger.error(f"Failed to convert embeddings for {pdf_name}: {str(e)}")
                            raise ParameterValidationError(
                                f"Invalid embeddings format for {pdf_name}: {str(e)}",
                                field="embeddings",
                                value=pdf_name
                            )
            
            self.logger.info(f"Initialized similarity calculator for {len(embeddings)} PDFs")

    @handle_exceptions(default_return=(0.0, 0.0, 0.0))
    def compute_pairwise_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> Tuple[float, float, float]:
        """
        Computes the max, min, and mean cosine similarity between all page embeddings of two PDFs,
        using batched matrix multiplication for memory efficiency.

        Args:
            emb1: Tensor of shape (N, D) for PDF A
            emb2: Tensor of shape (M, D) for PDF B
            
        Returns:
            Tuple of (max_similarity, min_similarity, mean_similarity)
            
        Raises:
            RuntimeError: If tensor operations fail
        """
        try:
            # Ensure emb1 and emb2 are tensors (not lists)
            if isinstance(emb1, list):
                emb1 = torch.stack([torch.as_tensor(e) for e in emb1])
            if isinstance(emb2, list):
                emb2 = torch.stack([torch.as_tensor(e) for e in emb2])
            
            # Validate tensor dimensions
            if emb1.dim() != 2 or emb2.dim() != 2:
                raise ValueError(f"Expected 2D tensors, got shapes {emb1.shape} and {emb2.shape}")
            
            if emb1.size(1) != emb2.size(1):
                raise ValueError(f"Embedding dimensions don't match: {emb1.size(1)} vs {emb2.size(1)}")
            
            self.logger.debug(f"Computing similarity between tensors of shapes {emb1.shape} and {emb2.shape}")
            
            # Normalize embeddings
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            similarities = []

            # Compute similarities in batches for memory efficiency
            for i in range(0, emb1.size(0), self.batch_size):
                batch = emb1[i:i + self.batch_size]  # (batch_size x D)
                sim_chunk = torch.mm(batch, emb2.T)  # (batch_size x M)
                similarities.append(sim_chunk)

            sim_matrix = torch.cat(similarities, dim=0)  # (N x M)
            
            # Rescale cosine similarity from [-1, 1] to [0, 1]
            sim_matrix = (sim_matrix + 1) / 2
            
            max_sim = sim_matrix.max().item()
            min_sim = sim_matrix.min().item()
            mean_sim = sim_matrix.mean().item()

            self.logger.debug(f"Computed similarities - max: {max_sim:.3f}, min: {min_sim:.3f}, mean: {mean_sim:.3f}")
            return max_sim, min_sim, mean_sim
            
        except Exception as e:
            self.logger.error(f"Failed to compute pairwise similarity: {str(e)}")
            raise RuntimeError(f"Similarity computation failed: {str(e)}")


    @handle_exceptions(default_return={})
    def compute_all_pdf_similarities(self) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """
        Computes many-to-many PDF semantic similarity (excluding self-comparison).

        Returns:
            Dictionary with key (pdfA, pdfB) and value as tuple of (max_sim, min_sim, mean_sim)
        """
        with self.log_operation("compute_all_similarities"):
            similarity_scores = {}
            pdf_pairs = list(itertools.combinations(self.embeddings.keys(), 2))
            
            if not pdf_pairs:
                self.logger.warning("No PDF pairs found for similarity computation")
                return {}
            
            self.logger.info(f"Computing similarities for {len(pdf_pairs)} PDF pairs")
            
            successful_pairs = 0
            failed_pairs = 0

            for pdfA, pdfB in tqdm(pdf_pairs, desc="Calculating Similarities", unit="pair"):
                try:
                    embA = self.embeddings.get(pdfA, [])
                    embB = self.embeddings.get(pdfB, [])
                    
                    # Skip if either PDF has no embeddings
                    if not embA or not embB:
                        self.logger.warning(f"Skipping pair ({pdfA}, {pdfB}): missing embeddings")
                        failed_pairs += 1
                        continue
                    
                    # Convert lists to tensors if needed
                    if isinstance(embA, list):
                        embA = torch.stack([torch.as_tensor(e) for e in embA])
                    if isinstance(embB, list):
                        embB = torch.stack([torch.as_tensor(e) for e in embB])
                    
                    max_sim, min_sim, mean_sim = self.compute_pairwise_similarity(embA, embB)
                    similarity_scores[(pdfA, pdfB)] = (max_sim, min_sim, mean_sim)
                    successful_pairs += 1
                    
                    self.logger.debug(f"Computed similarity for ({pdfA}, {pdfB}): mean={mean_sim:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to compute similarity for pair ({pdfA}, {pdfB}): {str(e)}")
                    failed_pairs += 1
                    continue
            
            self.logger.info(f"Similarity computation completed: {successful_pairs} successful, {failed_pairs} failed")
            return similarity_scores
