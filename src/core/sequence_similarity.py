from typing import Dict, Tuple
from rapidfuzz import process, fuzz
import numpy as np
import itertools
from tqdm import tqdm
from rapidfuzz.utils import default_process


class SequenceSimilarityCalculator:
    """
    Computes sequence-based similarity between PDFs using RapidFuzz.
    Each page is treated as a chunk. Computes full pairwise similarity between PDFs.
    """

    def __init__(self, scorer=fuzz.QRatio):
        """
        :param scorer: RapidFuzz similarity scorer (default: token_sort_ratio)
        """
        self.scorer = scorer

    def _compare_two_pdfs(self, pages_a: list, pages_b: list) -> Tuple[float, float, float]:
        """
        Compute max, min, and mean page-to-page sequence similarity between two PDFs.

        :param pages_a: List of page texts from PDF A
        :param pages_b: List of page texts from PDF B
        :return: (max_similarity, min_similarity, mean_similarity) scores (0 to 100)
        """
        n, m = len(pages_a), len(pages_b)
        matrix = np.zeros((n, m), dtype=float)

        for i, text_a in enumerate(pages_a):
            # Efficient batch comparison: text_a vs all pages_b
            row_scores = process.cdist([text_a], pages_b, scorer=self.scorer)[0]
            matrix[i, :] = row_scores

        max_score = matrix.max()
        min_score = matrix.min()
        mean_score = matrix.mean()

        return max_score, min_score, mean_score

    def _preprocess_chunks(self, chunks: Dict[str, list]) -> Dict[str, list]:
        """
        Apply default_process to every page in every PDF and drop PDFs with no pages.
        """
        cleaned = {}
        for pdf, pages in chunks.items():
            if isinstance(pages, list) and len(pages) > 0:
                processed = [default_process(page) for page in pages if isinstance(page, str) and page.strip()]
                if processed:
                    cleaned[pdf] = processed
        return cleaned

    def _pair_similarity(self, args):
        pdfA, pdfB, cleaned_chunks = args
        pages_a = cleaned_chunks[pdfA]
        pages_b = cleaned_chunks[pdfB]
        return (pdfA, pdfB), self._compare_two_pdfs(pages_a, pages_b)

    def compute_all_pdf_similarities(self, chunks: Dict[str, list]) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
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
