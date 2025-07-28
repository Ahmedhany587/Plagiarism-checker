import hashlib
from typing import Dict, List, Tuple


class ExactMatchDetector:
    """
    Detects exact matching text between PDFs by hashing page content.
    """

    def __init__(self):
        pass

    def _hash_text(self, text: str) -> str:
        """
        Generate MD5 hash for a given page's text (whitespace stripped).
        """
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def find_exact_matches(
        self, chunks: Dict[str, List[str]]
    ) -> List[Tuple[str, int, str, int, str]]:
        """
        Compare all PDFs to find exact duplicate pages (by hash).
        Returns a list of matches with metadata.

        :param chunks: {pdf_name: [page1_text, page2_text, ...]}
        :return: List of tuples:
            (pdfA, page_numA, pdfB, page_numB, matched_text)
        """
        hash_map = {}  # hash -> (pdf_name, page_num, text)
        matches = []

        for pdf_name, pages in chunks.items():
            for i, text in enumerate(pages):
                page_hash = self._hash_text(text)

                if page_hash in hash_map:
                    prev_pdf, prev_page, prev_text = hash_map[page_hash]
                    # Skip self-comparison (same PDF)
                    if prev_pdf != pdf_name:
                        matches.append((prev_pdf, prev_page, pdf_name, i, text))
                else:
                    hash_map[page_hash] = (pdf_name, i, text)

        return matches
