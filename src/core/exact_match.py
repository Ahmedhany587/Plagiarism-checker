
import hashlib
import re
from typing import Dict, List, Tuple, Set


class ExactMatchDetector:
    """
    Detects exact matching text between PDFs by finding identical text blocks.
    Now works at sentence/paragraph level for better plagiarism detection.
    """

    def __init__(self, min_match_length: int = 50):
        """
        Initialize the exact match detector.
        
        :param min_match_length: Minimum character length for a match to be considered
        """
        self.min_match_length = min_match_length

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace and standardizing format.
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common formatting characters that might differ
        text = re.sub(r'[""''‚„«»]', '"', text)  # Normalize quotes
        text = re.sub(r'[–—]', '-', text)  # Normalize dashes
        return text.lower()

    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text for more granular matching.
        """
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.min_match_length:
                cleaned_sentences.append(self._normalize_text(sentence))
        
        return cleaned_sentences

    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from text for matching.
        """
        # Split on paragraph boundaries (double newlines or more)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) >= self.min_match_length:
                cleaned_paragraphs.append(self._normalize_text(paragraph))
        
        return cleaned_paragraphs

    def _hash_text(self, text: str) -> str:
        """
        Generate MD5 hash for normalized text.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def find_exact_matches(
        self, chunks: Dict[str, List[str]]
    ) -> List[Tuple[str, int, str, int, str]]:
        """
        Compare all PDFs to find exact duplicate text blocks at sentence/paragraph level.
        Returns a list of matches with metadata.

        :param chunks: {pdf_name: [page1_text, page2_text, ...]}
        :return: List of tuples:
            (pdfA, page_numA, pdfB, page_numB, matched_text)
        """
        # Hash map: hash -> [(pdf_name, page_num, original_text)]
        hash_map = {}
        matches = []

        # Extract text blocks from all documents
        for pdf_name, pages in chunks.items():
            for page_num, page_text in enumerate(pages):
                # Extract both sentences and paragraphs for comprehensive matching
                text_blocks = []
                
                # Add sentences
                sentences = self._extract_sentences(page_text)
                text_blocks.extend(sentences)
                
                # Add paragraphs (longer blocks)
                paragraphs = self._extract_paragraphs(page_text)
                text_blocks.extend(paragraphs)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_blocks = []
                for block in text_blocks:
                    if block not in seen and len(block.strip()) >= self.min_match_length:
                        seen.add(block)
                        unique_blocks.append(block)
                
                # Hash each unique text block
                for text_block in unique_blocks:
                    block_hash = self._hash_text(text_block)
                    
                    if block_hash in hash_map:
                        # Found a match!
                        for prev_pdf, prev_page, prev_text in hash_map[block_hash]:
                            # Skip self-comparison (same PDF)
                            if prev_pdf != pdf_name:
                                # Use the original text (before normalization) for display
                                matches.append((prev_pdf, prev_page, pdf_name, page_num, prev_text))
                    else:
                        hash_map[block_hash] = []
                    
                    # Store this occurrence
                    # Find the original text snippet that matches
                    original_snippet = self._find_original_text(page_text, text_block)
                    hash_map[block_hash].append((pdf_name, page_num, original_snippet))

        return matches

    def _find_original_text(self, original_text: str, normalized_block: str) -> str:
        """
        Find the original text snippet that corresponds to the normalized block.
        """
        # This is a simplified approach - in practice, you might want more sophisticated matching
        sentences = re.split(r'[.!?]+', original_text)
        
        for sentence in sentences:
            if len(sentence.strip()) >= self.min_match_length:
                if self._normalize_text(sentence) == normalized_block:
                    return sentence.strip() + "."
        
        # Fallback: try paragraphs
        paragraphs = re.split(r'\n\s*\n', original_text)
        for paragraph in paragraphs:
            if len(paragraph.strip()) >= self.min_match_length:
                if self._normalize_text(paragraph) == normalized_block:
                    return paragraph.strip()
        
        # Final fallback: return a truncated version of the normalized text
        return normalized_block[:200] + "..." if len(normalized_block) > 200 else normalized_block
