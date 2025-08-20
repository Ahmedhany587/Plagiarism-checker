import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import imagehash


class ImageDuplicationDetector:
    """
    Detects duplicate images across PDFs using perceptual hashing.
    Fast, reliable, and efficient duplicate detection with parallel processing support.
    """

    def __init__(self):
        """Initialize the image duplication detector."""
        # Storage for metadata
        self.filepaths: List[str] = []
        self.pdf_names: List[str] = []
        self.pages: List[int] = []
        self.image_indices: List[int] = []
        self._images: List[Image.Image] = []
        self.logger = logging.getLogger(__name__)

    def load_images_parallel(self, results: List[Dict[str, str]], max_workers: int = 3):
        """
        Load images and their PDF origin metadata using parallel processing.
        
        Args:
            results: list of {"filepath": str, "pdf": str, "page": int, "image_index": int}
            max_workers: Maximum number of parallel workers for image loading
        """
        self.filepaths = []
        self.pdf_names = []
        self.pages = []
        self.image_indices = []
        images = []
        
        self.logger.info(f"Loading {len(results)} images for analysis using {max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel image loading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all image loading tasks
            future_to_result = {
                executor.submit(self._load_single_image, r): r 
                for r in results
            }
            
            successful_loads = 0
            failed_loads = 0
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                
                try:
                    loaded_data = future.result()
                    if loaded_data is not None:
                        img, fp, pdf, page, img_idx = loaded_data
                        images.append(img)
                        self.filepaths.append(fp)
                        self.pdf_names.append(pdf)
                        self.pages.append(page)
                        self.image_indices.append(img_idx)
                        successful_loads += 1
                        
                        if successful_loads % 50 == 0:  # Progress update every 50 images
                            self.logger.info(f"Loaded {successful_loads}/{len(results)} images...")
                    else:
                        failed_loads += 1
                        
                except Exception as e:
                    failed_loads += 1
                    self.logger.error(f"Failed to load image from {result.get('filepath', 'unknown')}: {str(e)}")
        
        self._images = images
        self.logger.info(f"Successfully loaded {len(self._images)} images for analysis ({successful_loads} successful, {failed_loads} failed)")

    def _load_single_image(self, result: Dict[str, str]) -> Optional[tuple]:
        """
        Load a single image with error handling.
        
        Args:
            result: Dictionary containing image metadata
            
        Returns:
            Tuple of (image, filepath, pdf_name, page, image_index) or None if loading fails
        """
        try:
            fp = result["filepath"]
            if not os.path.isfile(fp):
                self.logger.warning(f"Skipping missing file: {fp}")
                return None
                
            img = Image.open(fp).convert("RGB")
            return (
                img,
                fp,
                result["pdf"],
                result.get("page", 0),
                result.get("image_index", 0)
            )
                
        except Exception as e:
            self.logger.error(f"Failed to load image {result.get('filepath', 'unknown')}: {str(e)}")
            return None

    def load_images(self, results: List[Dict[str, str]]):
        """
        Load images and their PDF origin metadata (legacy method).
        results: list of {"filepath": str, "pdf": str, "page": int, "image_index": int}
        """
        # Use parallel loading by default
        self.load_images_parallel(results, max_workers=3)

    def detect_cross_pdf_duplicates_parallel(
        self,
        threshold: float = 0.85,
        max_workers: int = 3
    ) -> Optional[List[Dict[str, object]]]:
        """
        Enhanced parallel duplicate detection using perceptual hashing.
        
        Args:
            threshold: Similarity threshold (0.0-1.0). Higher values = more strict matching.
                      0.85 is good for near-identical images.
                      0.75 is good for very similar images.
                      0.65 is good for somewhat similar images.
            max_workers: Maximum number of parallel workers for hash computation
        
        Returns:
            List of duplicate pairs with similarity scores, or None if no duplicates found.
        """
        if not hasattr(self, '_images') or not self._images:
            raise RuntimeError("No images loaded. Call load_images() first.")
        
        self.logger.info(f"Computing perceptual hashes for {len(self._images)} images using {max_workers} workers...")
        
        # Compute perceptual hashes in parallel
        hashes = self._compute_hashes_parallel(max_workers)
        
        self.logger.info(f"Finding duplicate pairs with threshold {threshold}...")
        
        # Find similar hashes using parallel processing
        pairs = self._find_duplicate_pairs_parallel(hashes, threshold, max_workers)
        
        if pairs:
            # Sort by similarity score (highest first)
            pairs.sort(key=lambda x: x['similarity'], reverse=True)
            self.logger.info(f"Analysis complete! Found {len(pairs)} duplicate pairs across different PDFs")
            self.logger.info(f"Similarity scores range: {pairs[-1]['similarity']:.2%} to {pairs[0]['similarity']:.2%}")
        else:
            self.logger.info("Analysis complete! No duplicate pairs found across different PDFs")
        
        return pairs if pairs else None

    def _compute_hashes_parallel(self, max_workers: int) -> List:
        """
        Compute perceptual hashes for all images in parallel.
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of perceptual hashes
        """
        hashes = [None] * len(self._images)  # Pre-allocate list
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all hash computation tasks
            future_to_index = {
                executor.submit(self._compute_single_hash, i, img): i 
                for i, img in enumerate(self._images)
            }
            
            successful_hashes = 0
            failed_hashes = 0
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    hash_result = future.result()
                    if hash_result is not None:
                        hashes[index] = hash_result
                        successful_hashes += 1
                        
                        if successful_hashes % 50 == 0:  # Progress update
                            self.logger.info(f"Computed hashes for {successful_hashes}/{len(self._images)} images...")
                    else:
                        failed_hashes += 1
                        
                except Exception as e:
                    failed_hashes += 1
                    self.logger.error(f"Failed to hash image {index}: {str(e)}")
        
        self.logger.info(f"Hash computation completed: {successful_hashes} successful, {failed_hashes} failed")
        return hashes

    def _compute_single_hash(self, index: int, img: Image.Image):
        """
        Compute perceptual hash for a single image.
        
        Args:
            index: Image index
            img: PIL Image object
            
        Returns:
            Perceptual hash or None if computation fails
        """
        try:
            # Use perceptual hash (good for finding similar images)
            phash = imagehash.phash(img, hash_size=8)
            return phash
        except Exception as e:
            self.logger.error(f"Failed to hash image {index}: {str(e)}")
            return None

    def _find_duplicate_pairs_parallel(self, hashes: List, threshold: float, max_workers: int) -> List[Dict[str, object]]:
        """
        Find duplicate pairs using parallel processing.
        
        Args:
            hashes: List of perceptual hashes
            threshold: Similarity threshold
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of duplicate pairs
        """
        # Generate all pairs to compare
        pairs_to_check = []
        for i in range(len(hashes)):
            if hashes[i] is None:
                continue
            for j in range(i + 1, len(hashes)):
                if hashes[j] is None:
                    continue
                # Skip same PDF
                if self.pdf_names[i] == self.pdf_names[j]:
                    continue
                pairs_to_check.append((i, j))
        
        if not pairs_to_check:
            return []
        
        self.logger.info(f"Checking {len(pairs_to_check)} image pairs for duplicates...")
        
        # Use ThreadPoolExecutor for parallel pair checking
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pair checking tasks
            future_to_pair = {
                executor.submit(self._check_single_pair, i, j, hashes, threshold): (i, j) 
                for i, j in pairs_to_check
            }
            
            pairs = []
            successful_checks = 0
            failed_checks = 0
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_pair):
                pair_indices = future_to_pair[future]
                
                try:
                    pair_result = future.result()
                    if pair_result is not None:
                        pairs.append(pair_result)
                        successful_checks += 1
                    else:
                        failed_checks += 1
                        
                except Exception as e:
                    failed_checks += 1
                    self.logger.error(f"Failed to check pair {pair_indices}: {str(e)}")
        
        self.logger.info(f"Pair checking completed: {successful_checks} successful, {failed_checks} failed")
        return pairs

    def _check_single_pair(self, i: int, j: int, hashes: List, threshold: float) -> Optional[Dict[str, object]]:
        """
        Check a single image pair for similarity.
        
        Args:
            i: First image index
            j: Second image index
            hashes: List of perceptual hashes
            threshold: Similarity threshold
            
        Returns:
            Dictionary with pair information if similar, None otherwise
        """
        try:
            # Compute hash similarity (smaller distance = more similar)
            hash_diff = hashes[i] - hashes[j]
            similarity = 1.0 - (hash_diff / 64.0)  # Normalize to 0-1 scale
            
            if similarity >= threshold:
                return {
                    "orig_path": self.filepaths[i],
                    "dup_path": self.filepaths[j],
                    "orig_pdf": self.pdf_names[i],
                    "dup_pdf": self.pdf_names[j],
                    "orig_page": self.pages[i],
                    "dup_page": self.pages[j],
                    "orig_image_index": self.image_indices[i],
                    "dup_image_index": self.image_indices[j],
                    "similarity": float(similarity)
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to check pair ({i}, {j}): {str(e)}")
            return None

    def detect_cross_pdf_duplicates(
        self,
        threshold: float = 0.85
    ) -> Optional[List[Dict[str, object]]]:
        """
        Finds duplicate image pairs across different PDFs using perceptual hashing.
        
        Args:
            threshold: Similarity threshold (0.0-1.0). Higher values = more strict matching.
                      0.85 is good for near-identical images.
                      0.75 is good for very similar images.
                      0.65 is good for somewhat similar images.
        
        Returns:
            List of duplicate pairs with similarity scores, or None if no duplicates found.
        """
        if not hasattr(self, '_images') or not self._images:
            raise RuntimeError("No images loaded. Call load_images() first.")
        
        print(f"Computing perceptual hashes for {len(self._images)} images...")
        
        # Compute perceptual hashes
        hashes = []
        for i, img in enumerate(self._images):
            try:
                # Use perceptual hash (good for finding similar images)
                phash = imagehash.phash(img, hash_size=8)
                hashes.append(phash)
                
                if (i + 1) % 50 == 0:  # Progress update
                    print(f"Computed hashes for {i + 1}/{len(self._images)} images...")
                    
            except Exception as e:
                print(f"Failed to hash image {i}: {str(e)}")
                hashes.append(None)
        
        print(f"Finding duplicate pairs with threshold {threshold}...")
        
        # Find similar hashes
        pairs = []
        seen = set()
        comparisons = 0
        total_comparisons = (len(hashes) * (len(hashes) - 1)) // 2
        
        for i in range(len(hashes)):
            if hashes[i] is None:
                continue
                
            for j in range(i + 1, len(hashes)):
                comparisons += 1
                
                if hashes[j] is None:
                    continue
                
                # Skip same PDF
                if self.pdf_names[i] == self.pdf_names[j]:
                    continue
                
                # Compute hash similarity (smaller distance = more similar)
                hash_diff = hashes[i] - hashes[j]
                similarity = 1.0 - (hash_diff / 64.0)  # Normalize to 0-1 scale
                
                if similarity >= threshold:
                    key = (min(i, j), max(i, j))  # Ensure consistent ordering
                    if key not in seen:
                        seen.add(key)
                        pairs.append({
                            "orig_path": self.filepaths[i],
                            "dup_path": self.filepaths[j],
                            "orig_pdf": self.pdf_names[i],
                            "dup_pdf": self.pdf_names[j],
                            "orig_page": self.pages[i],
                            "dup_page": self.pages[j],
                            "orig_image_index": self.image_indices[i],
                            "dup_image_index": self.image_indices[j],
                            "similarity": float(similarity)
                        })
                
                # Progress update for large datasets
                if comparisons % 10000 == 0:
                    print(f"Compared {comparisons}/{total_comparisons} pairs...")
        
        print(f"Analysis complete! Found {len(pairs)} duplicate pairs across different PDFs")
        
        if pairs:
            # Sort by similarity score (highest first)
            pairs.sort(key=lambda x: x['similarity'], reverse=True)
            print(f"Similarity scores range: {pairs[-1]['similarity']:.2%} to {pairs[0]['similarity']:.2%}")
        
        return pairs if pairs else None

    def clear(self):
        """
        Reset stored data.
        """
        self.filepaths = []
        self.pdf_names = []
        self.pages = []
        self.image_indices = []
        self._images = []
        print("Cleared all stored image data")