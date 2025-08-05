import os
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image
import imagehash


class ImageDuplicationDetector:
    """
    Detects duplicate images across PDFs using perceptual hashing.
    Fast, reliable, and efficient duplicate detection.
    """

    def __init__(self):
        """Initialize the image duplication detector."""
        # Storage for metadata
        self.filepaths: List[str] = []
        self.pdf_names: List[str] = []
        self._images: List[Image.Image] = []

    def load_images(self, results: List[Dict[str, str]]):
        """
        Load images and their PDF origin metadata.
        results: list of {"filepath": str, "pdf": str}
        """
        self.filepaths = []
        self.pdf_names = []
        images = []
        
        print(f"Loading {len(results)} images for analysis...")
        
        for i, r in enumerate(results):
            fp = r["filepath"]
            if not os.path.isfile(fp):
                print(f"Skipping missing file: {fp}")
                continue
                
            try:
                img = Image.open(fp).convert("RGB")
                images.append(img)
                self.filepaths.append(fp)
                self.pdf_names.append(r["pdf"])
                
                if (i + 1) % 50 == 0:  # Progress update every 50 images
                    print(f"Loaded {i + 1}/{len(results)} images...")
                    
            except Exception as e:
                print(f"Failed to load image {fp}: {str(e)}")
                continue
        
        self._images = images
        print(f"Successfully loaded {len(self._images)} images for analysis")

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
        self._images = []
        print("Cleared all stored image data")