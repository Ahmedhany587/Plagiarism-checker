import os
import shutil
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union
from contextlib import contextmanager

import fitz  # PyMuPDF
from PIL import Image
import imagehash
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class PDFImageExtractor:
    """Extracts all unique images from PDF documents using temporary storage."""

    def __init__(
        self,
        out_dir: Optional[Union[str, Path]] = None,
        min_std_dev: float = 10.0,
        min_width: int = 50,
        min_height: int = 50,
    ):
        """
        Initialize PDF Image Extractor.
        
        Args:
            out_dir: Output directory for extracted images (None = temp directory)
            min_std_dev: Minimum color variance to consider image relevant (default: 10.0)
                        Images with lower variance are considered plain colors and filtered out
            min_width: Minimum image width in pixels (default: 50)
            min_height: Minimum image height in pixels (default: 50)
        """
        self._out_dir_path = Path(out_dir) if out_dir else None
        self._temp_dir_manager = None
        self.out_dir = None
        self.min_std_dev = min_std_dev
        self.min_width = min_width
        self.min_height = min_height

    def __enter__(self):
        """Context manager entry - sets up output directory."""
        if self._out_dir_path:
            # Use specified directory
            self.out_dir = self._out_dir_path
            if self.out_dir.exists():
                shutil.rmtree(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using permanent directory: {self.out_dir}")
        else:
            # Use temporary directory with automatic cleanup
            self._temp_dir_manager = tempfile.TemporaryDirectory(prefix="pdf_images_")
            self.out_dir = Path(self._temp_dir_manager.name)
            logger.info(f"Using temporary directory: {self.out_dir}")
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp directory if needed."""
        if self._temp_dir_manager:
            self._temp_dir_manager.cleanup()
            logger.info(f"Cleaned up temporary directory: {self.out_dir}")
            self._temp_dir_manager = None
        self.out_dir = None

    @classmethod
    @contextmanager
    def create_temp_extractor(cls):
        """Convenience context manager factory for temporary extraction."""
        with cls(out_dir=None) as extractor:
            yield extractor

    @classmethod
    @contextmanager
    def create_permanent_extractor(
        cls,
        out_dir: Union[str, Path],
    ):
        """Convenience context manager factory for permanent directory extraction."""
        with cls(out_dir=out_dir) as extractor:
            yield extractor

    def is_image_relevant(self, image_bytes: bytes) -> bool:
        """
        Checks if the image is relevant for plagiarism detection.
        
        Filters out:
        - Plain/solid color images (low color variance)
        - Very small images (likely decorative elements)
        - Invalid images
        
        Args:
            image_bytes: Image data as bytes
        
        Returns:
            True if image is relevant, False if it should be filtered out
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            
            # Filter 1: Check image size (skip very small images)
            width, height = img.size
            if width < self.min_width or height < self.min_height:
                logger.debug(f"Skipping small image: {width}x{height} pixels")
                return False
            
            # Filter 2: Check for plain/solid color images
            # Convert to RGB if needed and then to numpy array
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            
            # Calculate standard deviation for each color channel
            std_devs = np.std(img_array, axis=(0, 1))
            avg_std = np.mean(std_devs)
            
            # If average standard deviation is too low, it's a plain color image
            if avg_std < self.min_std_dev:
                logger.debug(f"Skipping plain color image (std dev: {avg_std:.2f})")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking image relevance: {e}")
            return False

    def get_phash(self, image_bytes: bytes) -> Optional[str]:
        """Computes perceptual hash of the image."""
        try:
            img = Image.open(BytesIO(image_bytes)).convert("L")
            return str(imagehash.phash(img, hash_size=16))
        except Exception:
            return None

    def extract_images_from_pdf(self, pdf_path: str) -> List[dict]:
        """Extracts all unique images from a given PDF file."""
        if self.out_dir is None:
            raise RuntimeError("PDFImageExtractor must be used as a context manager. Use 'with PDFImageExtractor(...) as extractor:'")
        
        results = []
        if not os.path.isfile(pdf_path):
            logger.warning(f"PDF not found, skipping: {pdf_path}")
            return results

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Cannot open PDF {pdf_path}: {e}")
            return results

        base_name = Path(pdf_path).stem
        
        # Reset seen_hashes for each PDF to allow cross-PDF duplicate detection
        # This ensures the same image in different PDFs is extracted from both
        pdf_seen_hashes = set()
        
        # Tracking counters for filtering statistics
        total_images = 0
        filtered_small = 0
        filtered_plain = 0
        filtered_duplicate = 0

        for page_num, page in enumerate(doc, start=1):
            for idx, img in enumerate(page.get_images(full=True), start=1):
                total_images += 1
                xref = img[0]
                data = doc.extract_image(xref)
                img_bytes = data.get("image")

                if not img_bytes:
                    continue
                
                # Check if image is relevant (filters plain colors and small images)
                if not self.is_image_relevant(img_bytes):
                    # Try to determine why it was filtered for statistics
                    try:
                        img = Image.open(BytesIO(img_bytes))
                        if img.size[0] < self.min_width or img.size[1] < self.min_height:
                            filtered_small += 1
                        else:
                            filtered_plain += 1
                    except:
                        pass
                    continue

                phash = self.get_phash(img_bytes)
                # Only skip if duplicate within the SAME PDF (not across PDFs)
                if not phash or phash in pdf_seen_hashes:
                    if phash in pdf_seen_hashes:
                        filtered_duplicate += 1
                    continue

                pdf_seen_hashes.add(phash)

                ext = data.get("ext", "png")
                filename = f"{base_name}_p{page_num}_i{idx}.{ext}"
                filepath = self.out_dir / filename

                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                results.append({
                    "filepath": str(filepath), 
                    "pdf": base_name, 
                    "page": page_num,
                    "image_index": idx
                })

        doc.close()
        
        # Log extraction summary with filtering statistics
        filtered_total = filtered_small + filtered_plain + filtered_duplicate
        logger.info(
            f"Extracted {len(results)} images from {pdf_path} "
            f"(Total: {total_images}, Filtered: {filtered_total} "
            f"[{filtered_small} too small, {filtered_plain} plain color, {filtered_duplicate} duplicates])"
        )
        return results


# Image duplication detection functions have been moved to image_duplication_detector.py



