import os
import uuid
import shutil
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union
from contextlib import contextmanager

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import imagehash
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class PDFImageExtractor:
    """Extracts relevant and unique images from PDF documents using temporary storage."""

    def __init__(
        self,
        out_dir: Optional[Union[str, Path]] = None,
        min_byte_size: int = 1024,
        min_width: int = 50,
        min_height: int = 50,
        std_threshold: float = 10.0,
    ):
        self._out_dir_path = Path(out_dir) if out_dir else None
        self._temp_dir_manager = None
        self.out_dir = None
        
        self.min_byte_size = min_byte_size
        self.min_width = min_width
        self.min_height = min_height
        self.std_threshold = std_threshold
        self.seen_hashes = set()

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
    def create_temp_extractor(
        cls,
        min_byte_size: int = 1024,
        min_width: int = 50,
        min_height: int = 50,
        std_threshold: float = 10.0,
    ):
        """Convenience context manager factory for temporary extraction."""
        with cls(
            out_dir=None,
            min_byte_size=min_byte_size,
            min_width=min_width,
            min_height=min_height,
            std_threshold=std_threshold,
        ) as extractor:
            yield extractor

    @classmethod
    @contextmanager
    def create_permanent_extractor(
        cls,
        out_dir: Union[str, Path],
        min_byte_size: int = 1024,
        min_width: int = 50,
        min_height: int = 50,
        std_threshold: float = 10.0,
    ):
        """Convenience context manager factory for permanent directory extraction."""
        with cls(
            out_dir=out_dir,
            min_byte_size=min_byte_size,
            min_width=min_width,
            min_height=min_height,
            std_threshold=std_threshold,
        ) as extractor:
            yield extractor

    def is_image_relevant(self, image_bytes: bytes) -> bool:
        """Determines whether an image is relevant based on size and visual content."""
        if len(image_bytes) < self.min_byte_size:
            return False
        try:
            img = Image.open(BytesIO(image_bytes))
            if img.width < self.min_width or img.height < self.min_height:
                return False
            gray = np.array(img.convert("L"))
            return np.std(gray) >= self.std_threshold
        except Exception:
            return False

    def get_phash(self, image_bytes: bytes) -> Optional[str]:
        """Computes perceptual hash of the image."""
        try:
            img = Image.open(BytesIO(image_bytes)).convert("L")
            return str(imagehash.phash(img, hash_size=16))
        except Exception:
            return None

    def extract_images_from_pdf(self, pdf_path: str) -> List[dict]:
        """Extracts relevant and unique images from a given PDF file."""
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

        for page_num, page in enumerate(doc, start=1):
            for idx, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                data = doc.extract_image(xref)
                img_bytes = data.get("image")

                if not img_bytes or not self.is_image_relevant(img_bytes):
                    continue

                phash = self.get_phash(img_bytes)
                if not phash or phash in self.seen_hashes:
                    continue

                self.seen_hashes.add(phash)

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
        logger.info(f"Extracted {len(results)} images from {pdf_path}")
        return results


# Image duplication detection functions have been moved to image_duplication_detector.py



