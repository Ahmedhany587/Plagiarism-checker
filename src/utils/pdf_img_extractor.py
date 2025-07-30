import os
import uuid
import shutil
import logging
import tempfile
import atexit
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import imagehash
import matplotlib.pyplot as plt
# FiftyOne imports moved to image_duplication_detector.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class PDFImageExtractor:
    """Extracts relevant and unique images from PDF documents using temporary storage."""

    def __init__(
        self,
        out_dir: Optional[str] = None,
        min_byte_size: int = 1024,
        min_width: int = 50,
        min_height: int = 50,
        std_threshold: float = 10.0,
    ):
        # Use temporary directory if no output directory specified
        if out_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="pdf_images_")
            self.out_dir = Path(self.temp_dir)
            self.is_temp = True
            # Register cleanup function
            atexit.register(self.cleanup)
        else:
            self.out_dir = Path(out_dir)
            self.is_temp = False
            self.temp_dir = None
            
        self.min_byte_size = min_byte_size
        self.min_width = min_width
        self.min_height = min_height
        self.std_threshold = std_threshold
        self.seen_hashes = set()

        # Only clear directory if it's not a temp dir (temp dirs are created fresh)
        if not self.is_temp and self.out_dir.exists():
            shutil.rmtree(self.out_dir)
            
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using {'temporary' if self.is_temp else 'permanent'} directory: {self.out_dir}")

    def cleanup(self):
        """Clean up temporary directory if created."""
        if self.is_temp and self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp directory."""
        self.cleanup()

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

                results.append({"filepath": str(filepath), "pdf": base_name})

        doc.close()
        logger.info(f"Extracted {len(results)} images from {pdf_path}")
        return results


# Image duplication detection functions have been moved to image_duplication_detector.py



