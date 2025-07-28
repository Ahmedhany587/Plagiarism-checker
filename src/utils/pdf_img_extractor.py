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
import fiftyone as fo
import fiftyone.brain as fob

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


def build_dataset_from_results(results: List[dict], dataset_name: str) -> fo.Dataset:
    """Creates a FiftyOne dataset from extracted images and stores PDF origin."""
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    samples = [fo.Sample(filepath=r["filepath"], pdf_name=r["pdf"]) for r in results]
    dataset.add_samples(samples)

    logger.info(f"Dataset '{dataset_name}' created with {len(samples)} images.")
    return dataset


def index_and_report_cross_pdf_duplicates(dataset: fo.Dataset, brain_key: str, thresh: float, progress_callback=None) -> None:
    """Detects image duplicates across different PDF documents."""
    if dataset.has_brain_run(brain_key):
        dataset.delete_brain_run(brain_key)

    # Notify about embedding start
    if progress_callback:
        progress_callback("starting", 0, len(dataset), "Initializing CLIP model...")
    
    # Compute embeddings with progress tracking
    try:
        index = fob.compute_similarity(dataset, model="clip-vit-base32-torch", brain_key=brain_key)
        
        # Notify embedding completion
        if progress_callback:
            progress_callback("embedding_complete", len(dataset), len(dataset), "Computing similarity matrix...")
            
    except Exception as e:
        if progress_callback:
            progress_callback("error", 0, len(dataset), f"Embedding failed: {str(e)}")
        raise
    
    # Find duplicates
    if progress_callback:
        progress_callback("finding_duplicates", len(dataset), len(dataset), "Finding duplicate pairs...")
    
    index.find_duplicates(thresh=thresh)
    
    if progress_callback:
        progress_callback("complete", len(dataset), len(dataset), "Analysis complete!")

    dup_view = index.duplicates_view(type_field="dup_type", id_field="nearest_id", dist_field="distance")

    seen_pairs = set()
    cross_pdf_pairs = []

    for sample in dup_view:
        if sample["dup_type"] != "duplicate":
            continue

        orig_id = sample["nearest_id"]
        dup_id = sample.id

        orig_pdf = dataset[orig_id]["pdf_name"]
        dup_pdf = sample["pdf_name"]

        if orig_pdf == dup_pdf:
            continue

        pair_key = tuple(sorted((orig_id, dup_id)))
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        cross_pdf_pairs.append((orig_id, dup_id, sample["distance"]))

    if not cross_pdf_pairs:
        logger.info("No cross-PDF duplicates found.")
        return None

    logger.info("Cross-PDF duplicate pairs:")
    pairs_info = []
    for orig_id, dup_id, dist in cross_pdf_pairs:
        orig_sample = dataset[orig_id]
        dup_sample = dataset[dup_id]
        pair_info = {
            'orig_path': orig_sample.filepath,
            'dup_path': dup_sample.filepath,
            'orig_pdf': orig_sample.pdf_name,
            'dup_pdf': dup_sample.pdf_name,
            'distance': dist,
            'similarity': 1 - dist
        }
        pairs_info.append(pair_info)
        logger.info(
            f"{Path(orig_sample.filepath).name} ({orig_sample.pdf_name}) <-> "
            f"{Path(dup_sample.filepath).name} ({dup_sample.pdf_name}) dist={dist:.3f}"
        )
    
    return pairs_info



