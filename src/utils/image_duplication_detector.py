import logging
from pathlib import Path
from typing import List, Optional, Callable

import fiftyone as fo
import fiftyone.brain as fob

# Configure logging
logger = logging.getLogger(__name__)


class ImageDuplicationDetector:
    """Handles image duplication detection using FiftyOne and CLIP embeddings."""
    
    def __init__(self, dataset_name: str = "image_duplicates"):
        self.dataset_name = dataset_name
        self.dataset = None
        
    def build_dataset_from_results(self, results: List[dict]) -> fo.Dataset:
        """Creates a FiftyOne dataset from extracted images and stores PDF origin."""
        if fo.dataset_exists(self.dataset_name):
            fo.delete_dataset(self.dataset_name)

        self.dataset = fo.Dataset(self.dataset_name)
        samples = [fo.Sample(filepath=r["filepath"], pdf_name=r["pdf"]) for r in results]
        self.dataset.add_samples(samples)

        logger.info(f"Dataset '{self.dataset_name}' created with {len(samples)} images.")
        return self.dataset

    def detect_cross_pdf_duplicates(
        self, 
        dataset: Optional[fo.Dataset] = None,
        brain_key: str = "similarity_index",
        threshold: float = 0.8,
        progress_callback: Optional[Callable] = None
    ) -> Optional[List[dict]]:
        """Detects image duplicates across different PDF documents using CLIP embeddings."""
        if dataset is None:
            dataset = self.dataset
            
        if dataset is None:
            raise ValueError("No dataset available. Call build_dataset_from_results first or provide a dataset.")
            
        if dataset.has_brain_run(brain_key):
            dataset.delete_brain_run(brain_key)

        # Notify about embedding start
        if progress_callback:
            progress_callback("starting", 0, len(dataset), "Initializing CLIP model...")
        
        # Compute embeddings with progress tracking
        try:
            index = fob.compute_similarity(dataset, model="clip-vit-base32-torch", brain_key=brain_key, backend="sklearn")
            
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
        
        index.find_duplicates(thresh=threshold)
        
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

    def cleanup_dataset(self):
        """Clean up the FiftyOne dataset."""
        if fo.dataset_exists(self.dataset_name):
            fo.delete_dataset(self.dataset_name)
            logger.info(f"Cleaned up dataset: {self.dataset_name}")


# Backward compatibility functions for existing code
def build_dataset_from_results(results: List[dict], dataset_name: str) -> fo.Dataset:
    """
    Backward compatibility function for the old interface.
    Creates a FiftyOne dataset from extracted images and stores PDF origin.
    """
    detector = ImageDuplicationDetector(dataset_name)
    return detector.build_dataset_from_results(results)


def index_and_report_cross_pdf_duplicates(
    dataset: fo.Dataset, 
    brain_key: str, 
    thresh: float, 
    progress_callback=None
) -> Optional[List[dict]]:
    """
    Backward compatibility function for the old interface.
    Detects image duplicates across different PDF documents.
    """
    detector = ImageDuplicationDetector()
    detector.dataset = dataset
    return detector.detect_cross_pdf_duplicates(dataset, brain_key, thresh, progress_callback)