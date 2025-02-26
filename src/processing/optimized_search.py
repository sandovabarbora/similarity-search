import io
import logging
import threading
import time
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Flag to determine if FAISS is available
try:
    import faiss

    FAISS_AVAILABLE = True
    logger.info("FAISS library successfully imported")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not available - will use brute force search")
except Exception as e:
    FAISS_AVAILABLE = False
    logger.warning(f"Error importing FAISS: {e} - will use brute force search")


class FastSearchEngine:
    """High-performance similarity search using FAISS with CPU support."""

    def __init__(self, h5_file_path: str):
        """Initialize the search engine with the path to the feature database."""
        self.h5_file_path = h5_file_path
        self.index = None
        self.paths = None
        self.feature_dim = None
        self.is_loaded = False
        self.is_loading = False
        self.load_error = None

        # Background loading lock
        self._load_lock = threading.Lock()

    def start_loading(self) -> bool:
        """Start loading the index in a background thread."""
        with self._load_lock:
            if self.is_loaded or self.is_loading:
                return False

            self.is_loading = True

            # Start background loading
            thread = threading.Thread(target=self._load_index)
            thread.daemon = True
            thread.start()
            return True

    def _load_index(self) -> None:
        """Load the feature database and build the FAISS index."""
        if not FAISS_AVAILABLE:
            with self._load_lock:
                self.is_loading = False
                self.load_error = "FAISS library not available"
            return

        try:
            logger.info(f"Loading features from {self.h5_file_path}")
            start_time = time.time()

            # Check dataset size first for memory management
            with h5py.File(self.h5_file_path, "r") as f:
                feature_shape = f["features"].shape
                num_vectors = feature_shape[0]
                self.feature_dim = feature_shape[1]

                # Load paths first since they're usually small
                logger.info(f"Loading {num_vectors} paths")
                self.paths = [
                    p.decode("utf-8") if isinstance(p, bytes) else p for p in f["paths"][:]
                ]

                # Decide on index type based on dataset size
                logger.info(f"Dataset has {num_vectors} vectors with {self.feature_dim} dimensions")

                # Use simple flat index for stability - no quantization
                logger.info("Using flat index for exact search (CPU only)")
                self.index = faiss.IndexFlatIP(self.feature_dim)

                # Load and process features in smaller batches to reduce memory pressure
                batch_size = 10000  # Smaller batch size for better stability
                for i in range(0, num_vectors, batch_size):
                    end = min(i + batch_size, num_vectors)
                    logger.info(f"Processing vectors {i} to {end}")

                    # Load batch
                    batch = f["features"][i:end].astype(np.float32)

                    # Check for NaN or infinity
                    if np.isnan(batch).any() or np.isinf(batch).any():
                        logger.warning(
                            f"Found NaN or Inf values in batch {i} to {end}, cleaning..."
                        )
                        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)

                    # Normalize for cosine similarity
                    norms = np.linalg.norm(batch, axis=1, keepdims=True)
                    # Avoid division by zero
                    norms[norms == 0] = 1.0
                    batch = batch / norms

                    # Add to index
                    self.index.add(batch)

                    # Explicit cleanup to reduce memory usage
                    del batch
                    if hasattr(faiss, "garbageCollectionPeriod"):
                        # This only exists in some faiss versions
                        faiss.garbageCollectionPeriod()

            load_time = time.time() - start_time
            logger.info(f"Index loaded and built in {load_time:.2f}s")

            with self._load_lock:
                self.is_loading = False
                self.is_loaded = True

        except Exception as e:
            logger.exception(f"Error loading index: {e}")
            with self._load_lock:
                self.is_loading = False
                self.load_error = str(e)

    def wait_until_loaded(self, timeout: float = 30.0) -> bool:
        """
        Wait until the index is loaded or timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if loaded successfully, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._load_lock:
                if self.is_loaded:
                    return True
                if not self.is_loading:
                    return False
            time.sleep(0.1)
        return False

    def search(self, query_features: np.ndarray, top_k: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors to the query in the index.

        Args:
            query_features: Query feature vector
            top_k: Number of results to return

        Returns:
            Tuple of (indices, scores)
        """
        with self._load_lock:
            if not self.is_loaded:
                if self.load_error:
                    raise RuntimeError(f"Index failed to load: {self.load_error}")
                raise RuntimeError("Index not loaded yet. Call wait_until_loaded() first.")

        # Handle potential shape issues
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)

        # Ensure correct data type
        query_features = query_features.astype(np.float32)

        # Check for NaN or infinity
        if np.isnan(query_features).any() or np.isinf(query_features).any():
            logger.warning("Query features contain NaN or Inf values, cleaning...")
            query_features = np.nan_to_num(query_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize query vector
        norm = np.linalg.norm(query_features)
        if norm > 0:
            query_features = query_features / norm

        # Perform search with error handling
        try:
            scores, indices = self.index.search(query_features, min(top_k, self.index.ntotal))
            return indices[0], scores[0]
        except Exception as e:
            logger.exception(f"FAISS search error: {e}")
            raise RuntimeError(f"Error during similarity search: {e}")

    def get_path(self, index: int) -> str:
        """Get the path for a given index."""
        if self.paths is None:
            raise RuntimeError("Paths not loaded")
        if 0 <= index < len(self.paths):
            return self.paths[index]
        else:
            return f"unknown_image_{index}"


# Fallback brute force search implementation
class BruteForceSearch:
    """Fallback search using brute force method."""

    def __init__(self, h5_file_path: str):
        """Initialize with the path to the feature database."""
        self.h5_file_path = h5_file_path
        self.paths = None
        self.is_loaded = False
        self.is_loading = False
        self.load_error = None

        # Background loading lock
        self._load_lock = threading.Lock()

    def start_loading(self) -> bool:
        """Start loading paths in a background thread."""
        with self._load_lock:
            if self.is_loaded or self.is_loading:
                return False

            self.is_loading = True

            # Start background loading
            thread = threading.Thread(target=self._load_paths)
            thread.daemon = True
            thread.start()
            return True

    def _load_paths(self) -> None:
        """Load paths for faster lookup."""
        try:
            logger.info(f"Loading paths from {self.h5_file_path}")
            start_time = time.time()

            with h5py.File(self.h5_file_path, "r") as f:
                self.paths = [
                    p.decode("utf-8") if isinstance(p, bytes) else p for p in f["paths"][:]
                ]

            logger.info(f"Loaded {len(self.paths)} paths in {time.time() - start_time:.2f}s")

            with self._load_lock:
                self.is_loading = False
                self.is_loaded = True

        except Exception as e:
            logger.exception(f"Error loading paths: {e}")
            with self._load_lock:
                self.is_loading = False
                self.load_error = str(e)

    def wait_until_loaded(self, timeout: float = 30.0) -> bool:
        """Wait until paths are loaded or timeout is reached."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._load_lock:
                if self.is_loaded:
                    return True
                if not self.is_loading:
                    return False
            time.sleep(0.1)
        return False

    def search(self, query_features: np.ndarray, top_k: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Perform brute force search for similar features."""
        # Reshape and normalize query
        query_features = query_features.reshape(1, -1).astype(np.float32)
        query_norm = np.linalg.norm(query_features)
        if query_norm > 0:
            query_features = query_features / query_norm

        # Use efficient brute force search with batching
        batch_size = 10000

        import heapq

        min_heap = []

        with h5py.File(self.h5_file_path, "r") as f:
            features = f["features"]
            total_vectors = features.shape[0]

            for i in range(0, total_vectors, batch_size):
                end = min(i + batch_size, total_vectors)

                # Load batch
                batch = features[i:end].astype(np.float32)

                # Normalize batch
                batch_norms = np.linalg.norm(batch, axis=1, keepdims=True)
                batch_norms[batch_norms == 0] = 1.0
                normalized_batch = batch / batch_norms

                # Compute similarities
                batch_scores = np.dot(query_features, normalized_batch.T)[0]

                # Update top-k results
                for j, score in enumerate(batch_scores):
                    if len(min_heap) < top_k:
                        heapq.heappush(min_heap, (score, i + j))
                    elif score > min_heap[0][0]:
                        heapq.heappushpop(min_heap, (score, i + j))

        # Get final results
        sorted_results = sorted(min_heap, reverse=True)
        top_scores = np.array([score for score, _ in sorted_results])
        top_indices = np.array([idx for _, idx in sorted_results])

        return top_indices, top_scores

    def get_path(self, index: int) -> str:
        """Get the path for a given index."""
        if not self.is_loaded:
            # Fallback to reading from file
            try:
                with h5py.File(self.h5_file_path, "r") as f:
                    return f["paths"][index].decode("utf-8")
            except Exception as e:
                logger.error(f"Error getting path for index {index}: {e}")
                return f"image_{index}"

        if 0 <= index < len(self.paths):
            return self.paths[index]
        else:
            return f"unknown_image_{index}"


# Global search engine instance
search_engine = None


def initialize_search_engine(h5_file_path: str) -> None:
    """Initialize the global search engine."""
    global search_engine

    if FAISS_AVAILABLE:
        logger.info("Initializing Fast Search Engine with FAISS")
        try:
            search_engine = FastSearchEngine(h5_file_path)
            search_engine.start_loading()
            return
        except Exception as e:
            logger.exception(f"Error initializing FAISS search engine: {e}")

    # Fallback to brute force
    logger.info("Initializing Brute Force Search Engine (fallback)")
    search_engine = BruteForceSearch(h5_file_path)
    search_engine.start_loading()


def get_image_extractor():
    """Lazy-load the image feature extractor only when needed."""
    from src.processing.image_feature_extractor import ImageFeatureExtractor

    return ImageFeatureExtractor()


def fast_search_similar_images(image_data, top_k: int = 9) -> Tuple[List[Dict[str, Any]], float]:
    """
    Search for similar images using the search engine.

    Args:
        image_data: Raw image data bytes
        top_k: Number of results to return

    Returns:
        Tuple of (results list, processing_time)
    """
    global search_engine

    if search_engine is None:
        raise RuntimeError("Search engine not initialized")

    # Make sure we have the search engine ready
    if not search_engine.is_loaded:
        logger.info("Waiting for search engine to be ready...")
        if not search_engine.wait_until_loaded(timeout=30.0):
            raise RuntimeError("Timed out waiting for search engine to be ready")

    start_time = time.time()

    try:
        # Load image and extract features
        image = Image.open(io.BytesIO(image_data))

        # Get feature extractor
        extractor = get_image_extractor()
        query_features = extractor.extract_single_image_features(image)

        logger.info(f"Feature extraction took {time.time() - start_time:.2f}s")
        search_start = time.time()

        # Search using search engine
        indices, scores = search_engine.search(query_features, top_k)

        # Format results
        results = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            results.append(
                {
                    "path": search_engine.get_path(idx),
                    "similarity_score": float(score),
                    "rank": i + 1,
                }
            )

        search_time = time.time() - search_start
        total_time = time.time() - start_time

        logger.info(f"Search completed in {search_time:.2f}s (total: {total_time:.2f}s)")

        return results, total_time

    except Exception as e:
        logger.exception(f"Error during image search: {e}")
        raise
