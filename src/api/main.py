import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils.logger import Logger, logger

# Global variables for search configuration
use_optimized_search = False
search_engine = None
FAISS_AVAILABLE = False

# Try to import optimized search module
try:
    from src.processing.optimized_search import (
        FAISS_AVAILABLE,
        fast_search_similar_images,
        initialize_search_engine,
    )

    use_optimized_search = True
    logger.info("Successfully imported optimized search module")
except Exception as e:
    logger.warning(f"Could not import optimized search: {e}. Will use original search.")

# Initialize FastAPI app
app = FastAPI(title="STRV Similarity Search")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for minimal startup
h5_file_path = "src/models/features.h5"
image_feature_dimension = None
image_count = None
api_ready = False
initialization_error = None

# Path cache for faster access during search
path_lookup = {}
path_cache_loaded = False


# Function to ensure a valid path is returned
def ensure_valid_path(path):
    """
    Ensure the path exists or try alternative paths
    """
    # If path is already valid, return it
    if os.path.exists(path):
        return path

    # Try different path formats
    path_options = [
        path,
        os.path.join("models", os.path.basename(path)),
        os.path.join("src", "models", os.path.basename(path)),
        os.path.abspath(path),
        os.path.join(project_root, path),
        os.path.join(project_root, "models", os.path.basename(path)),
        os.path.join(project_root, "src", "models", os.path.basename(path)),
    ]

    for option in path_options:
        if os.path.exists(option):
            logger.info(f"Found valid path: {option} for original path: {path}")
            return option

    logger.warning(f"Could not find valid path for: {path}")
    return path  # Return original path if no valid option found


# Initialize in background thread
def init_background():
    global image_feature_dimension, image_count, api_ready, initialization_error, path_cache_loaded
    global use_optimized_search, search_engine, FAISS_AVAILABLE

    try:
        # Check if H5 file exists
        h5_path = ensure_valid_path(h5_file_path)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at {h5_path}")

        logger.info(f"Using H5 file at: {h5_path}")

        # Get dataset metadata
        logger.info("Getting dataset metadata")
        with h5py.File(h5_path, "r") as f:
            image_feature_dimension = f["features"].shape[1]
            image_count = f["features"].shape[0]

            # Log available keys in the file
            logger.info(f"H5 file has keys: {list(f.keys())}")

            logger.info(
                f"Dataset has {image_count} images with {image_feature_dimension} dimensions"
            )

        # Initialize search engine
        if use_optimized_search:
            try:
                logger.info("Initializing optimized search engine")
                initialize_search_engine(h5_path)
                logger.info(f"Using {'FAISS' if FAISS_AVAILABLE else 'brute force'} search")
            except Exception as e:
                logger.error(
                    f"Error initializing optimized search: {e}. Falling back to original search."
                )
                use_optimized_search = False

        # If not using optimized search, load path cache
        if not use_optimized_search:
            path_thread = threading.Thread(target=lambda: load_path_cache(h5_path))
            path_thread.daemon = True
            path_thread.start()

        # Mark API as ready
        api_ready = True
        logger.info(f"API initialized successfully and ready to serve requests")

    except Exception as e:
        initialization_error = str(e)
        logger.exception(f"Error during API initialization: {str(e)}")


# Load path cache in a separate thread
def load_path_cache(h5_path):
    global path_lookup, path_cache_loaded

    try:
        logger.info("Starting background load of path lookup table")
        start_time = time.time()

        with h5py.File(h5_path, "r") as f:
            if "paths" not in f:
                logger.error(
                    f"'paths' dataset not found in H5 file. Available keys: {list(f.keys())}"
                )
                raise KeyError("'paths' dataset not found in H5 file")

            # Get paths and handle byte strings
            paths = f["paths"][:]
            decoded_paths = []

            for path in paths:
                if isinstance(path, bytes):
                    path = path.decode("utf-8")
                decoded_paths.append(path)

            # Build lookup dictionary with validated paths
            for i, path in enumerate(decoded_paths):
                valid_path = ensure_valid_path(path)
                path_lookup[i] = valid_path

        path_cache_loaded = True
        logger.info(
            f"Path cache loaded in {time.time() - start_time:.2f}s, {len(path_lookup)} entries"
        )

        # Log some example paths for debugging
        num_examples = min(5, len(path_lookup))
        for i in range(num_examples):
            logger.info(
                f"Path example {i}: {path_lookup[i]}, exists: {os.path.exists(path_lookup[i])}"
            )

    except Exception as e:
        logger.exception(f"Error loading path cache: {str(e)}")


# Start initialization in background
bg_thread = threading.Thread(target=init_background)
bg_thread.daemon = True
bg_thread.start()


# Get image path with fallback if cache not loaded
def get_image_path(index):
    """Get image path by index, with fallback to direct file access if cache not loaded"""
    if path_cache_loaded and index in path_lookup:
        return path_lookup[index]
    else:
        # Fallback - read directly from file
        try:
            with h5py.File(ensure_valid_path(h5_file_path), "r") as f:
                path = f["paths"][index]
                if isinstance(path, bytes):
                    path = path.decode("utf-8")
                return ensure_valid_path(path)
        except Exception as e:
            logger.error(f"Error getting path for index {index}: {e}")
            return f"image_{index}"


# Lazy-load image feature extractor only when needed
def get_image_extractor():
    """Lazy-load the image feature extractor only when needed"""
    from src.processing.image_feature_extractor import ImageFeatureExtractor

    return ImageFeatureExtractor()


# Original search function (kept as fallback)
def search_similar_images_from_file(image_data, top_k: int = 9):
    """Original brute-force image search with fixed normalization"""
    start_time = time.time()

    try:
        # Load image and extract features on-demand
        image = Image.open(io.BytesIO(image_data))

        # Get feature extractor only when needed
        extractor = get_image_extractor()
        query_features = extractor.extract_single_image_features(image)

        logger.info(f"Feature extraction took {time.time() - start_time:.2f}s")
        search_start = time.time()

        # Normalize query features
        query_features = query_features.reshape(1, -1).astype(np.float32)
        query_norm = np.linalg.norm(query_features)
        if query_norm > 0:
            query_features = query_features / query_norm

        # Use min heap for memory-efficient top-k tracking
        import heapq

        min_heap = []  # Will keep the top-k highest scores

        # Use smaller batch size for more stability
        batch_size = 10000

        with h5py.File(ensure_valid_path(h5_file_path), "r") as f:
            features = f["features"]
            total_vectors = features.shape[0]

            # Process in batches
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

        logger.info(f"Search completed in {time.time() - search_start:.2f}s")
        return top_indices, top_scores

    except Exception as e:
        logger.exception(f"Error in search: {e}")
        raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    req_logger = Logger().add_extra_fields(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_host=request.client.host if request.client else None,
    )

    req_logger.info(f"Incoming request: {request.method} {request.url}")
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        req_logger.info(
            "Request completed",
            extra={
                "process_time_ms": round(process_time * 1000, 2),
                "status_code": response.status_code,
            },
        )
        return response
    except Exception as e:
        req_logger.exception(f"Request failed: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global use_optimized_search, search_engine

    if initialization_error:
        return {"status": "error", "error": initialization_error}

    search_status = "initializing"
    if use_optimized_search and search_engine:
        if hasattr(search_engine, "is_loaded") and search_engine.is_loaded:
            search_status = "ready"
        elif hasattr(search_engine, "is_loading") and search_engine.is_loading:
            search_status = "loading"
        elif hasattr(search_engine, "load_error") and search_engine.load_error:
            search_status = "error"

    return {
        "status": "ok" if api_ready else "initializing",
        "image_count": image_count,
        "search_method": "optimized" if use_optimized_search else "original",
        "search_status": search_status,
        "path_cache_loaded": path_cache_loaded,
        "api_ready": api_ready,
        "using_faiss": FAISS_AVAILABLE if use_optimized_search else False,
        "h5_file": ensure_valid_path(h5_file_path),
    }


@app.post("/search/image")
async def search_similar_images(file: UploadFile = File(...)):
    """Upload an image and find similar images in the dataset."""
    global use_optimized_search

    if not api_ready:
        raise HTTPException(status_code=503, detail="API is initializing. Please try again later.")

    req_logger = Logger().add_extra_fields(
        filename=file.filename,
        content_type=file.content_type,
        search_method="optimized" if use_optimized_search else "original",
    )

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        content = await file.read()
        start_time = time.time()

        # Use optimized search if available
        if use_optimized_search:
            try:
                # Try optimized search
                similar_images, process_time = fast_search_similar_images(content)
                req_logger.info(f"Optimized image search completed in {process_time:.2f}s")

                # Validate paths
                for item in similar_images:
                    item["path"] = ensure_valid_path(item["path"])
                    req_logger.debug(
                        f"Path: {item['path']}, exists: {os.path.exists(item['path'])}"
                    )

                return {"status": "success", "results": similar_images}
            except Exception as e:
                # Fall back to original search if optimized fails
                req_logger.warning(
                    f"Optimized search failed: {e}. Falling back to original search."
                )
                use_optimized_search = False

        # Original search method (fallback)
        indices, scores = search_similar_images_from_file(content)

        # Get paths for results
        similar_images = []
        for idx, score in zip(indices, scores):
            path = get_image_path(idx)
            similar_images.append(
                {"path": path, "similarity_score": float(score), "rank": len(similar_images) + 1}
            )
            req_logger.debug(f"Result {len(similar_images)}: Path={path}, Score={score:.4f}")

        process_time = time.time() - start_time
        req_logger.info(f"Original image search completed in {process_time:.2f}s")

        return {"status": "success", "results": similar_images}

    except HTTPException as he:
        raise he
    except Exception as e:
        req_logger.exception(f"Error processing image search: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image search")


@app.get("/stats")
async def get_stats():
    """Get statistics about the dataset"""
    global use_optimized_search, search_engine

    if not api_ready:
        return {"api_status": "initializing", "eta_seconds": 5}  # Estimated time remaining

    try:
        # Get sample paths
        sample_paths = []
        if (
            use_optimized_search
            and search_engine
            and hasattr(search_engine, "paths")
            and search_engine.paths
        ):
            sample_paths = [ensure_valid_path(p) for p in search_engine.paths[:5]]
        elif path_cache_loaded:
            sample_paths = [
                path_lookup.get(i, f"image_{i}") for i in range(min(5, len(path_lookup)))
            ]

        # Validate paths
        valid_sample_paths = []
        for path in sample_paths:
            valid_path = ensure_valid_path(path)
            valid_sample_paths.append(valid_path)
            logger.debug(
                f"Sample path: {path} -> {valid_path}, exists: {os.path.exists(valid_path)}"
            )

        search_status = "initializing"
        if use_optimized_search and search_engine:
            if hasattr(search_engine, "is_loaded") and search_engine.is_loaded:
                search_status = "ready"
            elif hasattr(search_engine, "is_loading") and search_engine.is_loading:
                search_status = "loading"

        stats = {
            "images": {
                "total": image_count,
                "feature_dimension": image_feature_dimension,
                "sample_paths": valid_sample_paths,
            },
            "texts": {"total": 0, "available": False, "feature_dimension": 0},
            "api_status": "ready" if api_ready else "initializing",
            "search_method": "optimized" if use_optimized_search else "original",
            "search_status": search_status,
            "path_cache_loaded": path_cache_loaded,
            "using_faiss": FAISS_AVAILABLE if use_optimized_search else False,
            "h5_file": ensure_valid_path(h5_file_path),
        }

        return stats
    except Exception as e:
        logger.exception(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting STRV Similarity Search API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, reload=False)
