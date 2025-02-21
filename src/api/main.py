from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import uuid
import time
from pathlib import Path
import json
import h5py
import numpy as np
from PIL import Image
import io

from pathlib import Path
import sys
# Add just the src directory to Python path
src_dir = str(Path(__file__).parent.parent)
sys.path.append(src_dir)

from utils.logger import logger, Logger
from processing.feature_extractor import FeatureExtractor
from utils.log_decorators import log_function_call

# Initialize FastAPI app
app = FastAPI(title="STRV Similarity Search")

# Initialize feature extractor
feature_extractor = FeatureExtractor()

# Load pre-computed features at startup
def load_features():
    try:
        logger.info("Loading pre-computed features")
        with h5py.File('data/processed/features.h5', 'r') as f:
            features = f['features'][:]
            image_paths = [path.decode('utf-8') for path in f['image_paths'][:]]
        logger.info(f"Loaded {len(features)} feature vectors")
        return features, image_paths
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        raise

features, image_paths = load_features()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    req_logger = Logger().add_extra_fields(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_host=request.client.host if request.client else None
    )
    
    req_logger.info(f"Incoming request: {request.method} {request.url}")
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        req_logger.info(
            "Request completed",
            extra={
                'process_time_ms': round(process_time * 1000, 2),
                'status_code': response.status_code
            }
        )
        return response
        
    except Exception as e:
        req_logger.exception(f"Request failed: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {"status": "ok", "features_loaded": len(features)}

@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    """
    Upload an image and find similar images in the dataset.
    Returns a list of similar images with their similarity scores.
    """
    req_logger = Logger().add_extra_fields(
        filename=file.filename,
        content_type=file.content_type
    )
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        req_logger.info("Processing uploaded image")
        
        # Read and process image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Extract features
        query_features = feature_extractor.extract_features(image)
        
        # Find similar images
        indices, scores = feature_extractor.compute_similarity(
            query_features,
            features,
            9  # Pass top_k as a positional argument instead
        )
        
        # Prepare results
        similar_images = []
        for idx, score in zip(indices, scores):
            similar_images.append({
                'path': image_paths[idx],
                'similarity_score': float(score),
                'rank': len(similar_images) + 1
            })
        
        req_logger.info(
            "Search completed successfully",
            extra={'num_results': len(similar_images)}
        )
        
        return {
            "status": "success",
            "results": similar_images
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        req_logger.exception(f"Error processing search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing image search"
        )

@app.get("/stats")
async def get_stats():
    """Get statistics about the loaded dataset"""
    try:
        return {
            "total_images": len(image_paths),
            "feature_dimension": features.shape[1],
            "sample_paths": image_paths[:5]  # Show first 5 image paths
        }
    except Exception as e:
        logger.exception(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving statistics"
        )

@app.get("/images")
async def get_images(page: int = 0, limit: int = 20):
    """Get paginated images from the database"""
    try:
        start_idx = page * limit
        end_idx = start_idx + limit
        
        return {
            "status": "success",
            "total": len(image_paths),
            "images": image_paths[start_idx:end_idx],
            "page": page,
            "limit": limit
        }
    except Exception as e:
        logger.exception(f"Error getting images: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving images"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting STRV Similarity Search API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)