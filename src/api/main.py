from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import uuid
import time
from pathlib import Path
import json
import h5py
import numpy as np
from PIL import Image
import io
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.utils.logger import logger, Logger
from src.processing.image_feature_extractor import ImageFeatureExtractor
from src.processing.text_feature_extractor import TextFeatureExtractor
from src.utils.log_decorators import log_function_call

# Initialize FastAPI app
app = FastAPI(title="STRV Similarity Search")

# Initialize feature extractors
image_extractor = ImageFeatureExtractor()
text_extractor = TextFeatureExtractor()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-computed features at startup
def load_features():
    try:
        # Load image features
        logger.info("Loading pre-computed image features")
        with h5py.File('models/features.h5', 'r') as f:
            image_features = f['features'][:]
            image_paths = [path.decode('utf-8') for path in f['image_paths'][:]]
        logger.info(f"Loaded {len(image_features)} image feature vectors")

        # Load text features
        logger.info("Loading pre-computed text features")
        with h5py.File('models/tweet_features.h5', 'r') as f:
            text_features = f['features'][:]
            texts = [text.decode('utf-8') for text in f['texts'][:]]
            tweet_ids = [tid.decode('utf-8') for tid in f['tweet_ids'][:]]
            timestamps = [ts.decode('utf-8') for ts in f['timestamps'][:]]
            retweets = f['retweets'][:]
            likes = f['likes'][:]
        logger.info(f"Loaded {len(text_features)} text feature vectors")

        return (image_features, image_paths), (text_features, texts, tweet_ids, timestamps, retweets, likes)
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        raise

# Load features
(image_features, image_paths), (text_features, texts, tweet_ids, timestamps, retweets, likes) = load_features()

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
    return {
        "status": "ok",
        "image_features_loaded": len(image_features),
        "text_features_loaded": len(text_features)
    }

@app.post("/search/image")
async def search_similar_images(file: UploadFile = File(...)):
    """Upload an image and find similar images in the dataset."""
    req_logger = Logger().add_extra_fields(
        filename=file.filename,
        content_type=file.content_type
    )
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Extract features
        query_features = image_extractor.extract_features(image)
        
        # Find similar images
        indices, scores = image_extractor.compute_similarity(
            query_features,
            image_features,
            9
        )
        
        # Prepare results
        similar_images = []
        for idx, score in zip(indices, scores):
            similar_images.append({
                'path': image_paths[idx],
                'similarity_score': float(score),
                'rank': len(similar_images) + 1
            })
        
        return {"status": "success", "results": similar_images}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        req_logger.exception(f"Error processing image search: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image search")

@app.post("/search/text")
async def search_similar_texts(text: str = Form(...)):
    """Find similar texts in the dataset."""
    req_logger = Logger().add_extra_fields(text_length=len(text))
    
    try:
        # Extract features
        query_features = text_extractor.extract_features(text)
        
        # Find similar texts
        indices, scores = text_extractor.compute_similarity(
            query_features,
            text_features,
            9
        )
        
        # Prepare results
        similar_texts = []
        for idx, score in zip(indices, scores):
            similar_texts.append({
                'text': texts[idx],
                'tweet_id': tweet_ids[idx],
                'timestamp': timestamps[idx],
                'retweets': int(retweets[idx]),
                'likes': int(likes[idx]),
                'similarity_score': float(score),
                'rank': len(similar_texts) + 1
            })
        
        return {"status": "success", "results": similar_texts}
        
    except Exception as e:
        req_logger.exception(f"Error processing text search: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing text search")

@app.get("/stats")
async def get_stats():
    """Get statistics about both datasets"""
    try:
        return {
            "images": {
                "total": len(image_paths),
                "feature_dimension": image_features.shape[1],
                "sample_paths": image_paths[:5]
            },
            "texts": {
                "total": len(texts),
                "feature_dimension": text_features.shape[1],
                "sample_texts": texts[:5]
            }
        }
    except Exception as e:
        logger.exception(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting STRV Similarity Search API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)