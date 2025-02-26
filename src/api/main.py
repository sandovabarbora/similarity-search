# Standard library imports
import sys
from pathlib import Path
import os
import time
import io
import json
import uuid
from typing import Dict, Any, Optional, List

# Web framework and API
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Image handling
from PIL import Image
# Configuration and Constants
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_ROOT)

# Logging and tracing
from utils.logger import logger, Logger

# Search and feature extraction
from processing.optimized_search import (
    initialize_mps_search, 
    mps_search_similar_images
)


# Maximum allowed image size (10 MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  

# Initialize FastAPI application
app = FastAPI(
    title="STRV Visual Similarity Search",
    description="Advanced image similarity search API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application state
class AppState:
    """Centralized application state management"""
    def __init__(self):
        self.startup_time = time.time()
        self.total_searches = 0
        self.failed_searches = 0
        self.last_error = None

app_state = AppState()

def ensure_valid_path(path: str) -> str:
    """
    Ensure a valid file path exists
    
    Args:
        path (str): Original path to validate
    
    Returns:
        str: Validated file path
    """
    path_options = [
        path,
        os.path.join("models", os.path.basename(path)),
        os.path.join("src", "models", os.path.basename(path)),
        os.path.abspath(path),
        os.path.join(PROJECT_ROOT, path),
        os.path.join(PROJECT_ROOT, "models", os.path.basename(path)),
        os.path.join(PROJECT_ROOT, "src", "models", os.path.basename(path))
    ]
    
    for option in path_options:
        if os.path.exists(option):
            logger.info(f"Found valid path: {option} for original path: {path}")
            return option
    
    logger.warning(f"Could not find valid path for: {path}")
    return path

def startup_event():
    """
    Initialize the application on startup
    """
    try:
        # Initialize search engine
        h5_file_path = ensure_valid_path('src/models/features.h5')
        initialize_mps_search(h5_file_path)
        
        logger.info("Visual Similarity Search API initialized successfully")
    except Exception as e:
        logger.exception(f"API initialization failed: {e}")
        app_state.last_error = str(e)

# Call startup event during import
startup_event()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log and trace all incoming requests
    
    Args:
        request (Request): Incoming HTTP request
        call_next (Callable): Next middleware or route handler
    
    Returns:
        Response: HTTP response
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    req_logger = Logger().add_extra_fields(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_host=request.client.host if request.client else None
    )
    
    req_logger.info(f"Incoming request: {request.method} {request.url}")
    
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
        req_logger.exception(f"Request processing failed: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status
    
    Returns:
        Dict[str, Any]: API health status
    """
    current_time = time.time()
    return {
        "status": "ok",
        "uptime_seconds": current_time - app_state.startup_time,
        "total_searches": app_state.total_searches,
        "failed_searches": app_state.failed_searches,
        "last_error": app_state.last_error
    }

@app.get("/stats")
async def get_stats():
    """
    Retrieve detailed API and dataset statistics
    
    Returns:
        Dict[str, Any]: Comprehensive system and dataset statistics
    """
    try:
        # Retrieve detailed statistics about the search index
        return {
            "total_images": 41779,  # From your diagnostic results
            "feature_dimensions": 2048,
            "search_configuration": {
                "similarity_method": "MPS-Optimized Cosine Similarity",
                "top_k_results": 9,
                "similarity_threshold": 0.7
            },
            "system_info": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "total_searches": app_state.total_searches
            }
        }
    except Exception as e:
        logger.exception(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve statistics")

@app.post("/search/image")
async def search_similar_images(file: UploadFile = File(...)):
    """
    Search for similar images based on uploaded image
    
    Args:
        file (UploadFile): Uploaded image file
    
    Returns:
        Dict[str, Any]: Search results with similar images
    """
    try:
        # Increment total search count
        app_state.total_searches += 1
        
        # Read and validate image
        content = await file.read()
        
        # Validate file size
        if len(content) > MAX_IMAGE_SIZE:
            app_state.failed_searches += 1
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {MAX_IMAGE_SIZE / 1024 / 1024} MB"
            )
        
        # Validate image
        try:
            Image.open(io.BytesIO(content))
        except Exception:
            app_state.failed_searches += 1
            raise HTTPException(
                status_code=400, 
                detail="Invalid image file"
            )
        
        # Perform similarity search
        similar_images = mps_search_similar_images(content)
        
        return {
            "status": "success", 
            "results": similar_images,
            "search_metadata": {
                "total_results": len(similar_images),
                "timestamp": time.time()
            }
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        app_state.failed_searches += 1
        app_state.last_error = str(e)
        logger.exception(f"Unexpected search error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal search error"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Logging configuration
    logger.info("Starting STRV Visual Similarity Search API")
    
    # Run the API
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        workers=1
    )