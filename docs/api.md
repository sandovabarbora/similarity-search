# STRV Similarity Search API Documentation

This document provides detailed information about the STRV Similarity Search API implementation, endpoints, and usage.

## Overview

The STRV Similarity Search API is built with FastAPI and provides a RESTful interface for image similarity search. It uses deep learning-based feature extraction (ResNet50) and efficient search algorithms to find visually similar images in a large dataset.

## Core Features

- **Optimized Search**: Utilizes FAISS for high-performance vector similarity search when available
- **Fallback Mechanism**: Gracefully falls back to optimized brute force search if FAISS is unavailable
- **Efficient Feature Loading**: Loads image features from an H5 file with path caching
- **Health Monitoring**: Provides detailed system health checks and statistics
- **Structured Logging**: Comprehensive request tracking and error logging
- **CORS Support**: Full cross-origin request support for integration with web applications

## API Endpoints

### GET /health
Check system health and initialization status.

**Response**:
```json
{
    "status": "ok",
    "image_count": 10000,
    "search_method": "optimized",
    "search_status": "ready",
    "path_cache_loaded": true,
    "api_ready": true,
    "using_faiss": true,
    "h5_file": "path/to/features.h5"
}
```

**Status Codes**:
- `200 OK`: System is healthy
- `503 Service Unavailable`: API is initializing or encountered errors

### POST /search/image
Upload an image and find similar images in the dataset.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response**:
```json
{
    "status": "success",
    "results": [
        {
            "path": "path/to/image.jpg",
            "similarity_score": 0.95,
            "rank": 1
        },
        ...
    ]
}
```

**Status Codes**:
- `200 OK`: Search completed successfully
- `400 Bad Request`: Invalid file format (not an image)
- `500 Internal Server Error`: Error processing image search
- `503 Service Unavailable`: API is initializing

### GET /stats
Get statistics about the dataset.

**Response**:
```json
{
    "images": {
        "total": 10000,
        "feature_dimension": 2048,
        "sample_paths": [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg",
            ...
        ]
    },
    "texts": {
        "total": 0,
        "available": false,
        "feature_dimension": 0
    },
    "api_status": "ready",
    "search_method": "optimized",
    "search_status": "ready",
    "path_cache_loaded": true,
    "using_faiss": true,
    "h5_file": "path/to/features.h5"
}
```

**Status Codes**:
- `200 OK`: Statistics retrieved successfully
- `500 Internal Server Error`: Error retrieving statistics

## Initialization Process

The API initializes in the background using the following steps:

1. **Feature Loading**: Loads the H5 file containing pre-computed image features
2. **Path Caching**: Builds a lookup table for file paths to optimize retrieval
3. **Search Engine Setup**: Initializes either FAISS search or brute force search
4. **Ready State**: Marks the API as ready when initialization completes

## Search Implementation

### Optimized Search (with FAISS)
When FAISS is available, the API uses it for efficient nearest neighbor search:

1. Image is processed through the feature extractor (ResNet50)
2. Features are normalized and passed to the FAISS index
3. Top-k similar vectors are retrieved with similarity scores
4. Results are formatted with file paths and returned

### Brute Force Search (Fallback)
When FAISS is unavailable, an optimized brute force search is used:

1. Image is processed through the feature extractor
2. Features are compared batch-by-batch to the dataset
3. A min-heap is used to efficiently track the top-k results
4. Results are sorted and returned with similar formatting

## Path Resolution

The API implements a robust path resolution system that:

1. Tries multiple path formats to handle different deployment environments
2. Caches validated paths for faster access
3. Falls back to direct file access if cache is not loaded
4. Returns readable error messages when paths cannot be resolved

## Error Handling

The API implements comprehensive error handling:

1. Request validation errors (e.g., non-image files) return 400 responses
2. Internal errors during processing return 500 responses with details
3. All errors are logged with context for debugging
4. Initialization failures are reported through the health endpoint

## Middleware

A custom middleware logs each request with:
- Unique request ID
- HTTP method and URL
- Client IP address
- Processing time
- Status code
- Detailed error information (when applicable)

## CORS Configuration

The API is configured to allow cross-origin requests with:
- All origins allowed
- Credentials supported
- All methods and headers permitted

## Running the API

To run the API, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

The API can be configured through environment variables or by modifying the config.py file:

- **Database settings**: Connection strings for PostgreSQL
- **Redis settings**: Cache configuration
- **Storage settings**: Upload paths and vector dimensions
- **API settings**: Host and port configuration
- **Processing settings**: Batch size and worker count

## Performance Considerations

For optimal performance:
- Use FAISS when available for faster searches
- Ensure the H5 file is stored on fast storage (SSD preferred)
- Adjust the batch size based on available memory
- Consider using multiple workers in production