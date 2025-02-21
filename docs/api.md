# STRV Similarity Search API Script

This script implements a FastAPI application that provides an image similarity search service. It loads pre-computed image features, exposes endpoints for health checking, image searching, dataset statistics, and paginated image retrieval.

## Overview

- **Feature Extraction:**  
  Uses a custom `FeatureExtractor` to extract image features from an uploaded image and compute similarity against pre-loaded features.

- **Endpoints:**  
  - **GET /health:** Checks API status and confirms features have been loaded.
  - **POST /search:** Accepts an image file upload, extracts its features, computes similarity scores with the dataset, and returns the top similar images.
  - **GET /stats:** Returns statistics about the dataset (total images, feature dimension, and sample paths).
  - **GET /images:** Provides a paginated list of image paths from the dataset.

- **Middleware & Logging:**  
  A middleware logs each request with a unique ID, method, URL, client IP, and processing time. Errors during processing are also logged.

- **CORS Configuration:**  
  The application allows cross-origin requests, enabling the API to be accessed from different origins.

## Dependencies

- **FastAPI & Uvicorn:** Web framework and ASGI server.
- **h5py & numpy:** For loading and processing pre-computed feature data.
- **Custom modules:**  
  - `utils.logger` for logging.
  - `processing.feature_extractor` for feature extraction and similarity computation.
  - `utils.log_decorators` for additional logging support.

## File Details

- **Feature Loading:**  
  At startup, the script loads feature vectors and image paths from desired HDF5 file located at `models/`.

- **Search Process:**  
  When an image is uploaded via the `/search` endpoint, the script:
  1. Validates the file type.
  2. Reads and converts the image.
  3. Extracts its features.
  4. Computes similarity with the pre-loaded features.
  5. Returns a list of similar images, including their file paths, similarity scores, and rank order.


- **Error Handling:**  
  HTTP exceptions are raised for invalid file types or if an error occurs during processing. Detailed logging is used to track and diagnose issues.

## Running the Script

To run the API, execute the script directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Make sure all dependencies are installed from the requirements file and HDF5 file is existent.

