# STRV Similarity Search

## Project Overview
STRV Similarity Search is a visual intelligence platform that uses deep learning to find visually similar images within a dataset. The system extracts feature vectors from images using a ResNet50 model and enables efficient similarity search with FAISS optimization when available.

## System Architecture

### Key Components
- **Feature Extraction Service**: ResNet50-based extractor that generates 2048-dimensional feature vectors
- **FastAPI Backend**: RESTful API for image processing and similarity search
- **Streamlit Frontend**: User-friendly interface for uploading images and viewing results
- **Optimized Search**: FAISS-accelerated search when available, with fallback to brute force
- **Structured Logging**: Comprehensive logging system with JSON formatting and rotation

### Data Flow
1. User uploads an image through the Streamlit interface
2. Image is sent to FastAPI backend
3. Feature extractor processes the image
4. System searches for similar images in the pre-computed feature database using either FAISS or brute force
5. Results are returned and displayed to the user

## Setup and Installation

### Prerequisites
- Python 3.9+
- pip (package installer)
- CUDA-capable GPU (optional but recommended for faster processing)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/organization/strv-similarity-search
cd strv-similarity-search

# Create a virtual environment
python -m venv env

# Activate the environment
# On Windows
env\Scripts\activate
# On Unix/Linux/macOS
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Feature Database Preparation
Before running the application, you need to process images to create the feature database:

```bash
python src/process_images.py --input_dir /path/to/images --output_file src/models/features.h5 --batch_size 32
```

This will:
1. Scan the input directory for images
2. Extract features using ResNet50
3. Save the features to an H5 file for fast retrieval

## Running the Application

### Starting the Backend API
```bash
# Navigate to the src directory
cd src

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Starting the Frontend
```bash
# In a separate terminal, navigate to the project directory
cd strv-similarity-search

# Run the Streamlit app
streamlit run src/app.py
```

Once both are running, access the web interface at `http://localhost:8501`

## API Reference

### GET /health
Check system health and initialization status.

Response:
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

### POST /search/image
Upload an image and find similar images in the dataset.

Request:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

Response:
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

### GET /stats
Get statistics about the dataset.

Response:
```json
{
    "images": {
        "total": 10000,
        "feature_dimension": 2048,
        "sample_paths": [...]
    },
    "api_status": "ready",
    "search_method": "optimized",
    "search_status": "ready",
    "path_cache_loaded": true,
    "using_faiss": true,
    "h5_file": "path/to/features.h5"
}
```

## Frontend Features

The Streamlit frontend provides a user-friendly interface for:
- Image uploading and similarity search
- Exploration of the image dataset
- Filtering and category browsing
- Authentication and user preferences
- Saving favorite images

## Technical Details

### Feature Extraction
- Based on ResNet50 pretrained on ImageNet
- Outputs 2048-dimensional feature vectors
- Supports CPU, CUDA, and MPS (Apple Silicon) processing
- Includes batch processing for efficient dataset processing

### Similarity Search
- Primary implementation: FAISS-based search (when available)
- Fallback: Optimized brute force search with batching
- Cosine similarity metric for comparing feature vectors
- Top-k retrieval with memory-efficient heap implementation

### Logging System
- Structured JSON logging with rotation
- Console logging with color formatting
- Separate error logging
- Request tracking with unique IDs
- Performance monitoring

## Memory Optimization
- Batch processing of large datasets
- Efficient path caching for faster lookup
- Lazy loading of heavy components
- Memory-efficient search algorithms

## Troubleshooting

### Common Issues

1. **API initialization fails**
   - Check that the features.h5 file exists in the models directory
   - Verify the file format and structure match expectations

2. **Image search is slow**
   - Enable FAISS for faster search (requires additional installation)
   - Reduce the dataset size or optimize the feature vectors

3. **Path lookup fails**
   - Ensure image paths in the H5 file are valid
   - Use the ensure_valid_path function to handle path variations

4. **Out of memory errors**
   - Reduce batch size in process_images.py
   - Use a machine with more RAM, especially for larger datasets

## Future Improvements
1. Implement more sophisticated user authentication system
2. Add persistent storage for user preferences and saved images
3. Expand search capabilities with text and metadata search
4. Include image clustering and tag recommendations
5. Extend to video similarity search
