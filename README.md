# STRV Similarity Search Documentation

## Project Overview
STRV Similarity Search is an image similarity search system that uses deep learning to find visually similar images in a dataset. The system processes the Flickr30k dataset and allows users to upload images to find similar ones through a web interface.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Setup and Installation](#setup-and-installation)
3. [Components](#components)
4. [Usage](#usage)
5. [Development](#development)
6. [API Reference](#api-reference)

## System Architecture

### High-Level Overview
The system consists of three main components:
- Feature Extraction Service (ResNet50)
- FastAPI Backend
- Streamlit Frontend

### Data Flow
1. User uploads an image through the Streamlit interface
2. Image is sent to FastAPI backend
3. Feature extractor processes the image
4. System searches for similar images in the pre-computed feature database
5. Results are returned and displayed to the user

## Setup and Installation

### Prerequisites
- Python 3.9+
- Make (for using Makefile commands)
- Flickr30k dataset

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/sandovabarbora/strv-similarity-search
cd strv-similarity-search

# Create virtual environment and install dependencies
make setup

# Activate virtual environment but should be handled by Makefile
# On Unix/Linux/MacOS:
source env/bin/activate
# On Windows:
.\env\Scripts\activate
```

### Data Processing
Process the Flickr30k dataset to extract features:
```bash
#FLICKR_DIR by default in data/raw/flickr30k_images
make process-data FLICKR_DIR=/path/to/flickr30k
```

### Running the Application
```bash
# Run both API and frontend
make run-all

# Or run them separately
make run-api
make run-frontend
```

## Components

### Feature Extractor
Location: `src/processing/feature_extractor.py`

The feature extractor uses ResNet50 pretrained on ImageNet to generate image feature vectors. Key features:
- Removes final classification layer
- Outputs 2048-dimensional feature vectors
- Includes batch processing capabilities
- Normalizes features for better similarity comparison

### FastAPI Backend
Location: `src/api/main.py`

The API provides endpoints for:
- Image upload and processing
- Similarity search
- System health checks
- Dataset statistics

### Streamlit Frontend
Location: `src/frontend/streamlit_app.py`

Provides a user interface for:
- Image upload
- Similar image display
- System statistics
- Error handling

### Logging System
Location: `src/utils/logger.py`

Features:
- Structured JSON logging
- Separate error logs
- Request tracking
- Performance monitoring

## Usage

### Web Interface
1. Access the Streamlit interface at `http://localhost:8501`
2. Upload an image using the file uploader
3. Click "Find Similar Images"
4. View results in the grid display

### API Endpoints
- `GET /health` - Check system health
- `POST /search` - Find similar images
- `GET /stats` - Get dataset statistics

## Development

### Project Structure
```
strv-similarity-search/
├── data/
│   └── raw/
│       └── flickr30k_images/
├── src/
│   ├── api/
│   ├── frontend/
│   ├── processing/
│   ├── scripts/
│   ├── storage/
│   └── utils/
├── tests/
├── Makefile
└── requirements.txt
```

### Available Make Commands
```bash
make help          # Show available commands
make setup         # Set up environment
make dev-setup    # Set up development environment
make clean        # Clean up project files
make lint         # Run linting
make test         # Run tests
make format       # Format code
make run-all      # Run full application
make check-logs   # Monitor logs
```

### Adding New Features
1. Create feature branch
2. Implement changes
3. Add tests
4. Run formatting and linting
5. Submit pull request

## API Reference

### POST /search
Upload an image and find similar ones.

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
        }
    ]
}
```

### GET /stats
Get system statistics.

Response:
```json
{
    "total_images": 1000,
    "feature_dimension": 2048,
    "sample_paths": [...]
}
```

## Future Improvements
1. Add user authentication
2. Implement image caching
3. Add more similarity metrics
4. Improve search performance
5. Add image preprocessing options

## Troubleshooting

### Common Issues
1. Environment activation fails
   - Solution: Ensure Python 3.9+ is installed
   - Check virtual environment path

2. Feature extraction errors
   - Solution: Verify CUDA availability
   - Check image format support

3. API connection issues
   - Solution: Check ports are available
   - Verify API is running

### Logging
Monitor application logs:
```bash
make check-logs
```

## Additional Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)