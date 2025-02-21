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

# STRV Similarity Search

## Project Setup and Environment Management

### Prerequisites
- Python 3.9+
- Make (for using Makefile commands)

### Environment Management

#### Quick Start
```bash
# Clone the repository
git clone https://github.com/sandovabarbora/strv-similarity-search
cd strv-similarity-search

# Set up the entire environment (creates virtual env and installs dependencies)
make setup

# Install dependencies (if environment is already activated)
make install
```

### Available Make Commands

#### Environment Commands
- `make setup`: 
  - Creates a virtual environment if it doesn't exist
  - Upgrades pip
  - Prepares the development environment

- `make install`: 
  - Installs project dependencies
  - Must be run within an activated virtual environment

- `make activate`: 
  - Provides instructions for manually activating the virtual environment
  
- `make clean`: 
  - Removes the entire virtual environment


### Virtual Environment Activation

#### Unix/Linux/macOS
```bash
source env/bin/activate
```

#### Windows
```cmd
# CMD
call env\Scripts\activate.bat

# PowerShell
.\env\Scripts\Activate.ps1
```

### Troubleshooting

#### Common Setup Issues
1. **Python Version**: Ensure Python 3.9+ is installed
2. **Make Availability**: 
   - On Windows, install Make via Windows Subsystem for Linux or MinGW
   - On macOS, install via Homebrew: `brew install make`
   - On Linux, use package manager (e.g., `apt-get install make`)

3. **Virtual Environment**:
   - If activation fails, verify Python installation
   - Check that you're in the correct project directory
   - Ensure no conflicting Python environments are active

### Notes
- Always run `make` commands from the project root directory
- To exit the virtual environment, use the `deactivate` command
- For detailed project setup, refer to the full documentation in this README

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
│   └── processed/
│   └── uploads/
├── src/
│   ├── api/
│   ├── frontend/
│   ├── processing/
│   ├── scripts/
│   ├── storage/
│   └── utils/
│── models/
├── tests/
├── Makefile
└── requirements.txt
```

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


## Additional Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)