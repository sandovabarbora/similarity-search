# STRV Similarity Search

## Project Overview
STRV Similarity Search is a visual intelligence platform that uses deep learning to find visually similar images within a dataset. The system extracts feature vectors from images using a ResNet50 model and enables efficient similarity search.

## System Architecture

### Key Components
- **Feature Extraction**: ResNet50-based extractor that generates 2048-dimensional feature vectors 
- **FastAPI Backend**: RESTful API for image processing and similarity search
- **Streamlit Frontend**: User-friendly interface for uploading images and viewing results
- **Optimized Search**: MPS-optimized search when available, with fallback to brute force
- **Structured Logging**: Comprehensive logging system with JSON formatting and rotation

### Data Flow
1. User uploads an image through the Streamlit interface 
2. Image is sent to FastAPI backend
3. Feature extractor processes the image
4. System searches for similar images in the pre-computed feature database
5. Results are returned and displayed to the user

## Setup and Installation

### Prerequisites
- Python 3.9+
- pip (package installer) 

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/organization/strv-similarity-search  
cd strv-similarity-search

# Create a virtual environment 
make setup

# Activate the virtual environment
make activate

# Install dependencies
make install
```

#### Note:
If the `make activate` does not work, activate the enviroment manually, the `make setup` should provide suitable commands.

### Feature Database Preparation
Before running the application, you need to process images to create the feature database:

```bash
make process-images
```

To append new images to an existing database:

```bash
make process-images-append  
```

This will:
1. Scan the input directory for images
2. Extract features using ResNet50  
3. Save the features to an H5 file for fast retrieval

## Running the Application

### Starting the Backend API
```bash
make run-api
```

### Starting the Frontend
```bash  
make run-streamlit
```

Once both are running, access the web interface at `http://localhost:8501`

## Development Workflow

The project includes a Makefile with helpful commands for development and testing:

- `make setup`: Create (if needed) the virtual environment and upgrade pip 
- `make install`: Install dependencies  
- `make activate`: Print instructions for manually activating the virtual environment
- `make test`: Run all tests
- `make test-coverage`: Run tests with coverage report
- `make lint`: Run linter to check code quality 
- `make format`: Automatically format code with black
- `make isort`: Automatically sort imports
- `make autoflake`: Automatically remove unused imports and variables
- `make autofix`: Run all auto-fixers (black, isort, autoflake)
- `make style`: Run linting check only without auto-fixing
- `make check`: Run both tests and linting
- `make clean`: Remove the virtual environment
- `make deepclean`: Comprehensive removal of all build artifacts and caches

## Troubleshooting

### Common Issues  

1. **API initialization fails**
   - Check that the features.h5 file exists in the models directory
   - Verify the file format and structure match expectations

2. **Image search is slow** 
   - Enable MPS for faster search on compatible systems
   - Reduce the dataset size or optimize the feature vectors

3. **Path lookup fails**
   - Ensure image paths in the H5 file are valid  
   - The system attempts to resolve paths intelligently, but manual fixes may be needed

4. **Out of memory errors**
   - Reduce batch size when processing images
   - Use a machine with more RAM, especially for larger datasets

## Future Improvements
1. Implement more sophisticated user authentication system
2. Add persistent storage for user preferences and saved images  
3. Expand search capabilities with text and metadata search
4. Include image clustering and tag recommendations
5. Extend to video similarity search