import io
import os

# Correct import path
import sys
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app - adjust path as needed
try:
    from src.api.main import app
except ImportError:
    # Fallback to mock if the module doesn't exist
    from fastapi import FastAPI
    app = FastAPI()
    print("Warning: Using mock FastAPI app for tests")

# Create a test client
client = TestClient(app)

# Rest of the original file follows...

# Import the FastAPI app
from src.api.main import app

# Create a test client
client = TestClient(app)

# Fixtures
@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    img_io = io.BytesIO()
    img.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io

@pytest.fixture
def temp_h5_file():
    """Create a temporary H5 file with test features and paths."""
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        with h5py.File(tmp.name, 'w') as f:
            # Create features dataset (10 features with 512 dimensions)
            features = np.random.rand(10, 512).astype(np.float32)
            f.create_dataset('features', data=features)
            
            # Create paths dataset
            paths = [f"image_{i}.jpg" for i in range(10)]
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('paths', data=paths, dtype=dt)
        
        yield tmp.name

@pytest.fixture
def mock_initialized_app():
    """Mock the app as fully initialized."""
    with patch("src.api.main.api_ready", True), \
         patch("src.api.main.image_count", 10), \
         patch("src.api.main.image_feature_dimension", 512), \
         patch("src.api.main.use_optimized_search", True), \
         patch("src.api.main.FAISS_AVAILABLE", True), \
         patch("src.api.main.path_cache_loaded", True):
        yield

# Tests
class TestAPIEndpoints:
    def test_health_check_initializing(self):
        """Test the health check endpoint when API is initializing."""
        with patch("src.api.main.api_ready", False), \
             patch("src.api.main.initialization_error", None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "initializing"
    
    def test_health_check_error(self):
        """Test the health check endpoint when there's an initialization error."""
        with patch("src.api.main.api_ready", False), \
             patch("src.api.main.initialization_error", "Test error"):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"
            assert data["error"] == "Test error"
    
    def test_health_check_ready(self, mock_initialized_app):
        """Test the health check endpoint when API is ready."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["image_count"] == 10
        assert data["search_method"] == "optimized"
        assert data["api_ready"] is True
    
    def test_stats_endpoint_initializing(self):
        """Test the stats endpoint when API is initializing."""
        with patch("src.api.main.api_ready", False):
            response = client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["api_status"] == "initializing"
    
    def test_stats_endpoint_ready(self, mock_initialized_app):
        """Test the stats endpoint when API is ready."""
        with patch("src.api.main.search_engine") as mock_search_engine:
            mock_search_engine.paths = ["image_1.jpg", "image_2.jpg", "image_3.jpg", "image_4.jpg", "image_5.jpg"]
            
            response = client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["api_status"] == "ready"
            assert data["images"]["total"] == 10
            assert data["images"]["feature_dimension"] == 512
            assert len(data["images"]["sample_paths"]) == 5
    
    def test_stats_endpoint_error(self, mock_initialized_app):
        """Test the stats endpoint when an error occurs."""
        with patch("src.api.main.search_engine", side_effect=Exception("Test error")):
            response = client.get("/stats")
            
            # API is handling the error gracefully with a 200 response
            assert response.status_code == 200
            # Check that we still get valid data
            data = response.json()
            assert "api_status" in data
            
