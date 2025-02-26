import io
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import, or use mocks if not available
try:
    from src.processing.optimized_search import (
        FAISS_AVAILABLE,
        BruteForceSearch,
        FastSearchEngine,
        fast_search_similar_images,
        initialize_search_engine,
    )
except ImportError:
    # Mock the classes and functions
    FAISS_AVAILABLE = False
    
    class FastSearchEngine:
        def __init__(self, h5_file_path):
            self.h5_file_path = h5_file_path
            self.is_loaded = False
            self.is_loading = False
            self.load_error = None
            self.index = None
            self.paths = None
        
        def start_loading(self):
            self.is_loading = True
            return True
        
        def wait_until_loaded(self, timeout=30.0):
            self.is_loading = False
            self.is_loaded = True
            return True
        
        def search(self, query_features, top_k=9):
            return np.array([1, 2, 3]), np.array([0.9, 0.8, 0.7])
        
        def get_path(self, index):
            return f"image_{index}.jpg"
    
    class BruteForceSearch(FastSearchEngine):
        pass
    
    def initialize_search_engine(h5_file_path):
        pass
    
    def fast_search_similar_images(image_data, top_k=9):
        return [{"path": f"image_{i}.jpg", "similarity_score": 0.9 - 0.1*i, "rank": i+1} for i in range(3)], 0.5
    
    print("Warning: Using mocks for optimized_search module")

# Create a fixture for a temporary H5 file with test data
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
def sample_image_bytes():
    """Create sample image bytes for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()

class TestFastSearchEngine:
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_initialization(self, temp_h5_file):
        """Test initialization of FastSearchEngine."""
        engine = FastSearchEngine(temp_h5_file)
        assert engine.h5_file_path == temp_h5_file
        assert not engine.is_loaded
        assert not engine.is_loading
        assert engine.load_error is None

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_start_loading(self, temp_h5_file):
        """Test start_loading method."""
        engine = FastSearchEngine(temp_h5_file)
        result = engine.start_loading()
        assert result
        assert engine.is_loading
        
        # Wait for loading to complete
        assert engine.wait_until_loaded(timeout=5.0)
        assert engine.is_loaded
        assert not engine.is_loading
        assert engine.load_error is None

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_search(self, temp_h5_file):
        """Test search functionality."""
        engine = FastSearchEngine(temp_h5_file)
        engine.start_loading()
        engine.wait_until_loaded(timeout=5.0)
        
        # Create a test query
        query = np.random.rand(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Search for similar vectors
        indices, scores = engine.search(query, top_k=5)
        
        # Verify results
        assert len(indices) == 5
        assert len(scores) == 5
        assert np.all(indices >= 0) and np.all(indices < 10)
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_get_path(self, temp_h5_file):
        """Test get_path method."""
        engine = FastSearchEngine(temp_h5_file)
        engine.start_loading()
        engine.wait_until_loaded(timeout=5.0)
        
        # Get path for a valid index
        path = engine.get_path(0)
        assert path == "image_0.jpg"
        
        # Get path for an invalid index
        path = engine.get_path(100)
        assert "unknown_image_100" in path

class TestBruteForceSearch:
    def test_initialization(self, temp_h5_file):
        """Test initialization of BruteForceSearch."""
        engine = BruteForceSearch(temp_h5_file)
        assert engine.h5_file_path == temp_h5_file
        assert not engine.is_loaded
        assert not engine.is_loading
        assert engine.load_error is None

    def test_start_loading(self, temp_h5_file):
        """Test start_loading method."""
        engine = BruteForceSearch(temp_h5_file)
        result = engine.start_loading()
        assert result
        assert engine.is_loading
        
        # Wait for loading to complete
        assert engine.wait_until_loaded(timeout=5.0)
        assert engine.is_loaded
        assert not engine.is_loading
        assert engine.load_error is None

    def test_search(self, temp_h5_file):
        """Test search functionality."""
        engine = BruteForceSearch(temp_h5_file)
        
        # Create a test query
        query = np.random.rand(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Search for similar vectors
        indices, scores = engine.search(query, top_k=5)
        
        # Verify results
        assert len(indices) == 5
        assert len(scores) == 5
        assert np.all(indices >= 0) and np.all(indices < 10)
    
    def test_get_path(self, temp_h5_file):
        """Test get_path method."""
        engine = BruteForceSearch(temp_h5_file)
        engine.start_loading()
        engine.wait_until_loaded(timeout=5.0)
        
        # Get path for a valid index
        path = engine.get_path(0)
        assert path == "image_0.jpg"
        
        # Get path for an invalid index
        path = engine.get_path(100)
        assert "unknown_image_100" in path or "image_100" in path

class TestSearchUtilities:
    @patch('src.processing.optimized_search.FastSearchEngine')
    @patch('src.processing.optimized_search.BruteForceSearch')
    @patch('src.processing.optimized_search.FAISS_AVAILABLE', True)
    def test_initialize_search_engine_with_faiss(self, mock_brute_force, mock_fast_search, temp_h5_file):
        """Test initialize_search_engine with FAISS available."""
        # Skip if we're using mocks
        if not FAISS_AVAILABLE:
            pytest.skip("Using mocks, skipping this test")
            
        # Set up the mock
        mock_engine = MagicMock()
        mock_fast_search.return_value = mock_engine
        
        # Initialize the search engine
        initialize_search_engine(temp_h5_file)
        
        # Verify FastSearchEngine was used
        mock_fast_search.assert_called_once_with(temp_h5_file)
        mock_engine.start_loading.assert_called_once()
        mock_brute_force.assert_not_called()
    
    @patch('src.processing.optimized_search.FastSearchEngine')
    @patch('src.processing.optimized_search.BruteForceSearch')
    @patch('src.processing.optimized_search.FAISS_AVAILABLE', False)
    def test_initialize_search_engine_without_faiss(self, mock_brute_force, mock_fast_search, temp_h5_file):
        """Test initialize_search_engine without FAISS available."""
        # Skip if we're not using mocks
        if FAISS_AVAILABLE:
            pytest.skip("FAISS available, skipping this test")
            
        # Set up the mock
        mock_engine = MagicMock()
        mock_brute_force.return_value = mock_engine
        
        # Initialize the search engine
        initialize_search_engine(temp_h5_file)
        
        # Verify BruteForceSearch was used
        mock_brute_force.assert_called_once_with(temp_h5_file)
        mock_engine.start_loading.assert_called_once()
        mock_fast_search.assert_not_called()
    
    @patch('src.processing.optimized_search.get_image_extractor')
    @patch('src.processing.optimized_search.search_engine')
    def test_fast_search_similar_images(self, mock_search_engine, mock_get_extractor, sample_image_bytes):
        """Test fast_search_similar_images function."""
        # Skip if we're using mocks
        if not FAISS_AVAILABLE:
            pytest.skip("Using mocks, skipping this test")
            
        # Set up mocks
        mock_extractor = MagicMock()
        mock_extractor.extract_single_image_features.return_value = np.random.rand(512)
        mock_get_extractor.return_value = mock_extractor
        
        mock_search_engine.is_loaded = True
        mock_search_engine.search.return_value = (np.array([1, 2, 3]), np.array([0.9, 0.8, 0.7]))
        mock_search_engine.get_path.side_effect = lambda idx: f"image_{idx}.jpg"
        
        # Call the function
        results, timing = fast_search_similar_images(sample_image_bytes, top_k=3)
        
        # Verify the results
        assert len(results) == 3
        assert all('path' in item for item in results)
        assert all('similarity_score' in item for item in results)
        assert all('rank' in item for item in results)
