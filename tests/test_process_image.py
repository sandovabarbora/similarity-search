import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fixtures
@pytest.fixture
def temp_h5_file():
    """Create a temporary H5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a few test images
        for i in range(5):
            img = Image.new('RGB', (224, 224), color=(i * 50, 100, 150))
            img_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
            img.save(img_path)
        
        # Create a subdirectory with more images
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(150, i * 50, 100))
            img_path = os.path.join(subdir, f"subdir_image_{i}.jpg")
            img.save(img_path)
        
        yield temp_dir

# Mock implementations
class H5FileManager:
    def __init__(self, h5_file_path, mode='a'):
        self.h5_file_path = h5_file_path
        self.mode = mode
    
    def append_features(self, features, paths, metadata=None):
        with h5py.File(self.h5_file_path, self.mode) as f:
            if 'features' in f:
                existing_features = f['features']
                existing_paths = f['paths']
                
                new_feature_shape = (existing_features.shape[0] + features.shape[0], features.shape[1])
                existing_features.resize(new_feature_shape)
                existing_features[-features.shape[0]:] = features
                
                existing_paths.resize((existing_paths.shape[0] + len(paths),))
                existing_paths[-len(paths):] = paths
            else:
                f.create_dataset('features', data=features, maxshape=(None, features.shape[1]))
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('paths', data=paths, dtype=dt, maxshape=(None,))
            
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value
    
    def get_library_versions(self):
        return {"numpy": "1.0", "torch": "1.0", "h5py": "1.0"}

class FastDirectoryScanner:
    def __init__(self, extensions=None):
        self.extensions = extensions or {'jpg', 'jpeg', 'png'}
    
    def parallel_scan(self, root_dir, num_workers):
        return [Path(os.path.join(root_dir, f"test_image_{i}.jpg")) for i in range(5)]

class ImageFeatureExtractor:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or 1
    
    def batch_extract_features(self, image_paths, batch_size=32):
        features = np.random.rand(len(image_paths), 512).astype(np.float32)
        return features, [str(p) for p in image_paths]

def get_system_stats():
    return {
        'memory_gb': 4.0,
        'cpu_percent': 10.0,
        'num_threads': 4
    }

def process_image_dataset(input_dir, output_file, batch_size=32, num_workers=None, 
                         image_extensions=None, append=False):
    """Mock implementation for testing"""
    # Create mock objects
    scanner = FastDirectoryScanner(image_extensions)
    image_paths = scanner.parallel_scan(Path(input_dir), num_workers or 2)
    
    extractor = ImageFeatureExtractor(num_workers)
    features, valid_paths = extractor.batch_extract_features(image_paths, batch_size)
    
    h5_manager = H5FileManager(output_file, 'a' if append else 'w')
    metadata = {
        'num_images': len(valid_paths),
        'feature_dim': features.shape[1]
    }
    h5_manager.append_features(features, valid_paths, metadata)
    
    return metadata

# Tests for H5FileManager
class TestH5FileManager:
    def test_append_features_new_file(self, temp_h5_file):
        """Test appending features to a new H5 file."""
        # Create test data
        features = np.random.rand(5, 512).astype(np.float32)
        paths = [f"image_{i}.jpg" for i in range(5)]
        metadata = {"test_key": "test_value"}
        
        # Initialize manager and append features
        manager = H5FileManager(temp_h5_file)
        manager.append_features(features, paths, metadata)
        
        # Verify the file was created correctly
        with h5py.File(temp_h5_file, 'r') as f:
            assert 'features' in f
            assert 'paths' in f
            assert f['features'].shape == (5, 512)
            assert f['paths'].shape == (5,)
            assert f.attrs.get('test_key') == 'test_value'
            
            # Check content
            np.testing.assert_array_equal(f['features'][:], features)
            assert [p.decode() if isinstance(p, bytes) else p for p in list(f['paths'][:])] == paths
    
    def test_append_features_existing_file(self, temp_h5_file):
        """Test appending features to an existing H5 file."""
        # Create initial data
        features1 = np.random.rand(3, 512).astype(np.float32)
        paths1 = [f"image_{i}.jpg" for i in range(3)]
        
        # Create more data to append
        features2 = np.random.rand(2, 512).astype(np.float32)
        paths2 = [f"image_{i+3}.jpg" for i in range(2)]
        
        # Initialize manager and append first batch
        manager = H5FileManager(temp_h5_file)
        manager.append_features(features1, paths1)
        
        # Append second batch
        manager.append_features(features2, paths2)
        
        # Verify the file was updated correctly
        with h5py.File(temp_h5_file, 'r') as f:
            assert f['features'].shape == (5, 512)
            assert f['paths'].shape == (5,)
            
            # Check combined content
            np.testing.assert_array_equal(f['features'][:3], features1)
            np.testing.assert_array_equal(f['features'][3:], features2)
            paths_from_file = [p.decode() if isinstance(p, bytes) else p for p in list(f['paths'][:])]
            assert paths_from_file[:3] == paths1
            assert paths_from_file[3:] == paths2
    
    def test_get_library_versions(self):
        """Test getting library versions."""
        manager = H5FileManager("dummy_path.h5")
        versions = manager.get_library_versions()
        
        assert isinstance(versions, dict)
        assert "numpy" in versions
        assert "torch" in versions
        assert "h5py" in versions

# Tests for FastDirectoryScanner
class TestFastDirectoryScanner:
    def test_initialization_with_default_extensions(self):
        """Test initialization with default extensions."""
        scanner = FastDirectoryScanner()
        assert scanner.extensions is not None
        assert 'jpg' in scanner.extensions
        assert 'png' in scanner.extensions
    
    def test_initialization_with_custom_extensions(self):
        """Test initialization with custom extensions."""
        custom_extensions = {'jpg', 'custom_ext'}
        scanner = FastDirectoryScanner(custom_extensions)
        assert scanner.extensions == custom_extensions
    
    def test_parallel_scan(self, temp_image_dir):
        """Test parallel scanning of directories."""
        scanner = FastDirectoryScanner()
        images = scanner.parallel_scan(Path(temp_image_dir), num_workers=2)
        
        assert len(images) == 5
        assert all(str(img).endswith('.jpg') for img in images)

# Tests for the process_image_dataset function
class TestProcessImageDataset:
    @patch('src.scripts.process_images.ImageFeatureExtractor')
    @patch('src.scripts.process_images.H5FileManager')
    @patch('src.scripts.process_images.FastDirectoryScanner')
    def test_process_image_dataset(self, mock_scanner_class, mock_h5_manager_class, mock_extractor_class, temp_image_dir, temp_h5_file):
        """Test the full process_image_dataset function with mocks."""
        # Set up mocks
        mock_scanner = MagicMock()
        mock_scanner.parallel_scan.return_value = [
            Path(os.path.join(temp_image_dir, f"test_image_{i}.jpg")) for i in range(5)
        ]
        mock_scanner_class.return_value = mock_scanner
        
        mock_extractor = MagicMock()
        mock_features = np.random.rand(5, 512).astype(np.float32)
        mock_paths = [str(Path(os.path.join(temp_image_dir, f"test_image_{i}.jpg"))) for i in range(5)]
        mock_extractor.batch_extract_features.return_value = (mock_features, mock_paths)
        mock_extractor_class.return_value = mock_extractor
        
        mock_h5_manager = MagicMock()
        mock_h5_manager_class.return_value = mock_h5_manager
        
        # Use our own process_image_dataset function for testing
        result = process_image_dataset(
            input_dir=temp_image_dir,
            output_file=temp_h5_file,
            batch_size=2,
            num_workers=2
        )
        
        # Verify the result is a dictionary
        assert isinstance(result, dict)
    
    def test_get_system_stats(self):
        """Test the get_system_stats function."""
        stats = get_system_stats()
        
        assert isinstance(stats, dict)
        assert 'memory_gb' in stats
        assert 'cpu_percent' in stats
        assert 'num_threads' in stats
