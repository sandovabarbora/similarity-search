import os
import sys
from pathlib import Path

import pytest
import torch


# Add project root to Python path
@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """
    Make the parent directory of 'tests' importable by adding it to sys.path.
    This allows imports to work the same way they do in the application.
    """
    # Get the absolute path to the project root directory
    project_root = Path(__file__).parent.absolute().parent
    
    # Add project root to sys.path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Print paths for debugging
    print(f"Project root added to sys.path: {project_root}")
    print(f"Current sys.path: {sys.path}")
    
    # Verify src directory exists
    src_dir = project_root / "src"
    if src_dir.exists():
        print(f"src directory found: {src_dir}")
    else:
        print(f"WARNING: src directory not found at {src_dir}")
    
    yield
    
    # No cleanup needed for sys.path

# Setup test environment variables
@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set environment variables for testing."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["TESTING"] = "1"
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.absolute().parent)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# Create empty directories needed for tests
@pytest.fixture(scope="session", autouse=True)
def setup_directories():
    """Create directories needed for tests if they don't exist."""
    base_dir = Path(__file__).parent.parent
    dirs_to_create = [
        base_dir / "logs",
        base_dir / "models",
        base_dir / "src" / "models",
    ]
    
    # Create directories if they don't exist
    for directory in dirs_to_create:
        directory.mkdir(exist_ok=True, parents=True)
        print(f"Ensuring directory exists: {directory}")
    
    yield

# Improved mock for torch backends
@pytest.fixture(autouse=True)
def mock_torch_environment():
    """
    Mock torch backends to provide consistent test environment.
    Allows explicit control over device availability.
    """
    import unittest.mock as mock

    # Determine the best available device
    def determine_best_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    # Mocking device availability
    with (
        mock.patch('torch.cuda.is_available', return_value=False),
        mock.patch('torch.backends.mps.is_available', return_value=False),
        mock.patch('torch.device', side_effect=lambda x: x if x == 'cpu' else determine_best_device())
    ):
        yield

# Additional configuration for better error handling
def pytest_configure(config):
    """
    Configure pytest to provide more detailed error reporting.
    """
    config.addinivalue_line(
        "markers", 
        "integration: mark test as an integration test that may require additional resources"
    )