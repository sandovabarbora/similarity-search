# Unit Tests for STRV Similarity Search

This directory contains unit tests for the STRV Similarity Search application. The tests cover various components of the system, including feature extraction, search algorithms, API endpoints, and utility functions.

## Test Structure

The tests are organized by component:

- `test_image_feature_extractor.py` - Tests for the ResNet50-based feature extractor
- `test_optimized_search.py` - Tests for the search engines (FAISS and brute force)
- `test_api_endpoints.py` - Tests for the FastAPI endpoints
- `test_utils.py` - Tests for utility functions and logging
- `test_process_images.py` - Tests for the image processing pipeline

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_image_feature_extractor.py
```

To run tests with verbose output:

```bash
pytest -v
```

To generate a test coverage report:

```bash
pytest --cov=src
```

## Test Dependencies

The tests require the following additional packages:

- pytest
- pytest-cov
- requests

These can be installed via:

```bash
pip install pytest pytest-cov requests
```

## Writing New Tests

When adding new tests, follow these guidelines:

1. Place tests in the appropriate test file based on component
2. Use fixtures for common setup and teardown
3. Use appropriate mocking to isolate the component under test
4. Include both positive and negative test cases
5. Follow the naming convention `test_<component>_<behavior>.py`

## Common Test Fixtures

Several fixtures are available in `conftest.py` for common test scenarios:

- `temp_h5_file`: Creates a temporary H5 file for testing
- `temp_image_dir`: Creates a directory with test images
- `sample_image_file`: Creates a sample image file for API testing
- `mock_initialized_app`: Sets up a mocked, initialized API for testing

## Important Notes

1. **Testing the Feature Extractor**: Some tests for the feature extractor are marked with `@pytest.mark.skip` as they require a GPU or real images. These can be enabled by removing the skip decorator when needed.

2. **Path Resolution**: Tests for path resolution use temporary files and directories to ensure they work across different environments.

3. **Mocking Dependencies**: External dependencies like torch, FAISS, and h5py are mocked where appropriate to ensure tests can run in CI environments.

4. **Test Isolation**: Each test is designed to be isolated from others, so they can run in any order.