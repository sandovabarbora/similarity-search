# Environment variables
VENV_NAME := env
VENV_BIN := $(VENV_NAME)/bin
PYTHON := python3
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
COVERAGE := $(VENV_BIN)/coverage
FLAKE8 := $(VENV_BIN)/flake8
BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
PROJECT_NAME := strv-similarity-search

# API and Frontend settings
API_HOST := 0.0.0.0
API_PORT := 8000
STREAMLIT_PORT := 8501

# Operating system detection
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV_NAME)/Scripts
    PYTHON_VENV := $(VENV_BIN)/python.exe
    RM_CMD := rd /s /q
    ACTIVATE_CMD := .\env\Scripts\activate
else
    PYTHON_VENV := $(VENV_BIN)/python
    RM_CMD := rm -rf
    ACTIVATE_CMD := source env/bin/activate
endif

.PHONY: help activate check-venv create-venv install-deps setup clean lint test format coverage docker-build docker-run process-data dev-setup run-api run-frontend run-all check-logs

help:
	@echo "Available commands:"
	@echo "make activate     - Show how to activate the virtual environment"
	@echo "make setup       - Create virtual environment and install all dependencies"
	@echo "make dev-setup  - Setup development environment (including dev dependencies)"
	@echo "make clean      - Remove Python cache files and virtual environment"
	@echo "make lint       - Run linting checks"
	@echo "make test       - Run tests"
	@echo "make format     - Format code using black and isort"
	@echo "make coverage   - Run tests with coverage report"
	@echo "make run-api    - Run FastAPI server"
	@echo "make run-frontend - Run Streamlit frontend"
	@echo "make run-all    - Run both API and frontend"
	@echo "make check-logs - Check application logs"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run   - Run Docker container"
	@echo "make process-data - Process Flickr30k dataset"

activate:
	@echo "To activate the virtual environment, run:"
	@echo "source env/bin/activate  # For Unix/Linux/MacOS"
	@echo ".\env\Scripts\activate   # For Windows"

check-venv:
	@if [ ! -f "$(PYTHON_VENV)" ]; then \
		echo "Virtual environment not found. Creating one..."; \
		$(PYTHON) -m venv $(VENV_NAME); \
	fi

create-venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_NAME)
	@echo "Upgrading pip..."
	@$(PYTHON_VENV) -m pip install --upgrade pip

install-deps: check-venv
	@echo "Installing dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully!"

dev-setup: check-venv
	@echo "Installing development dependencies..."
	@$(PIP) install -r requirements.txt
	@$(PIP) install -r requirements-dev.txt
	@$(PYTHON_VENV) -m pre_commit install
	@echo "Development environment setup complete!"

setup: create-venv install-deps
	@echo "Basic setup complete! Run 'make dev-setup' if you need development tools."
	@echo "Don't forget to activate your virtual environment:"
	@echo "$(ACTIVATE_CMD)"

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec $(RM_CMD) {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec $(RM_CMD) {} +
	@find . -type d -name "*.egg" -exec $(RM_CMD) {} +
	@find . -type d -name ".pytest_cache" -exec $(RM_CMD) {} +
	@find . -type d -name ".coverage" -exec $(RM_CMD) {} +
	@find . -type d -name "htmlcov" -exec $(RM_CMD) {} +
	@find . -type d -name "dist" -exec $(RM_CMD) {} +
	@find . -type d -name "build" -exec $(RM_CMD) {} +
	@find . -type d -name ".eggs" -exec $(RM_CMD) {} +
	@[ -d $(VENV_NAME) ] && $(RM_CMD) $(VENV_NAME) || true
	@echo "Cleanup complete!"

lint: check-venv
	@echo "Running flake8..."
	@$(FLAKE8) src tests
	@echo "Running mypy..."
	@$(VENV_BIN)/mypy src
	@echo "Running pylint..."
	@$(VENV_BIN)/pylint src tests

test: check-venv
	@echo "Running tests..."
	@$(PYTEST) tests -v

format: check-venv
	@echo "Formatting code..."
	@$(BLACK) src tests
	@$(ISORT) src tests
	@echo "Formatting complete!"

coverage: check-venv
	@echo "Running tests with coverage..."
	@$(COVERAGE) run -m pytest tests/
	@$(COVERAGE) report
	@$(COVERAGE) html
	@echo "Coverage report generated in htmlcov/index.html"

run-api: check-venv
	@echo "Starting FastAPI server..."
	@$(PYTHON_VENV) -m uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

run-frontend: check-venv
	@echo "Starting Streamlit frontend..."
	@$(PYTHON_VENV) -m streamlit run src/frontend/streamlit_app.py --server.port $(STREAMLIT_PORT)

run-all: check-venv
	@echo "Starting both API and frontend..."
	@$(PYTHON_VENV) -m uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload & \
	$(PYTHON_VENV) -m streamlit run src/frontend/streamlit_app.py --server.port $(STREAMLIT_PORT)

docker-build:
	@echo "Building Docker image..."
	docker build -t $(PROJECT_NAME) .

docker-run:
	@echo "Running Docker container..."
	docker run -p $(API_PORT):$(API_PORT) -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) $(PROJECT_NAME)

process-data: check-venv
	@echo "Processing Flickr30k dataset..."
	@if [ -z "$(FLICKR_DIR)" ]; then \
		echo "Error: FLICKR_DIR not set. Usage: make process-data FLICKR_DIR=/path/to/flickr30k"; \
		exit 1; \
	fi
	@$(PYTHON_VENV) src/scripts/process_flickr.py \
		--flickr_dir $(FLICKR_DIR) \
		--output_file data/flickr30k_features.h5 \
		--batch_size 32

check-logs:
	@echo "Checking application logs..."
	@tail -f logs/app.log