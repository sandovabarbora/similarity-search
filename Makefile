SHELL := /bin/bash
# Choose python3 on Unix, python on Windows
ifeq ($(OS),Windows_NT)
 PYTHON := python
 VENV_ACTIVATE := env\\Scripts\\activate.bat
else
 PYTHON := python3
 VENV_ACTIVATE := source env/bin/activate
endif

# Phony targets
.PHONY: help setup install activate shell clean test test-coverage lint format autofix isort autoflake check style

help:
	@echo "Available commands:"
	@echo " make setup - Create (if needed) the virtual environment and upgrade pip"
	@echo " make install - Install dependencies (must be run within an activated environment, or use 'make setup' first)"
	@echo " make activate - Print instructions for manually activating the virtual environment"
	@echo " make test - Run all tests"
	@echo " make test-coverage - Run tests with coverage report"
	@echo " make lint - Run linter to check code quality"
	@echo " make format - Automatically format code with black"
	@echo " make isort - Automatically sort imports"
	@echo " make autoflake - Automatically remove unused imports and variables"
	@echo " make autofix - Run all auto-fixers (black, isort, autoflake)"
	@echo " make style - Run linting check only without auto-fixing"
	@echo " make check - Run both tests and linting"
	@echo " make clean - Remove the virtual environment"
	@echo " make deepclean - Comprehensive removal of all build artifacts and caches"

setup:
	@echo "Setting up environment..."
	@if [ ! -d env ]; then \
		echo "No virtual environment found. Creating 'env'..."; \
		$(PYTHON) -m venv env; \
	fi
	@env/bin/pip install --upgrade pip
	@echo "Setup complete!"
	@echo "=============================================================================="
	@echo "To activate your virtual environment manually, run one of the following:"
ifeq ($(OS),Windows_NT)
	@echo " call env\\Scripts\\activate.bat (Windows CMD)"
	@echo " or .\\env\\Scripts\\Activate.ps1 (Windows PowerShell)"
else
	@echo " source env/bin/activate (Unix/Linux/macOS)"
endif
	@echo "Then run 'make install' to install dependencies."
	@echo "To deactivate the environment, run 'deactivate'."
	@echo "To remove the environment, run 'make clean'."
	@echo "In case of any issues, refer to the 'make help' or README.md."
	@echo "=============================================================================="

install:
	@echo "Installing dependencies..."
	@env/bin/pip install -r requirements.txt
	@echo "Installing test and development dependencies..."
	@env/bin/pip install pytest pytest-cov flake8 black isort autoflake pytest-asyncio
	@echo "Dependencies installed!"

clean:
	@echo "Cleaning up project..."
	@$(RM) env 2>/dev/null || true
	@$(RM) .pytest_cache 2>/dev/null || true
	@$(RM) .coverage 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec $(RM) {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@echo "Environment and basic caches removed."

deepclean: clean
	@echo "Performing deep clean..."
	@$(RM) build 2>/dev/null || true
	@$(RM) dist 2>/dev/null || true
	@$(RM) *.egg-info 2>/dev/null || true
	@$(RM) htmlcov 2>/dev/null || true
	@$(RM) .tox 2>/dev/null || true
	@$(RM) .mypy_cache 2>/dev/null || true
	@$(RM) logs 2>/dev/null || true
	@$(RM) models 2>/dev/null || true
	@echo "Deep clean completed. All build artifacts and caches removed."

run-api:
	@echo "Running API server..."
	@env/bin/python src/api/main.py

run-streamlit:
	@echo "Running Streamlit app..."
	@env/bin/streamlit run src/frontend/app.py

process-images:
	@echo "Processing images..."
	@env/bin/python src/scripts/process_images.py --input_dir data/raw/images --output_file models/features.h5 --batch_size 32 

process-images-append:
	@echo "Processing images..."
	@env/bin/python src/scripts/process_images.py --input_dir data/raw/images --output_file models/features.h5 --batch_size 32 --append

########################################################################################

test:
	@echo "Running tests..."
ifeq ($(OS),Windows_NT)
	@cmd /c "env\\Scripts\\activate.bat && PYTHONPATH=. pytest -v -x --tb=short"
else
	@bash -c "source env/bin/activate && PYTHONPATH=. PYTHONWARNINGS=ignore pytest -v -x --tb=short"
endif
	@echo "Tests completed!"

test-coverage:
	@echo "Running tests with coverage..."
ifeq ($(OS),Windows_NT)
	@cmd /c "env\\Scripts\\activate.bat && pytest --cov=src tests/ --cov-report=term-missing --cov-report=html"
else
	@bash -c "source env/bin/activate && PYTHONPATH=. pytest --cov=src tests/ --cov-report=term-missing --cov-report=html"
endif
	@echo "Coverage report generated in htmlcov/"

lint:
	@echo "Running linter..."
ifeq ($(OS),Windows_NT)
	@cmd /c "env\\Scripts\\activate.bat && flake8 src/ tests/"
else
	@bash -c "source env/bin/activate && flake8 src/ tests/"
endif
	@echo "Linting completed!"

style: lint
	@echo "Style check completed!"

format:
	@echo "Formatting code with black..."
ifeq ($(OS),Windows_NT)
	@cmd /c "env\\Scripts\\activate.bat && black src/ tests/"
else
	@bash -c "source env/bin/activate && black src/ tests/"
endif
	@echo "Formatting completed!"

isort:
	@echo "Sorting imports with isort..."
ifeq ($(OS),Windows_NT)
	@cmd /c "env\\Scripts\\activate.bat && isort src/ tests/"
else
	@bash -c "source env/bin/activate && isort src/ tests/"
endif
	@echo "Import sorting completed!"

autoflake:
	@echo "Removing unused imports and variables with autoflake..."
ifeq ($(OS),Windows_NT)
	@cmd /c "env\\Scripts\\activate.bat && autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src/ tests/"
else
	@bash -c "source env/bin/activate && autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src/ tests/"
endif
	@echo "Autoflake completed!"

autofix: autoflake isort format
	@echo "All auto-fixers have been run!"

check: style test
	@echo "All checks completed!"
