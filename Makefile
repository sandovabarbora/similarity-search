SHELL := /bin/bash

# Detect operating system
ifeq ($(OS),Windows_NT)
    PYTHON := python
    VENV_ACTIVATE := env\Scripts\activate.bat
    PIP := pip
else
    PYTHON := python3
    VENV_ACTIVATE := source env/bin/activate
    PIP := pip3
endif

# Phony targets
.PHONY: help setup install activate test test-coverage lint format clean

# Default target
help:
	@echo "Similarity Search Project Makefile"
	@echo "======================================"
	@echo "Available commands:"
	@echo "  make setup     - Create virtual environment"
	@echo "  make install   - Install project dependencies"
	@echo "  make activate  - Show environment activation instructions"
	@echo "  make test      - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make lint      - Run code quality checks"
	@echo "  make format    - Format code with black"
	@echo "  make clean     - Remove virtual environment"

# Create virtual environment
setup:
	@echo "Creating virtual environment..."
	@if [ ! -d env ]; then \
		$(PYTHON) -m venv env; \
	fi
	@( \
		. env/bin/activate; \
		$(PIP) install --upgrade pip; \
	)
	@echo "Virtual environment created successfully."
	@echo "Activate with: source env/bin/activate"

# Install dependencies
install:
	@( \
		. env/bin/activate; \
		$(PIP) install -r requirements.txt; \
		$(PIP) install pytest pytest-cov flake8 black isort autoflake; \
	)
	@echo "Dependencies installed successfully."

# Show activation instructions
activate:
	@echo "To activate the virtual environment:"
	@echo "  source env/bin/activate  (Unix/Linux/macOS)"
	@echo "  env\Scripts\activate     (Windows)"

# Run tests
test:
	@( \
		. env/bin/activate; \
		PYTHONPATH=. pytest tests/ -v -x --tb=short; \
	)

# Run tests with coverage
test-coverage:
	@( \
		. env/bin/activate; \
		PYTHONPATH=. pytest --cov=src tests/ \
		--cov-report=term-missing \
		--cov-report=html; \
	)

# Run linters
lint:
	@( \
		. env/bin/activate; \
		flake8 src/ tests/; \
		black --check src/ tests/; \
		isort --check src/ tests/; \
	)

# Format code
format:
	@( \
		. env/bin/activate; \
		black src/ tests/; \
		isort src/ tests/; \
	)

# Clean virtual environment
clean:
	@echo "Removing virtual environment..."
	@rm -rf env
	@echo "Virtual environment removed."

# Full check (lint and test)
check: lint test