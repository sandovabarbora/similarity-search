SHELL := /bin/bash

# Choose python3 on Unix, python on Windows
ifeq ($(OS),Windows_NT)
    PYTHON := python
else
    PYTHON := python3
endif

.PHONY: help setup install activate shell clean

help:
	@echo "Available commands:"
	@echo "  make setup    - Create (if needed) the virtual environment and upgrade pip"
	@echo "  make install  - Install dependencies (must be run within an activated environment, or use 'make setup' first)"
	@echo "  make activate - Print instructions for manually activating the virtual environment"
	@echo "  make clean    - Remove the virtual environment"

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
	@echo "  call env\\Scripts\\activate.bat   (Windows CMD)"
	@echo "  or .\\env\\Scripts\\Activate.ps1  (Windows PowerShell)"
else
	@echo "  source env/bin/activate           (Unix/Linux/macOS)"
endif
	@echo "Then run 'make install' to install dependencies."
	@echo "To deactivate the environment, run 'deactivate'."
	@echo "To remove the environment, run 'make clean'."
	@echo "In case of any issues, refer to the 'make help' or README.md."
	@echo "=============================================================================="

install:
	@echo "Installing dependencies..."
	@env/bin/pip install -r requirements.txt
	@echo "Dependencies installed!"	

clean:
	@echo "Removing virtual environment..."
	@if [ -d env ]; then rm -rf env; fi
	@echo "Environment removed."
