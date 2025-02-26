#!/bin/bash

# Create __init__.py files based on actual project structure

# Create directories if they don't exist
mkdir -p src/{api,processing,utils,scripts,frontend}
mkdir -p tests

# Create __init__.py files in all needed directories
touch src/__init__.py
touch src/api/__init__.py
touch src/processing/__init__.py
touch src/utils/__init__.py
touch src/scripts/__init__.py
touch src/frontend/__init__.py
touch tests/__init__.py

echo "Created __init__.py files for all required directories"