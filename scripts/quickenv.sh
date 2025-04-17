#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create virtual environment
/opt/homebrew/bin/python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch>=2.2 transformers>=4.40 datasets sacrebleu accelerate

# Confirm environment is ready
echo "✅  Env ready"
