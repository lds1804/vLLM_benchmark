#!/bin/bash
echo "Updating packages and installing vLLM and dependencies for the Benchmark..."

# Optional: Update pip
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt

echo "Installation finished successfully! You can now execute the server initialization script."
