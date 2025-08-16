#!/bin/bash

# Installation script for d3-skill-discovery project
# This script installs both the rsl_rl and unsupervised_RL packages in editable mode

set -e  # Exit on any error

echo "Installing d3-skill-discovery packages..."

# Install rsl_rl package
echo "Installing rsl_rl package..."
pip install -e ./rsl_rl/

# Install unsupervised_RL package  
echo "Installing unsupervised_RL package..."
pip install -e ./exts/unsupervised_RL/

echo "✅ All packages installed successfully!"
echo ""
echo "Installed packages:"
echo "  - rsl_rl (editable)"
echo "  - unsupervised_RL (editable)"
