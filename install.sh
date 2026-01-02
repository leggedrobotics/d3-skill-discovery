#!/bin/bash

# Installation script for d3-skill-discovery project
# This script installs both the d3_rsl_rl and d3_skill_discovery packages in editable mode

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Installing d3-skill-discovery packages..."

# Install d3_rsl_rl package
echo "Installing d3_rsl_rl package..."
pip install -e "${SCRIPT_DIR}/d3_rsl_rl/"

# Install d3_skill_discovery package
echo "Installing d3_skill_discovery package..."
pip install -e "${SCRIPT_DIR}/exts/d3_skill_discovery/"

echo "✅ All packages installed successfully!"
echo ""
echo "Installed packages:"
echo "  - d3_rsl_rl (editable)"
echo "  - d3_skill_discovery (editable)"
