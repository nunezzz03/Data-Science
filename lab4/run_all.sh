#!/bin/bash
# Lab 4 - Run All Models
# Usage: ./run_all.sh

cd "$(dirname "$0")"

echo "=============================================="
echo "LAB 4 - MODELLING AND OVERFITTING"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
pip3 install -q pandas numpy scikit-learn matplotlib

# Optional: Install XGBoost for extra model
echo "Installing XGBoost (optional)..."
pip3 install -q xgboost 2>/dev/null || echo "XGBoost not installed (optional)"

# Run the main script
echo ""
echo "Running Lab 4 pipeline..."
python3 run_all.py

echo ""
echo "=============================================="
echo "DONE!"
echo "=============================================="
