#!/bin/bash

# Lab 1 - Complete Pipeline Runner
# Runs all models and generates results in ~3-5 minutes

echo "🚀 Starting Lab 1 Complete Pipeline"
echo "====================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Data Preparation
echo "📊 Step 1/7: Preparing datasets (fixing leakage issues)..."
python models/prepare_data.py
if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi
echo ""

# Step 2: Naive Bayes
echo "🔵 Step 2/7: Training Naive Bayes models..."
python models/naive_bayes.py
echo ""

# Step 3: KNN
echo "🟣 Step 3/7: Training K-Nearest Neighbors..."
python models/knn.py
echo ""

# Step 4: Decision Trees
echo "🌳 Step 4/7: Training Decision Trees..."
python models/decision_tree.py
echo ""

# Step 5: Logistic Regression
echo "🟠 Step 5/7: Training Logistic Regression..."
python models/logistic_regression.py
echo ""

# Step 6: MLP
echo "🧠 Step 6/7: Training Multi-Layer Perceptron..."
python models/mlp.py
echo ""

# Step 7: Summary
echo "📋 Step 7/7: Generating final summary..."
python models/summary.py
echo ""

# Final Status
echo "====================================="
echo "✅ Pipeline Complete!"
echo ""
echo "📁 Generated Files:"
echo "   - data/processed/*.csv (8 train/test files)"
echo "   - images/*.png (30 charts)"
echo "   - results/baseline_results_summary.csv"
echo ""
echo "📖 See docs/FINAL_PERFORMANCE_RESULTS.md for full analysis"
echo ""
