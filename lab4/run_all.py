"""
Lab 4 - Run All Models
Run this script to execute the complete pipeline:
1. Prepare data (with data leakage fix)
2. Train all 7 models
3. Generate summary and comparisons
"""

import os
import sys

# Add lab4 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🎓 LAB 4 - MODELLING AND OVERFITTING")
print("=" * 70)

# Step 1: Prepare Data
print("\n" + "=" * 70)
print("STEP 1: DATA PREPARATION")
print("=" * 70)
import prepare_data

prepare_data.prepare_all()

# Step 2: Train Models
print("\n" + "=" * 70)
print("STEP 2: TRAINING MODELS")
print("=" * 70)

# Import and run each model
from models import naive_bayes
from models import logistic_regression
from models import knn
from models import decision_tree
from models import mlp
from models import random_forest
from models import gradient_boosting

print("\n--- Running Naive Bayes ---")
naive_bayes.run()

print("\n--- Running Logistic Regression ---")
logistic_regression.run()

print("\n--- Running KNN ---")
knn.run()

print("\n--- Running Decision Tree ---")
decision_tree.run()

print("\n--- Running MLP ---")
mlp.run()

print("\n--- Running Random Forest ---")
random_forest.run()

print("\n--- Running Gradient Boosting ---")
gradient_boosting.run()

# Step 3: Generate Summary
print("\n" + "=" * 70)
print("STEP 3: GENERATING SUMMARY")
print("=" * 70)
import summary

summary.run()

print("\n" + "=" * 70)
print("✅ LAB 4 COMPLETE!")
print("=" * 70)
print(f"\nResults saved to: lab4/results/")
print(f"Images saved to: lab4/images/")
print("\nGenerated images include:")
print("  - Hyperparameter study for each model")
print("  - Overfitting study (train vs test)")
print("  - Feature importance (where applicable)")
print("  - Model comparison charts")
