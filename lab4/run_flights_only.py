"""
Run all models for Flights dataset only
"""
import os
import sys

# Add lab4 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🎓 LAB 4 - FLIGHTS ONLY")
print("=" * 60)

import lab4_config as config

for flights in [config.DATASETS[1], config.DATASETS[3]]: # Flights Raw and Flights Lab3
    print(f"\nRunning all models for: {flights['name']}")
    print(f"Target: {flights['target']}")
    print("=" * 60)

    from models import naive_bayes
    naive_bayes.run_for_dataset(flights['file_tag'], flights['target'])

    from models import logistic_regression
    logistic_regression.run_for_dataset(flights['file_tag'], flights['target'])

    from models import knn
    knn.run_for_dataset(flights['file_tag'], flights['target'])

    from models import decision_tree
    decision_tree.run_for_dataset(flights['file_tag'], flights['target'])

    from models import mlp
    mlp.run_for_dataset(flights['file_tag'], flights['target'])

    from models import random_forest
    random_forest.run_for_dataset(flights['file_tag'], flights['target'])

    from models import gradient_boosting
    gradient_boosting.run_for_dataset(flights['file_tag'], flights['target'])

print("\n" + "=" * 60)
print("✅ All flights models complete!")
print("=" * 60)
