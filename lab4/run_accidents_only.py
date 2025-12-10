"""
Run all models for Traffic Accidents dataset only
"""
import os
import sys

# Add lab4 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🎓 LAB 4 - TRAFFIC ACCIDENTS ONLY")
print("=" * 60)

import lab4_config as config

for accidents in [config.DATASETS[0], config.DATASETS[2]]: # Accidents Raw and Accidents Lab3
    print(f"\nRunning all models for: {accidents['name']}")
    print(f"Target: {accidents['target']}")
    print("=" * 60)

    from models import naive_bayes
    naive_bayes.run_for_dataset(accidents['file_tag'], accidents['target'])

    from models import logistic_regression
    logistic_regression.run_for_dataset(accidents['file_tag'], accidents['target'])

    from models import knn
    knn.run_for_dataset(accidents['file_tag'], accidents['target'])

    from models import decision_tree
    decision_tree.run_for_dataset(accidents['file_tag'], accidents['target'])

    from models import mlp
    mlp.run_for_dataset(accidents['file_tag'], accidents['target'])

    from models import random_forest
    random_forest.run_for_dataset(accidents['file_tag'], accidents['target'])

    from models import gradient_boosting
    gradient_boosting.run_for_dataset(accidents['file_tag'], accidents['target'])

print("\n" + "=" * 60)
print("✅ All traffic accidents models complete!")
print("=" * 60)
