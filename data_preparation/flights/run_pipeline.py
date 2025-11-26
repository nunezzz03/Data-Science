"""
Pipeline Orchestrator - Runs all data preparation steps sequentially
"""
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib

# Import configuration
import lab3_config as config

# Import step modules dynamically (they start with numbers)
step1 = importlib.import_module("1_encoding")
step2 = importlib.import_module("2_imputation")
step3 = importlib.import_module("3_outliers")
step4 = importlib.import_module("4_scaling")
step5 = importlib.import_module("5_balancing")
step6 = importlib.import_module("6_selection")


def run_all():
    print("\n" + "=" * 80)
    print(" " * 20 + "FLIGHTS DATA PREPARATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Encoding
    try:
        step1.run_encoding()
    except Exception as e:
        print(f"\n!!! ERROR in Step 1 (Encoding): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Imputation
    try:
        step2.run_imputation()
    except Exception as e:
        print(f"\n!!! ERROR in Step 2 (Imputation): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Outliers
    try:
        step3.run_outliers()
    except Exception as e:
        print(f"\n!!! ERROR in Step 3 (Outliers): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Scaling
    try:
        step4.run_scaling()
    except Exception as e:
        print(f"\n!!! ERROR in Step 4 (Scaling): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Balancing
    try:
        step5.run_balancing()
    except Exception as e:
        print(f"\n!!! ERROR in Step 5 (Balancing): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Feature Selection
    try:
        step6.run_selection()
    except Exception as e:
        print(f"\n!!! ERROR in Step 6 (Selection): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n   Final dataset saved to: {config.FILE_SELECTED}")
    print(f"   Comparison charts saved to: {config.IMAGES_DIR}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_all()
