import sys
import os

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib

import lab3_config as config

step1 = importlib.import_module("1_encoding")
step2 = importlib.import_module("2_imputation")
step3 = importlib.import_module("3_outliers")
step4 = importlib.import_module("4_scaling")
step5 = importlib.import_module("5_balancing")
step6 = importlib.import_module("6_selection")


def run_all():
    print("=" * 60)
    print("STARTING TRAFFIC ACCIDENTS DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1: Encoding
    try:
        step1.run_encoding()
    except Exception as e:
        print(f"!!! Error in Step 1: {e}")
        return

    # Step 2: Imputation
    try:
        step2.run_imputation()
    except Exception as e:
        print(f"!!! Error in Step 2: {e}")
        return

    # Step 3: Outliers
    try:
        step3.run_outliers()
    except Exception as e:
        print(f"!!! Error in Step 3: {e}")
        return

    # Step 4: Scaling
    try:
        step4.run_scaling()
    except Exception as e:
        print(f"!!! Error in Step 4: {e}")
        return

    # Step 5: Balancing
    try:
        step5.run_balancing()
    except Exception as e:
        print(f"!!! Error in Step 5: {e}")
        return

    # Step 6: Feature Selection
    try:
        step6.run_selection()
    except Exception as e:
        print(f"!!! Error in Step 6: {e}")
        return

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final dataset saved to: {config.FILE_SELECTED}")
    print(f"Charts saved to: {config.IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
