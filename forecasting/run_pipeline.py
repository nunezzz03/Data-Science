import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from lab5_config import DATASETS

def run_aggregation(dataset_path, date_column, target_column, dataset_name):
    print("\n" + "="*80)
    print("STAGE 1: AGGREGATION")
    print("="*80)
    
    from data_preparation.aggregation import run_aggregation_study
    run_aggregation_study(dataset_path, date_column, target_column, dataset_name)


def run_smoothing(dataset_name):
    pass


def run_scaling(dataset_name):
    pass


def run_differentiation(aggregated_dataset_path, dataset_name):
    print("\n" + "="*80)
    print("STAGE 4: DIFFERENTIATION")
    print("="*80)
    
    from data_preparation.differentiation import run_differentiation_study
    run_differentiation_study(aggregated_dataset_path, dataset_name)


def run_full_pipeline(dataset_path, date_column, target_column, dataset_name):
    print("\n" + "="*80)
    print(f"TIME SERIES PREPARATION PIPELINE: {dataset_name.upper()}")
    print("="*80)
    
    run_aggregation(dataset_path, date_column, target_column, dataset_name)
    run_smoothing(dataset_name)
    run_scaling(dataset_name)
    
    aggregated_path = os.path.join(os.path.dirname(__file__), f"data_preparation/{dataset_name}/processed_data/aggregation/{dataset_name}_aggregated.csv")
    run_differentiation(aggregated_path, dataset_name)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    for dataset in DATASETS:
        run_full_pipeline(
            dataset_path=dataset['path'],
            date_column=dataset['date_column'],
            target_column=dataset['target_column'],
            dataset_name=dataset['name']
        )
