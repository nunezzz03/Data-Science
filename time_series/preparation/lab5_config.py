TRAIN_TEST_SPLIT = 0.70
LAG = 1
MIN_RECORDS_AFTER_DIFF = 20

AGGREGATION_CONFIGS = [
    {'name': 'Daily_Mean', 'gran_level': 'D', 'agg_func': 'mean'},
    {'name': 'Weekly_Mean', 'gran_level': 'W', 'agg_func': 'mean'},
    {'name': 'Weekly_Median', 'gran_level': 'W', 'agg_func': 'median'},
]

DIFFERENTIATION_CONFIGS = [
    {
        'name': 'FirstOrder', 
        'order': 1, 
        'seasonal_period': None,
        'description': 'First-order differencing (removes trend)'
    },
    {
        'name': 'SecondOrder', 
        'order': 2, 
        'seasonal_period': None,
        'description': 'Second-order differencing (removes quadratic trend)'
    },
    {
        'name': 'Seasonal_Weekly', 
        'order': 1, 
        'seasonal_period': 7,
        'description': 'First-order + seasonal differencing (period=7 for weekly patterns)'
    },
]

DATASETS = [
    {
        'name': 'economic',
        'path': 'data/raw/economic_indicators_dataset_2010_2023.csv',
        'date_column': 'Date',
        'target_column': 'Inflation Rate (%)'
    },
    {
        'name': 'traffic',
        'path': 'data/raw/TrafficTwoMonth.csv',
        'date_column': 'datetime',
        'target_column': 'Total'
    }
]
