"""
Configuration file for water level prediction project
"""

import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 28112001

# Data configuration
STATIONS = ['Can Tho', 'Chau Doc', 'Dai Ngai']
PARAMETERS = ['Rainfall', 'Water Level']

# Experiment configurations
EXPERIMENTS = {
    '7n_1n': {
        'N': 7,
        'M': 1,
        'description': '7 days input to predict water level at day 8 (not mean of days 8-14)'
    },
    '30n_1n': {
        'N': 30,
        'M': 1,
        'description': '30 days input to predict water level at day 31'
    },
    '30n_7n': {
        'N': 30,
        'M': 7,
        'description': '30 days input to predict water level at day 37 (not mean of days 31-37)'
    },
    '30n_30n': {
        'N': 30,
        'M': 30,
        'description': '30 days input to predict water level at day 60'
    },
    '90n_7n': {
        'N': 90,
        'M': 7,
        'description': '90 days input to predict water level at day 97'
    },
    '90n_30n': {
        'N': 90,
        'M': 30,
        'description': '90 days input to predict water level at day 120'
    }
}


# Grid search parameters for XGBoost
XGBOOST_PARAMS = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.8, 1.0]
}


# Grid search parameters for LSTM
# Updated to handle overfitting: higher dropout, lower patience
LSTM_PARAMS = {
    'units': [32, 64],
    'n_layers': [1, 2],
    'dropout': [0.2, 0.5],  # Increased from [0.1, 0.2] to prevent overfitting
    'batch_size': [256],
    'epochs': [100],
    'patience': [5]  # Decreased from [10] to stop earlier when overfitting
}

# Data preprocessing
TRAIN_TEST_SPLIT = 0.8  # 80% for training, 20% for testing

# Target variable (which location to predict)
TARGET_STATION = 'Can Tho'  # Middle stream - Trung nguồn
TARGET_PARAMETER = 'Water Level'

# File paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
NOTEBOOKS_DIR = 'notebooks'
SRC_DIR = 'src'

# Model evaluation metrics
METRICS = ['MAE', 'MSE', 'RMSE', 'R2']

# Early stopping for LSTM
EARLY_STOPPING = True
MONITOR = 'val_loss'
PATIENCE = 5
RESTORE_BEST_WEIGHTS = True