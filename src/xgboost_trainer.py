"""
XGBoost Training Module for Water Level Prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class XGBoostTrainer:
    def __init__(self, config_name, random_seed=28112001):
        self.config_name = config_name
        self.random_seed = random_seed
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None
        
    def load_data(self, data_folder):
        """Load training data from .npz files"""
        folder = f"{data_folder}/{self.config_name}_xgb"
        
        # Load from .npz compressed format (optimized in notebook 02)
        X_train_npz = np.load(f"{folder}/train_X.npz")
        X_test_npz = np.load(f"{folder}/test_X.npz")
        y_train_npz = np.load(f"{folder}/train_y.npz")
        y_test_npz = np.load(f"{folder}/test_y.npz")
        
        # Extract arrays using correct key names
        self.X_train = X_train_npz['X_train']
        self.X_test = X_test_npz['X_test']
        self.y_train = y_train_npz['y_train']
        self.y_test = y_test_npz['y_test']
        
        print(f"Loaded data for {self.config_name}:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
        
        return self
    
    def grid_search(self, param_grid, cv_folds=3, scoring='neg_mean_squared_error', 
                    n_jobs=-1, verbose=1):
        """Perform grid search with time series cross validation"""
        
        print(f"\nStarting grid search for {self.config_name}...")
        print(f"Parameter grid: {param_grid}")
        print(f"CV folds: {cv_folds}")
        
        # Calculate total fits
        from itertools import product
        total_combinations = len(list(product(*param_grid.values())))
        total_fits = total_combinations * cv_folds
        print(f"Total combinations: {total_combinations}")
        print(f"Total fits (with {cv_folds} folds): {total_fits}")
        
        # Time series split to maintain temporal order
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # XGBoost regressor with memory optimization
        xgb_model = xgb.XGBRegressor(
            random_state=self.random_seed,
            n_jobs=1,  # Set to 1 to avoid conflicts with GridSearchCV n_jobs
            tree_method='hist',  # Memory-efficient histogram method
            max_bin=256  # Reduce memory by limiting histogram bins
        )
        
        # Grid search with reduced parallelism to save memory
        # Use n_jobs=2 instead of -1 to avoid spawning too many processes
        actual_n_jobs = min(2, n_jobs) if n_jobs == -1 else n_jobs
        print(f"Using n_jobs={actual_n_jobs} (reduced from {n_jobs} to save memory)")
        
        self.grid_search_cv = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=actual_n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        # Fit
        print(f"Starting GridSearchCV fit...")
        self.grid_search_cv.fit(self.X_train, self.y_train)
        
        # Store results
        self.best_params = self.grid_search_cv.best_params_
        self.cv_results = pd.DataFrame(self.grid_search_cv.cv_results_)
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best CV score: {self.grid_search_cv.best_score_:.6f}")
        
        return self
    
    def train_best_model(self):
        """Train final model with best parameters"""
        
        print(f"\\nTraining final model with best parameters...")
        
        self.model = xgb.XGBRegressor(
            **self.best_params,
            random_state=self.random_seed
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Model trained successfully!")
        print(f"Top 10 most important features:")
        print(self.feature_importance.head(10))
        
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Metrics
        train_metrics = {
            'MAE': mean_absolute_error(self.y_train, y_train_pred),
            'MSE': mean_squared_error(self.y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'R2': r2_score(self.y_train, y_train_pred)
        }
        
        test_metrics = {
            'MAE': mean_absolute_error(self.y_test, y_test_pred),
            'MSE': mean_squared_error(self.y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'R2': r2_score(self.y_test, y_test_pred)
        }
        
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        
        print(f"\\n=== MODEL EVALUATION ===")
        print(f"Training metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        print(f"\\nTest metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return self
    
    def _analyze_cv_results(self):
        """Analyze CV results in detail"""
        
        if self.cv_results is None:
            return pd.DataFrame()
        
        # Create analysis dataframe
        analysis_data = []
        
        for idx, row in self.cv_results.iterrows():
            params_str = "_".join([f"{k}={v}" for k, v in row['params'].items()])
            
            analysis_data.append({
                'param_combination': params_str,
                'rank': row['rank_test_score'],
                'mean_test_score': row['mean_test_score'],
                'std_test_score': row['std_test_score'],
                'mean_train_score': row['mean_train_score'],
                'std_train_score': row['std_train_score'],
                'mean_fit_time': row['mean_fit_time'],
                'std_fit_time': row['std_fit_time'],
                'mean_score_time': row['mean_score_time'],
                'std_score_time': row['std_score_time'],
                'overfit_score': row['mean_train_score'] - row['mean_test_score'],
                **row['params']  # Include individual parameter values
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df = analysis_df.sort_values('rank')
        
        return analysis_df
    
    def save_results(self, models_folder):
        """Save model and results"""
        
        # Create folders
        config_folder = f"{models_folder}/{self.config_name}_xgb"
        os.makedirs(config_folder, exist_ok=True)
        
        # Save model
        model_path = f"{config_folder}/best_model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save feature importance
        self.feature_importance.to_csv(f"{config_folder}/feature_importance.csv", index=False)
        
        # Save full CV results with all metrics
        self.cv_results.to_csv(f"{config_folder}/cv_results_full.csv", index=False)
        
        # Save detailed CV results analysis
        cv_analysis = self._analyze_cv_results()
        cv_analysis.to_csv(f"{config_folder}/cv_analysis.csv", index=False)
        
        # Save evaluation results
        results = {
            'config_name': self.config_name,
            'model_type': 'XGBoost',
            'best_params': self.best_params,
            'best_cv_score': float(self.grid_search_cv.best_score_),
            'best_cv_std': float(self.cv_results.loc[self.grid_search_cv.best_index_, 'std_test_score']),
            'total_cv_combinations': len(self.cv_results),
            'train_metrics': {k: float(v) for k, v in self.train_metrics.items()},
            'test_metrics': {k: float(v) for k, v in self.test_metrics.items()},
            'cv_summary': {
                'best_score': float(self.grid_search_cv.best_score_),
                'best_std': float(self.cv_results.loc[self.grid_search_cv.best_index_, 'std_test_score']),
                'worst_score': float(self.cv_results['mean_test_score'].min()),
                'score_range': float(self.cv_results['mean_test_score'].max() - self.cv_results['mean_test_score'].min()),
                'avg_fit_time': float(self.cv_results['mean_fit_time'].mean()),
                'avg_score_time': float(self.cv_results['mean_score_time'].mean())
            },
            'data_shapes': {
                'X_train': list(self.X_train.shape),
                'X_test': list(self.X_test.shape),
                'y_train': list(self.y_train.shape),
                'y_test': list(self.y_test.shape)
            },
            'feature_count': len(self.X_train.columns),
            'top_10_features': self.feature_importance.head(10).to_dict('records'),
            'trained_at': datetime.now().isoformat(),
            'random_seed': self.random_seed
        }
        
        with open(f"{config_folder}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\nResults saved to {config_folder}/")
        print(f"  - best_model.pkl")
        print(f"  - feature_importance.csv")
        print(f"  - cv_results_full.csv (all CV combinations)")
        print(f"  - cv_analysis.csv (CV analysis)")
        print(f"  - results.json (summary)")
        
        return self

def train_xgboost_model(config_name, param_grid, data_folder="../data", 
                       models_folder="../models", cv_folds=3):
    """
    Complete training pipeline for XGBoost model
    
    Args:
        config_name: Configuration name (e.g., '7n_1n')
        param_grid: Grid search parameters
        data_folder: Folder containing training data
        models_folder: Folder to save models
        cv_folds: Number of CV folds
    
    Returns:
        XGBoostTrainer instance with trained model
    """
    
    trainer = XGBoostTrainer(config_name)
    
    try:
        # Training pipeline
        trainer.load_data(data_folder)
        trainer.grid_search(param_grid, cv_folds=cv_folds)
        trainer.train_best_model()
        trainer.evaluate()
        trainer.save_results(models_folder)
        
        print(f"\\n✅ XGBoost training completed for {config_name}")
        
    except Exception as e:
        print(f"\\n❌ Error training XGBoost for {config_name}: {e}")
        raise e
    
    return trainer

if __name__ == "__main__":
    # Example usage
    from config import XGBOOST_PARAMS
    
    config_name = "7n_1n"
    
    # Small parameter grid for testing
    test_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    trainer = train_xgboost_model(config_name, test_param_grid)