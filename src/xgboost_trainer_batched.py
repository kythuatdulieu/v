"""
XGBoost Training Module with BATCHED Grid Search for Memory Efficiency
Train parameters in small batches to avoid OOM
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
from itertools import product
import warnings

warnings.filterwarnings('ignore')

class XGBoostTrainerBatched:
    def __init__(self, config_name, random_seed=28112001):
        self.config_name = config_name
        self.random_seed = random_seed
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None
        self.all_results = []
        
    def load_data(self, data_folder):
        """Load training data from .npz files"""
        folder = f"{data_folder}/{self.config_name}_xgb"
        
        # Load from .npz compressed format
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
    
    def create_batches(self, param_grid, batch_size=8):
        """Split parameter grid into smaller batches"""
        # Generate all combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combinations = list(product(*values))
        
        # Split into batches
        batches = []
        for i in range(0, len(all_combinations), batch_size):
            batch_combinations = all_combinations[i:i+batch_size]
            # Convert back to param_grid format
            batch_grid = {}
            for j, key in enumerate(keys):
                batch_grid[key] = list(set([combo[j] for combo in batch_combinations]))
            batches.append(batch_grid)
        
        print(f"\n📦 Created {len(batches)} batches from {len(all_combinations)} total combinations")
        print(f"   Batch size: ~{batch_size} combinations each")
        
        return batches
    
    def grid_search_batched(self, param_grid, cv_folds=3, scoring='neg_mean_squared_error', 
                           batch_size=8, verbose=1):
        """Perform batched grid search to save memory"""
        
        print(f"\n🔍 Starting BATCHED grid search for {self.config_name}...")
        print(f"Full parameter grid: {param_grid}")
        
        # Calculate total combinations
        total_combinations = len(list(product(*param_grid.values())))
        print(f"Total combinations: {total_combinations}")
        print(f"Training in batches of ~{batch_size} to save memory")
        
        # Create batches
        batches = self.create_batches(param_grid, batch_size)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        best_score = float('-inf')
        best_params = None
        
        # Train each batch
        for batch_idx, batch_grid in enumerate(batches, 1):
            print(f"\n{'='*60}")
            print(f"📦 BATCH {batch_idx}/{len(batches)}")
            print(f"{'='*60}")
            print(f"Parameters in this batch: {batch_grid}")
            
            batch_combinations = len(list(product(*batch_grid.values())))
            print(f"Combinations in batch: {batch_combinations}")
            print(f"Fits for this batch: {batch_combinations * cv_folds}")
            
            # XGBoost with memory optimization
            xgb_model = xgb.XGBRegressor(
                random_state=self.random_seed,
                n_jobs=1,
                tree_method='hist',
                max_bin=256
            )
            
            # Grid search for this batch
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=batch_grid,
                cv=tscv,
                scoring=scoring,
                n_jobs=2,  # Limited parallelism
                verbose=verbose,
                return_train_score=True
            )
            
            print(f"🚀 Training batch {batch_idx}...")
            grid_search.fit(self.X_train, self.y_train)
            
            # Store results from this batch
            batch_results = pd.DataFrame(grid_search.cv_results_)
            self.all_results.append(batch_results)
            
            # Check if this batch has better results
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_params = grid_search.best_params_
                print(f"✨ New best score: {best_score:.6f}")
                print(f"   Best params: {best_params}")
            
            print(f"✅ Batch {batch_idx} completed")
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
        
        # Store final results
        self.best_params = best_params
        self.cv_results = pd.concat(self.all_results, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"🎯 BATCHED GRID SEARCH COMPLETED")
        print(f"{'='*60}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {best_score:.6f}")
        
        return self
    
    def train_best_model(self):
        """Train final model with best parameters"""
        
        print(f"\n🏋️ Training final model with best parameters...")
        
        self.model = xgb.XGBRegressor(
            **self.best_params,
            random_state=self.random_seed,
            tree_method='hist',
            max_bin=256
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature_index': range(self.X_train.shape[1]),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Model trained successfully")
        
        return self
    
    def evaluate(self):
        """Evaluate model on train and test sets"""
        
        print(f"\n📊 Evaluating model...")
        
        # Predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Metrics
        train_metrics = {
            'MAE': mean_absolute_error(self.y_train, train_pred),
            'MSE': mean_squared_error(self.y_train, train_pred),
            'RMSE': mean_squared_error(self.y_train, train_pred, squared=False),
            'R2': r2_score(self.y_train, train_pred)
        }
        
        test_metrics = {
            'MAE': mean_absolute_error(self.y_test, test_pred),
            'MSE': mean_squared_error(self.y_test, test_pred),
            'RMSE': mean_squared_error(self.y_test, test_pred, squared=False),
            'R2': r2_score(self.y_test, test_pred)
        }
        
        print(f"\n📈 Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"   {metric}: {value:.6f}")
            
        print(f"\n📉 Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"   {metric}: {value:.6f}")
        
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        
        return self
    
    def save_model(self, models_folder):
        """Save model and results"""
        
        folder = f"{models_folder}/{self.config_name}_xgb"
        os.makedirs(folder, exist_ok=True)
        
        # Save model
        model_file = f"{folder}/best_model.pkl"
        joblib.dump(self.model, model_file)
        print(f"\n💾 Model saved to {model_file}")
        
        # Save results
        results = {
            'config_name': self.config_name,
            'best_params': self.best_params,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = f"{folder}/results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {results_file}")
        
        # Save CV results
        cv_file = f"{folder}/cv_results.csv"
        self.cv_results.to_csv(cv_file, index=False)
        print(f"💾 CV results saved to {cv_file}")
        
        # Save feature importance
        fi_file = f"{folder}/feature_importance.csv"
        self.feature_importance.to_csv(fi_file, index=False)
        print(f"💾 Feature importance saved to {fi_file}")
        
        return self


def train_xgboost_model_batched(config_name, param_grid, data_folder='../data', 
                                models_folder='../models', cv_folds=3, batch_size=8):
    """
    Train XGBoost model with BATCHED grid search
    
    Args:
        config_name: Experiment configuration name
        param_grid: Parameter grid for grid search
        data_folder: Folder containing data
        models_folder: Folder to save models
        cv_folds: Number of cross-validation folds
        batch_size: Number of parameter combinations per batch
    
    Returns:
        Trained XGBoostTrainerBatched instance
    """
    
    trainer = XGBoostTrainerBatched(config_name)
    
    trainer.load_data(data_folder)
    trainer.grid_search_batched(param_grid, cv_folds=cv_folds, batch_size=batch_size)
    trainer.train_best_model()
    trainer.evaluate()
    trainer.save_model(models_folder)
    
    return trainer
