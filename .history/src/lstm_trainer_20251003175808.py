"""
LSTM Training Module for Water Level Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed=28112001):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class LSTMTrainer:
    def __init__(self, config_name, random_seed=28112001):
        self.config_name = config_name
        self.random_seed = random_seed
        self.model = None
        self.best_params = None
        self.best_score = float('inf')
        self.training_history = None
        self.grid_search_results = []
        
        set_seeds(random_seed)
        
    def load_data(self, data_folder):
        """Load training data"""
        folder = f"{data_folder}/{self.config_name}_lstm"
        
        self.X_train = np.load(f"{folder}/X_train.npy")
        self.X_test = np.load(f"{folder}/X_test.npy")
        self.y_train = np.load(f"{folder}/y_train.npy")
        self.y_test = np.load(f"{folder}/y_test.npy")
        
        # Handle different y shapes - FIXED: Don't average multi-step targets
        if len(self.y_train.shape) > 1 and self.y_train.shape[1] > 1:
            print(f"Multi-step target detected: {self.y_train.shape}")
            print(f"WARNING: Using last value instead of averaging for fair comparison")
            # Use the last day of the prediction period instead of averaging
            self.y_train = self.y_train[:, -1]  # Last day
            self.y_test = self.y_test[:, -1]    # Last day
        elif len(self.y_train.shape) > 1:
            self.y_train = self.y_train.squeeze()
            self.y_test = self.y_test.squeeze()
        
        print(f"Loaded data for {self.config_name}:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
        
        return self
    
    def create_model(self, units, n_layers, dropout, input_shape):
        """Create LSTM model"""
        model = Sequential()
        
        # First LSTM layer
        if n_layers == 1:
            model.add(LSTM(units, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, n_layers):
            if i == n_layers - 1:  # Last LSTM layer
                model.add(LSTM(units))
            else:
                model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def grid_search(self, param_grid, validation_split=0.2, epochs=100, patience=5, verbose=0):
        """Perform manual grid search for LSTM"""
        
        print(f"\\nStarting grid search for {self.config_name}...")
        print(f"Parameter combinations: {len(list(ParameterGrid(param_grid)))}")
        
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f"\\nTesting combination {i+1}: {params}")
            
            set_seeds(self.random_seed)  # Reset seeds for each model
            
            try:
                # Create model
                model = self.create_model(
                    units=params['units'],
                    n_layers=params['n_layers'],
                    dropout=params['dropout'],
                    input_shape=input_shape
                )
                
                # Callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                )
                
                # ✅ FIX #2: Sequential validation split (không random)
                # Trước: validation_split=0.2 tự động split RANDOM → temporal leakage
                # Sau: Manual split theo thứ tự thời gian (last 20% as validation)
                val_samples = int(len(self.X_train) * validation_split)
                train_samples = len(self.X_train) - val_samples
                
                X_train_fold = self.X_train[:train_samples]
                y_train_fold = self.y_train[:train_samples]
                X_val_fold = self.X_train[train_samples:]
                y_val_fold = self.y_train[train_samples:]
                
                # Train model with sequential validation
                history = model.fit(
                    X_train_fold, y_train_fold,
                    batch_size=params['batch_size'],
                    epochs=epochs,
                    validation_data=(X_val_fold, y_val_fold),  # Sequential validation
                    callbacks=[early_stopping],
                    verbose=verbose
                )
                
                # Get best validation loss
                best_val_loss = min(history.history['val_loss'])
                
                # Store results
                result = convert_to_json_serializable(params.copy())
                result.update({
                    'best_val_loss': float(best_val_loss),
                    'epochs_trained': int(len(history.history['loss'])),
                    'final_train_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1])
                })
                self.grid_search_results.append(result)
                
                print(f"  Val Loss: {best_val_loss:.6f}, Epochs: {len(history.history['loss'])}")
                
                # Update best model
                if best_val_loss < self.best_score:
                    self.best_score = best_val_loss
                    self.best_params = convert_to_json_serializable(params.copy())
                    self.model = model
                    self.training_history = history
                    print(f"  >>> New best model!")
                
            except Exception as e:
                print(f"  Error: {e}")
                result = convert_to_json_serializable(params.copy())
                result.update({
                    'best_val_loss': float('inf'),
                    'error': str(e)
                })
                self.grid_search_results.append(result)
        
        print(f"\\nBest parameters: {self.best_params}")
        print(f"Best validation loss: {self.best_score:.6f}")
        
        return self
    
    def train_best_model(self, epochs=100, patience=5, validation_split=0.2):
        """Train final model with best parameters"""
        
        print(f"\\nTraining final model with best parameters...")
        
        if self.best_params is None:
            raise ValueError("No best parameters found. Run grid_search first.")
        
        set_seeds(self.random_seed)
        
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        # Create final model
        self.model = self.create_model(
            units=self.best_params['units'],
            n_layers=self.best_params['n_layers'],
            dropout=self.best_params['dropout'],
            input_shape=input_shape
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # ✅ FIX #2: Sequential validation split cho final model
        val_samples = int(len(self.X_train) * validation_split)
        train_samples = len(self.X_train) - val_samples
        
        X_train_final = self.X_train[:train_samples]
        y_train_final = self.y_train[:train_samples]
        X_val_final = self.X_train[train_samples:]
        y_val_final = self.y_train[train_samples:]
        
        # Train with sequential validation
        self.training_history = self.model.fit(
            X_train_final, y_train_final,
            batch_size=self.best_params['batch_size'],
            epochs=epochs,
            validation_data=(X_val_final, y_val_final),  # Sequential validation
            callbacks=[early_stopping],
            verbose=1
        )
        
        print(f"Final model trained successfully!")
        
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train, verbose=0).squeeze()
        y_test_pred = self.model.predict(self.X_test, verbose=0).squeeze()
        
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
    
    def _analyze_grid_results(self):
        """Analyze grid search results for insights"""
        analysis_data = []
        
        for idx, result in enumerate(self.grid_search_results):
            # Extract parameter keys (excluding metric keys)
            param_keys = ['units', 'n_layers', 'dropout', 'batch_size']
            params = {k: result[k] for k in param_keys if k in result}
            params_str = "_".join([f"{k}={v}" for k, v in params.items()])
            
            analysis_data.append({
                'combination_id': idx,
                'param_combination': params_str,
                'val_loss': result.get('best_val_loss', float('inf')),
                'rank': idx + 1,  # Will be re-ranked
                'training_time': result.get('training_time', 0),
                'final_epoch': result.get('epochs_trained', 0),
                'early_stopped': result.get('early_stopped', False),
                **params  # Include individual parameter values
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        # Rank by validation loss (lower is better)
        analysis_df['rank'] = analysis_df['val_loss'].rank()
        analysis_df = analysis_df.sort_values('rank')
        
        return analysis_df
    
    def _calculate_grid_stats(self):
        """Calculate grid search statistics"""
        
        if not self.grid_search_results:
            return {}
        
        val_losses = [r.get('best_val_loss', float('inf')) for r in self.grid_search_results]
        # Filter out infinite values (failed experiments)
        val_losses = [v for v in val_losses if v != float('inf')]
        if not val_losses:
            return {'total_combinations': len(self.grid_search_results), 'successful_combinations': 0}
            
        training_times = [r.get('training_time', 0) for r in self.grid_search_results]
        
        return {
            'total_combinations': len(self.grid_search_results),
            'successful_combinations': len(val_losses),
            'best_val_loss': float(min(val_losses)),
            'worst_val_loss': float(max(val_losses)),
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),
            'val_loss_range': float(max(val_losses) - min(val_losses)),
            'avg_training_time': float(np.mean(training_times)) if training_times else 0,
            'total_training_time': float(sum(training_times)) if training_times else 0
        }
    
    def save_results(self, models_folder):
        """Save model and results"""
        
        # Create folders
        config_folder = f"{models_folder}/{self.config_name}_lstm"
        os.makedirs(config_folder, exist_ok=True)
        
        # Save model
        model_path = f"{config_folder}/best_model.h5"
        self.model.save(model_path)
        
        # Save grid search results
        grid_results_df = pd.DataFrame(self.grid_search_results)
        grid_results_df.to_csv(f"{config_folder}/grid_search_results_full.csv", index=False)
        
        # Save grid search analysis
        grid_analysis = self._analyze_grid_results()
        grid_analysis.to_csv(f"{config_folder}/grid_analysis.csv", index=False)
        
        # Save training history
        if self.training_history:
            history_df = pd.DataFrame(self.training_history.history)
            history_df.to_csv(f"{config_folder}/training_history.csv", index=False)
        
        # Calculate grid search statistics
        grid_stats = self._calculate_grid_stats()
        
        # Save evaluation results
        results = {
            'config_name': self.config_name,
            'model_type': 'LSTM',
            'best_params': convert_to_json_serializable(self.best_params),
            'best_val_loss': float(self.best_score),
            'train_metrics': {k: float(v) for k, v in self.train_metrics.items()},
            'test_metrics': {k: float(v) for k, v in self.test_metrics.items()},
            'grid_summary': grid_stats,
            'data_shapes': {
                'X_train': [int(x) for x in self.X_train.shape],
                'X_test': [int(x) for x in self.X_test.shape],
                'y_train': [int(x) for x in self.y_train.shape],
                'y_test': [int(x) for x in self.y_test.shape]
            },
            'model_params': {
                'total_params': int(self.model.count_params()),
                'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]))
            },
            'grid_search_combinations': int(len(self.grid_search_results)),
            'training_epochs': int(len(self.training_history.history['loss']) if self.training_history else 0),
            'trained_at': datetime.now().isoformat(),
            'random_seed': int(self.random_seed)
        }
        
        with open(f"{config_folder}/results.json", 'w') as f:
            json.dump(convert_to_json_serializable(results), f, indent=2)
        
        print(f"\\nResults saved to {config_folder}/")
        print(f"  - best_model.h5")
        print(f"  - grid_search_results_full.csv (all combinations)")
        print(f"  - grid_analysis.csv (analysis)")
        print(f"  - training_history.csv")
        print(f"  - results.json (summary)")
        
        return self

def train_lstm_model(config_name, param_grid, data_folder="../data", 
                    models_folder="../models", epochs=100, patience=5, 
                    validation_split=0.2, verbose=0):
    """
    Complete training pipeline for LSTM model
    
    Args:
        config_name: Configuration name (e.g., '7n_1n')
        param_grid: Grid search parameters
        data_folder: Folder containing training data
        models_folder: Folder to save models
        epochs: Maximum epochs for training
        patience: Early stopping patience
        validation_split: Validation split ratio
        verbose: Verbosity level
    
    Returns:
        LSTMTrainer instance with trained model
    """
    
    trainer = LSTMTrainer(config_name)
    
    try:
        # Training pipeline
        trainer.load_data(data_folder)
        trainer.grid_search(param_grid, validation_split=validation_split, 
                          epochs=epochs, patience=patience, verbose=verbose)
        trainer.train_best_model(epochs=epochs, patience=patience, 
                               validation_split=validation_split)
        trainer.evaluate()
        trainer.save_results(models_folder)
        
        print(f"\\n✅ LSTM training completed for {config_name}")
        
    except Exception as e:
        print(f"\\n❌ Error training LSTM for {config_name}: {e}")
        raise e
    
    return trainer

if __name__ == "__main__":
    # Example usage
    config_name = "7n_1n"
    
    # Small parameter grid for testing
    test_param_grid = {
        'units': [50, 100],
        'n_layers': [1, 2],
        'dropout': [0.1, 0.2],
        'batch_size': [32, 64]
    }
    
    trainer = train_lstm_model(config_name, test_param_grid, epochs=10, verbose=1)