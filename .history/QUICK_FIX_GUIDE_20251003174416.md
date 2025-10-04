# Quick Fix Guide: LSTM vs XGBoost Fair Comparison

## 🎯 Priority Fixes (Ranked by Impact)

---

## Fix #1: Add Rainfall Features to LSTM (CRITICAL - 90% Impact)

### Current Problem
LSTM only uses 3 water level features, while XGBoost uses 6 features (3 water level + 3 rainfall).

### File to Modify
`notebooks/02_feature_engineering.ipynb` - Cell #3

### Current Code (Line ~41):
```python
# LSTM features - CHỈ WATER LEVEL!
feature_cols_lstm = [
    'Can Tho_Water Level', 'Chau Doc_Water Level', 'Dai Ngai_Water Level'
]
```

### Fixed Code:
```python
# LSTM features - WATER LEVEL + RAINFALL (same as XGBoost)
feature_cols_lstm = [
    'Can Tho_Water Level', 'Chau Doc_Water Level', 'Dai Ngai_Water Level',
    'Can Tho_Rainfall', 'Chau Doc_Rainfall', 'Dai Ngai_Rainfall'
]
```

### After Making This Change:
1. Re-run the entire `02_feature_engineering.ipynb` notebook
2. Re-run `05_train_all_models.ipynb` to retrain LSTM models
3. Compare results - LSTM R² should improve dramatically

**Expected Impact:** LSTM R² should jump from negative/low values (0.0-0.3) to comparable with XGBoost (0.5-0.7).

---

## Fix #2: Sequential Validation Split for LSTM (HIGH - 15% Impact)

### Current Problem
LSTM uses `validation_split=0.2` which randomly splits data, causing temporal leakage.

### File to Modify
`src/lstm_trainer.py` - Method `grid_search()` (around line 130)

### Current Code (Lines ~135-146):
```python
# Train model
history = model.fit(
    self.X_train, self.y_train,
    batch_size=params['batch_size'],
    epochs=epochs,
    validation_split=validation_split,  # ❌ Random split
    callbacks=[early_stopping],
    verbose=verbose
)
```

### Fixed Code:
```python
# Calculate sequential split point (last 20% as validation)
val_samples = int(len(self.X_train) * validation_split)
train_samples = len(self.X_train) - val_samples

# Split sequentially (maintaining temporal order)
X_train_fold = self.X_train[:train_samples]
y_train_fold = self.y_train[:train_samples]
X_val_fold = self.X_train[train_samples:]
y_val_fold = self.y_train[train_samples:]

# Train model with sequential validation
history = model.fit(
    X_train_fold, y_train_fold,
    batch_size=params['batch_size'],
    epochs=epochs,
    validation_data=(X_val_fold, y_val_fold),  # ✅ Sequential validation
    callbacks=[early_stopping],
    verbose=verbose
)
```

### Also Update `train_best_model()` Method (around line 190):

**Current Code:**
```python
self.training_history = self.model.fit(
    self.X_train, self.y_train,
    batch_size=self.best_params['batch_size'],
    epochs=epochs,
    validation_split=validation_split,  # ❌ Random split
    callbacks=[early_stopping],
    verbose=1
)
```

**Fixed Code:**
```python
# Sequential validation split
val_samples = int(len(self.X_train) * validation_split)
train_samples = len(self.X_train) - val_samples

X_train_fold = self.X_train[:train_samples]
y_train_fold = self.y_train[:train_samples]
X_val_fold = self.X_train[train_samples:]
y_val_fold = self.y_train[train_samples:]

self.training_history = self.model.fit(
    X_train_fold, y_train_fold,
    batch_size=self.best_params['batch_size'],
    epochs=epochs,
    validation_data=(X_val_fold, y_val_fold),  # ✅ Sequential validation
    callbacks=[early_stopping],
    verbose=1
)
```

**Expected Impact:** More robust validation, better hyperparameter selection, +5-10% improvement in test metrics.

---

## Fix #3: Add Time-Series Cross-Validation to LSTM (MEDIUM - 10% Impact)

### Current Problem
LSTM only uses a single validation split, while XGBoost uses 3-fold time-series CV.

### File to Modify
`src/lstm_trainer.py` - Method `grid_search()`

### New Implementation:

```python
def grid_search_with_tscv(self, param_grid, n_splits=3, epochs=100, patience=5, verbose=0):
    """
    Perform time-series cross-validation grid search for LSTM
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    print(f"\nStarting time-series CV grid search for {self.config_name}...")
    print(f"Parameter combinations: {len(list(ParameterGrid(param_grid)))}")
    print(f"CV splits: {n_splits}")
    
    input_shape = (self.X_train.shape[1], self.X_train.shape[2])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f"\nTesting combination {i+1}: {params}")
        
        # Store validation scores across folds
        fold_val_losses = []
        
        # Time-series cross-validation
        for fold_num, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            print(f"  Fold {fold_num + 1}/{n_splits}...", end=" ")
            
            set_seeds(self.random_seed + fold_num)  # Different seed per fold
            
            # Split data for this fold
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            try:
                # Create model for this fold
                model = self.create_model(
                    units=params['units'],
                    n_layers=params['n_layers'],
                    dropout=params['dropout'],
                    input_shape=input_shape
                )
                
                # Train
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                )
                
                history = model.fit(
                    X_fold_train, y_fold_train,
                    batch_size=params['batch_size'],
                    epochs=epochs,
                    validation_data=(X_fold_val, y_fold_val),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Get best validation loss
                best_val_loss = min(history.history['val_loss'])
                fold_val_losses.append(best_val_loss)
                print(f"Val Loss: {best_val_loss:.6f}")
                
            except Exception as e:
                print(f"Error: {e}")
                fold_val_losses.append(float('inf'))
        
        # Average validation loss across folds
        avg_val_loss = np.mean([v for v in fold_val_losses if v != float('inf')])
        std_val_loss = np.std([v for v in fold_val_losses if v != float('inf')])
        
        # Store results
        result = convert_to_json_serializable(params.copy())
        result.update({
            'avg_val_loss': float(avg_val_loss),
            'std_val_loss': float(std_val_loss),
            'fold_val_losses': [float(v) for v in fold_val_losses],
            'n_folds': n_splits
        })
        self.grid_search_results.append(result)
        
        print(f"  Average Val Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
        
        # Update best model (train on full training set with best params)
        if avg_val_loss < self.best_score:
            self.best_score = avg_val_loss
            self.best_params = convert_to_json_serializable(params.copy())
            print(f"  >>> New best parameters!")
            
            # Train final model on full training set
            set_seeds(self.random_seed)
            self.model = self.create_model(
                units=params['units'],
                n_layers=params['n_layers'],
                dropout=params['dropout'],
                input_shape=input_shape
            )
            
            # Use last 20% of training data as validation
            val_samples = int(len(self.X_train) * 0.2)
            train_samples = len(self.X_train) - val_samples
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
            
            self.training_history = self.model.fit(
                self.X_train[:train_samples], self.y_train[:train_samples],
                batch_size=params['batch_size'],
                epochs=epochs,
                validation_data=(self.X_train[train_samples:], self.y_train[train_samples:]),
                callbacks=[early_stopping],
                verbose=0
            )
    
    print(f"\nBest parameters: {self.best_params}")
    print(f"Best average validation loss: {self.best_score:.6f}")
    
    return self
```

### Usage in Training Notebook:

Replace the current grid_search call with:
```python
trainer.grid_search_with_tscv(
    param_grid=LSTM_PARAMS,
    n_splits=3,  # Same as XGBoost
    epochs=100,
    patience=10
)
```

**Expected Impact:** More reliable hyperparameter selection, better generalization.

---

## Fix #4: Add Statistical Comparison (MEDIUM - 5% Impact on Confidence)

### File to Create
`src/statistical_comparison.py`

```python
"""
Statistical comparison of model performance
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    def __init__(self, model1_name, model1_predictions, 
                 model2_name, model2_predictions, y_true):
        """
        Compare two models statistically
        
        Args:
            model1_name: Name of first model (e.g., 'XGBoost')
            model1_predictions: Predictions from first model
            model2_name: Name of second model (e.g., 'LSTM')
            model2_predictions: Predictions from second model
            y_true: True target values
        """
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.model1_pred = np.array(model1_predictions)
        self.model2_pred = np.array(model2_predictions)
        self.y_true = np.array(y_true)
        
        # Calculate errors
        self.model1_errors = np.abs(self.y_true - self.model1_pred)
        self.model2_errors = np.abs(self.y_true - self.model2_pred)
        
    def paired_t_test(self):
        """Perform paired t-test on absolute errors"""
        t_stat, p_value = stats.ttest_rel(self.model1_errors, self.model2_errors)
        
        print("\n=== PAIRED T-TEST ===")
        print(f"H0: Mean errors are equal")
        print(f"H1: Mean errors are different")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            if t_stat < 0:
                print(f"✅ {self.model1_name} is significantly BETTER (p < 0.05)")
            else:
                print(f"✅ {self.model2_name} is significantly BETTER (p < 0.05)")
        else:
            print(f"⚠️ No significant difference (p >= 0.05)")
        
        return t_stat, p_value
    
    def wilcoxon_test(self):
        """Perform Wilcoxon signed-rank test (non-parametric)"""
        w_stat, p_value = stats.wilcoxon(self.model1_errors, self.model2_errors)
        
        print("\n=== WILCOXON SIGNED-RANK TEST ===")
        print(f"H0: Median errors are equal")
        print(f"H1: Median errors are different")
        print(f"W-statistic: {w_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            median_diff = np.median(self.model1_errors) - np.median(self.model2_errors)
            if median_diff < 0:
                print(f"✅ {self.model1_name} is significantly BETTER (p < 0.05)")
            else:
                print(f"✅ {self.model2_name} is significantly BETTER (p < 0.05)")
        else:
            print(f"⚠️ No significant difference (p >= 0.05)")
        
        return w_stat, p_value
    
    def error_distribution_comparison(self):
        """Compare error distributions"""
        print("\n=== ERROR DISTRIBUTION COMPARISON ===")
        print(f"{self.model1_name}:")
        print(f"  Mean error: {np.mean(self.model1_errors):.4f}")
        print(f"  Median error: {np.median(self.model1_errors):.4f}")
        print(f"  Std error: {np.std(self.model1_errors):.4f}")
        print(f"  95th percentile: {np.percentile(self.model1_errors, 95):.4f}")
        
        print(f"\n{self.model2_name}:")
        print(f"  Mean error: {np.mean(self.model2_errors):.4f}")
        print(f"  Median error: {np.median(self.model2_errors):.4f}")
        print(f"  Std error: {np.std(self.model2_errors):.4f}")
        print(f"  95th percentile: {np.percentile(self.model2_errors, 95):.4f}")
    
    def plot_comparison(self):
        """Plot error distributions and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error distributions
        axes[0, 0].hist(self.model1_errors, bins=30, alpha=0.6, label=self.model1_name, color='red')
        axes[0, 0].hist(self.model2_errors, bins=30, alpha=0.6, label=self.model2_name, color='blue')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        
        # Box plots
        axes[0, 1].boxplot([self.model1_errors, self.model2_errors],
                          labels=[self.model1_name, self.model2_name])
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Error Box Plot')
        
        # Prediction scatter
        axes[1, 0].scatter(self.y_true, self.model1_pred, alpha=0.5, label=self.model1_name, color='red')
        axes[1, 0].scatter(self.y_true, self.model2_pred, alpha=0.5, label=self.model2_name, color='blue')
        axes[1, 0].plot([self.y_true.min(), self.y_true.max()], 
                       [self.y_true.min(), self.y_true.max()], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('True Values')
        axes[1, 0].set_ylabel('Predictions')
        axes[1, 0].set_title('Prediction Scatter')
        axes[1, 0].legend()
        
        # Paired error comparison
        axes[1, 1].scatter(self.model1_errors, self.model2_errors, alpha=0.5)
        max_error = max(self.model1_errors.max(), self.model2_errors.max())
        axes[1, 1].plot([0, max_error], [0, max_error], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel(f'{self.model1_name} Absolute Error')
        axes[1, 1].set_ylabel(f'{self.model2_name} Absolute Error')
        axes[1, 1].set_title('Paired Error Comparison')
        
        # Add diagonal reference
        # Points below line: model2 has larger error
        # Points above line: model1 has larger error
        
        plt.tight_layout()
        return fig
    
    def full_report(self):
        """Generate full comparison report"""
        print("="*60)
        print(f"STATISTICAL COMPARISON: {self.model1_name} vs {self.model2_name}")
        print("="*60)
        
        self.error_distribution_comparison()
        self.paired_t_test()
        self.wilcoxon_test()
        
        fig = self.plot_comparison()
        plt.show()
        
        return fig

# Example usage:
"""
from statistical_comparison import ModelComparison

# Load predictions
xgb_predictions = joblib.load('models/7n_1n_xgb/predictions.pkl')
lstm_predictions = np.load('models/7n_1n_lstm/predictions.npy')
y_true = pd.read_csv('data/7n_1n_xgb/y_test.csv').iloc[:, 0].values

# Compare
comparison = ModelComparison('XGBoost', xgb_predictions, 'LSTM', lstm_predictions, y_true)
comparison.full_report()
"""
```

---

## Fix #5: Save Predictions for Analysis (LOW - Transparency)

### Modify Both Trainers to Save Predictions

**In `src/xgboost_trainer.py`, add to `save_results()` method:**

```python
# Save predictions (add after line 220)
predictions = {
    'y_train_true': self.y_train.values if hasattr(self.y_train, 'values') else self.y_train,
    'y_train_pred': self.model.predict(self.X_train),
    'y_test_true': self.y_test.values if hasattr(self.y_test, 'values') else self.y_test,
    'y_test_pred': self.model.predict(self.X_test)
}
joblib.dump(predictions, f"{config_folder}/predictions.pkl")
```

**In `src/lstm_trainer.py`, add to `save_results()` method:**

```python
# Save predictions (add after line ~380)
predictions = {
    'y_train_true': self.y_train,
    'y_train_pred': self.model.predict(self.X_train, verbose=0).squeeze(),
    'y_test_true': self.y_test,
    'y_test_pred': self.model.predict(self.X_test, verbose=0).squeeze()
}
np.save(f"{config_folder}/predictions.npy", predictions)
```

---

## 🚦 Implementation Priority

### Priority 1: CRITICAL (Do Today)
- ✅ Fix #1: Add rainfall to LSTM features
- ✅ Fix #2: Sequential validation split

### Priority 2: HIGH (This Week)
- ⬜ Fix #3: Time-series CV for LSTM
- ⬜ Fix #4: Statistical comparison

### Priority 3: MEDIUM (Next Week)
- ⬜ Fix #5: Save predictions
- ⬜ Add experiment tracking (MLflow)
- ⬜ Three-way split (train/val/test)

---

## 📊 Before vs After Comparison Template

After implementing fixes, document results:

```
| Metric | XGBoost | LSTM (Before) | LSTM (After Fix #1) | LSTM (After All Fixes) |
|--------|---------|---------------|---------------------|------------------------|
| Test MAE | 0.XXX | 0.YYY | 0.ZZZ | 0.AAA |
| Test RMSE | 0.XXX | 0.YYY | 0.ZZZ | 0.AAA |
| Test R² | 0.XXX | 0.YYY | 0.ZZZ | 0.AAA |
| Features Used | 6 | 3 | 6 | 6 |
| Validation | TimeSeriesCV | Random Split | Random Split | TimeSeriesCV |
```

---

## 🧪 Testing Checklist

After each fix:
- [ ] Re-run feature engineering notebook
- [ ] Re-train affected models
- [ ] Check for errors in console output
- [ ] Compare metrics before/after
- [ ] Update results spreadsheet
- [ ] Commit changes to git

---

## 📞 Support

If you encounter issues:
1. Check error messages carefully
2. Verify data shapes match expected dimensions
3. Ensure random seeds are set
4. Check if scalers are properly loaded
5. Verify file paths are correct

---

**Last Updated:** October 3, 2025  
**Version:** 1.0
