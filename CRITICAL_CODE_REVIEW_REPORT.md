# 🔴 CRITICAL CODE REVIEW REPORT
## Water Level Prediction Pipeline Analysis

**Date**: October 4, 2025  
**Reviewer**: GitHub Copilot  
**Status**: ✅ Code flow analyzed against PRD requirements

---

## ✅ EXECUTIVE SUMMARY

### Overall Assessment: **GOOD with Minor Improvements Needed**

The pipeline has been **significantly improved** and addresses most of the critical issues mentioned in your concerns. However, there are still some areas that need attention to fully align with the PRD.

### Key Findings:
- ✅ **FIXED**: Train/test split is now temporal (no shuffle)
- ✅ **FIXED**: Feature scaler is fit only on training data
- ✅ **FIXED**: LSTM now uses same features as XGBoost (including rainfall)
- ✅ **FIXED**: Target scaling for LSTM with proper inverse transform
- ✅ **FIXED**: Sequential validation split for LSTM (no random split)
- ⚠️ **NEEDS REVIEW**: Sequence creation logic for multi-step forecasting
- ⚠️ **NEEDS IMPROVEMENT**: Cross-validation not implemented for final comparison
- ⚠️ **MINOR**: Daily resampling (8 intervals/day) may lose intraday patterns

---

## 📊 DETAILED FLOW ANALYSIS

### 1️⃣ DATA CLEANING & EDA (01_data_cleaning_and_eda.ipynb)

#### ✅ What's Working Well:
```python
# ✅ CORRECT: Temporal split without shuffling
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
split_idx = int(len(df_clean) * 0.8)
train_data = df_clean.iloc[:split_idx].copy()
test_data = df_clean.iloc[split_idx:].copy()
```

**Analysis**: 
- ✅ Data is sorted by datetime before splitting
- ✅ Sequential split: 80% earliest data for training, 20% latest for testing
- ✅ No `shuffle=True` in train_test_split
- ✅ Maintains temporal order as required by PRD

#### ✅ Outlier Detection:
- Time-series aware outlier detection using rolling Z-score
- Rate-of-change analysis for unrealistic jumps
- Appropriate for hydrological data

#### ✅ Missing Data Handling:
- Linear interpolation for small gaps (≤1 hour)
- Polynomial interpolation for larger gaps (≤6 hours)
- Follows best practices for time-series imputation

---

### 2️⃣ FEATURE ENGINEERING (02_feature_engineering.ipynb)

#### ✅ What's Working Well:

**Scaler Fit on Training Data Only** (Lines 206-244):
```python
# ✅ CORRECT: No data leakage
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_data[feature_cols_xgb])
test_features_scaled = scaler.transform(test_data[feature_cols_xgb])
```

**Analysis**:
- ✅ Scaler is fitted ONLY on training data
- ✅ Same scaler is applied to test data (transform only)
- ✅ No future information leaks into preprocessing
- ✅ Aligns with PRD requirement: "Preprocessing must be fitted only on training folds"

**LSTM Feature Parity** (Lines 36-62):
```python
# ✅ FIXED: LSTM now uses same features as XGBoost
feature_cols_lstm = [col for col in train_data.columns 
                     if col not in ['datetime', 'month'] 
                     and 'WL_Change' not in col]
# Both models now use: Water Level + Rainfall (6 features total)
```

**Analysis**:
- ✅ LSTM now includes rainfall features (was missing before)
- ✅ Both models use identical feature sets for fair comparison
- ✅ Addresses your concern: "Thiếu biến dự báo (rainfall)"

#### ⚠️ Area for Review:

**Sequence Creation Functions** (Lines 68-130):
```python
def create_lag_features_xgb(data, feature_cols, target_col, N, M):
    N_intervals = N * 8  # 8 intervals per day
    M_intervals = M * 8
    # ...
    for i in range(N_intervals, len(data_sorted) - M_intervals + 1):
        # Features: N days back
        # Target: Day N+M (if M==1) or mean of M days (if M>1)
```

**Analysis**:
- ⚠️ **8 intervals per day** = 3-hour resolution (reduced from 15-min original data)
- ⚠️ This matches your concern: "Việc nén thời gian quá coarse có thể mất thông tin dao động ngắn hạn"
- ⚠️ For multi-step (M > 1), uses **mean** instead of **sequence** or **last value**

**Daily Resampling Update** (Lines 260-366):
```python
def create_lag_features_xgb_daily(data, feature_cols_xgb, target_col, N, M):
    """
    N: số ngày input
    M: gap để dự đoán (dự đoán ngày thứ N+M)
    Returns: y = 1 SỐ DUY NHẤT tại ngày N+M (KHÔNG phải mean)
    """
```

**Analysis**:
- ✅ Updated to predict **single value** at day N+M (not mean)
- ✅ More consistent with actual forecasting task
- ✅ Matches config descriptions: "predict water level at day 31"

---

### 3️⃣ XGBOOST TRAINING (src/xgboost_trainer.py)

#### ✅ What's Working Well:

**Time Series Cross-Validation** (Lines 42-72):
```python
tscv = TimeSeriesSplit(n_splits=cv_folds)
self.grid_search_cv = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,  # ✅ Time series CV maintains temporal order
    scoring=scoring,
    n_jobs=n_jobs
)
```

**Analysis**:
- ✅ Uses `TimeSeriesSplit` instead of standard KFold
- ✅ Maintains temporal ordering during cross-validation
- ✅ Each fold uses past data for training, future data for validation
- ✅ Prevents temporal leakage
- ✅ Aligns with PRD: "Rolling-origin CV respecting temporal order"

**No Target Scaling** (Lines 118-146):
```python
# XGBoost works directly on original scale
y_train_pred = self.model.predict(self.X_train)
test_metrics = {
    'MAE': mean_absolute_error(self.y_test, y_test_pred),
    'R2': r2_score(self.y_test, y_test_pred)
}
```

**Analysis**:
- ✅ Tree-based models don't require target scaling
- ✅ Predictions are directly in original scale (meters)
- ✅ Metrics are computed on original scale

---

### 4️⃣ LSTM TRAINING (src/lstm_trainer.py)

#### ✅ What's Working Well:

**Target Scaling** (Lines 62-92):
```python
# ✅ CORRECT: Scale target for neural network
self.target_scaler = StandardScaler()
self.y_train = self.target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
self.y_test = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

# Keep original for evaluation
self.y_train_original = y_train_raw
self.y_test_original = y_test_raw
```

**Analysis**:
- ✅ Target is scaled (neural networks need this)
- ✅ Scaler is fit ONLY on training targets
- ✅ Original values are saved for proper evaluation
- ✅ Addresses your concern: "Chuẩn hoá mục tiêu"

**Sequential Validation Split** (Lines 180-200):
```python
# ✅ FIX #2: Sequential validation split (không random)
val_samples = int(len(self.X_train) * validation_split)
train_samples = len(self.X_train) - val_samples

X_train_fold = self.X_train[:train_samples]
y_train_fold = self.y_train[:train_samples]
X_val_fold = self.X_train[train_samples:]  # Last 20% as validation
y_val_fold = self.y_train[train_samples:]
```

**Analysis**:
- ✅ Validation is taken from the END of training data (chronologically later)
- ✅ No random shuffling in validation split
- ✅ Maintains temporal order
- ✅ Addresses your concern: "Phân tách dữ liệu không theo thời gian"

**Inverse Transform for Evaluation** (Lines 290-325):
```python
# ✅ CORRECT: Evaluate on original scale
y_train_pred = self.target_scaler.inverse_transform(
    y_train_pred_scaled.reshape(-1, 1)
).flatten()

train_metrics = {
    'MAE': mean_absolute_error(self.y_train_original, y_train_pred),
    'R2': r2_score(self.y_train_original, y_train_pred)
}
```

**Analysis**:
- ✅ Predictions are transformed back to original scale
- ✅ Metrics computed on original scale (meters), same as XGBoost
- ✅ Fair comparison between models
- ✅ Addresses your concern: "Đánh giá trên dữ liệu đã scale"

**Architecture with Regularization** (Lines 109-150):
```python
model.add(LSTM(
    units, 
    dropout=dropout,
    recurrent_dropout=0.2  # ✅ Prevent overfitting in temporal patterns
))
model.add(Dropout(dropout))
```

**Analysis**:
- ✅ Both input dropout and recurrent dropout
- ✅ Helps prevent overfitting on sequential patterns
- ✅ Config uses dropout ∈ {0.2, 0.5} (increased from {0.1, 0.2})

#### ⚠️ Areas for Improvement:

**Multi-step Target Handling** (Lines 70-78):
```python
if len(y_train_raw.shape) > 1 and y_train_raw.shape[1] > 1:
    print(f"Multi-step target detected: {y_train_raw.shape}")
    print(f"WARNING: Using last value instead of averaging")
    y_train_raw = y_train_raw[:, -1]  # ⚠️ Takes last day
    y_test_raw = y_test_raw[:, -1]
```

**Analysis**:
- ⚠️ For multi-step forecasting (30n_7n, 30n_30n), only the LAST day is used
- ⚠️ This may not match the feature engineering expectation
- ⚠️ Need to verify consistency with `create_sequences_lstm()` function
- ⚠️ Matches your concern: "Sự thiếu nhất quán giữa chức năng tạo dữ liệu và huấn luyện"

**No Cross-Validation** (Grid search uses simple validation split):
```python
# ⚠️ Grid search uses single validation split, not K-fold CV
history = model.fit(
    X_train_fold, y_train_fold,
    validation_data=(X_val_fold, y_val_fold),
    callbacks=[early_stopping]
)
```

**Analysis**:
- ⚠️ LSTM doesn't use K-fold CV like XGBoost
- ⚠️ Uses single train/validation split within training data
- ⚠️ Less robust than K-fold CV for hyperparameter selection
- ⚠️ PRD recommends: "Use k-fold cross-validation within training set"

---

## 🎯 COMPARISON TO YOUR CONCERNS

### Your Concern #1: ❌ "Phân tách dữ liệu không theo thời gian (shuffle=True)"
**Status**: ✅ **FIXED**
- Train/test split is temporal (no shuffle)
- Validation split is also temporal (sequential, not random)
- No `shuffle=True` found anywhere in the pipeline

### Your Concern #2: ❌ "Data Leakage khi Scaling"
**Status**: ✅ **FIXED**
- Feature scaler fit ONLY on training data
- Target scaler (LSTM) fit ONLY on training targets
- Test data is only transformed, never fitted

### Your Concern #3: ❌ "LSTM reshape thành [samples, 1, features]"
**Status**: ✅ **FIXED**
- LSTM input shape is `[samples, N*8, num_features]` where N is lookback days
- For 30n_1n: shape is `[samples, 240, 6]` (30 days × 8 intervals × 6 features)
- Properly uses temporal sequences, not single timestep

### Your Concern #4: ❌ "Thiếu biến dự báo (rainfall)"
**Status**: ✅ **FIXED**
- LSTM now uses same features as XGBoost
- Both models use: 3 water level + 3 rainfall = 6 features
- Comment in code: "LSTM now has access to rainfall data"

### Your Concern #5: ❌ "Chuẩn hoá mục tiêu"
**Status**: ✅ **FIXED**
- LSTM scales target during training (neural network requirement)
- Predictions are inverse-transformed back to original scale
- Metrics computed on original scale for fair comparison

### Your Concern #6: ⚠️ "Kiến trúc và tham số chưa tối ưu"
**Status**: ⚠️ **PARTIALLY ADDRESSED**
- Dropout increased to {0.2, 0.5}
- Recurrent dropout added (0.2)
- Units expanded to {32, 64}
- Early stopping patience: 10 epochs
- ⚠️ Still no K-fold CV for LSTM (only single validation split)

### Your Concern #7: ⚠️ "Xử lý multi-step forecasting"
**Status**: ⚠️ **NEEDS VERIFICATION**
- Updated to use "last value" instead of "mean"
- Need to verify consistency between feature engineering and training
- Check if `create_sequences_lstm()` output matches `lstm_trainer.py` expectation

### Your Concern #8: ⚠️ "Khung thời gian input không hợp lý"
**Status**: ⚠️ **ACKNOWLEDGED BUT NOT FIXED**
- Still uses 8 intervals per day (3-hour resolution)
- Original data has 96 intervals/day (15-minute resolution)
- This coarser resolution may lose intraday patterns
- Trade-off: Reduces feature dimensionality significantly

---

## 🔍 PRD ALIGNMENT CHECK

### Section 5.2: Data Splitting
- ✅ Train/test split maintains chronological order
- ✅ XGBoost uses time-series cross-validation
- ⚠️ LSTM doesn't use K-fold CV (only single validation split)
- ✅ No shuffling in any split
- ✅ Scaler fitted only on training data

### Section 5.4: Feature Engineering
- ✅ Lag features created from past N days
- ✅ Rainfall included for both models
- ✅ Scaling done within training data only
- ⚠️ Seasonal features (sin/cos encoding) not implemented
- ⚠️ Statistical aggregations (mean, std over window) not added

### Section 5.5: Model Training
- ✅ LSTM and XGBoost both implemented
- ✅ Grid search for hyperparameters
- ✅ Early stopping for LSTM
- ✅ Same random seed for reproducibility
- ⚠️ LSTM doesn't use K-fold CV like XGBoost
- ✅ Experiment tracking (metadata saved)

### Section 5.6: Evaluation and Fairness
- ✅ Identical preprocessing for both models
- ✅ Identical train/test splits
- ✅ Metrics on original scale
- ✅ Same evaluation code (MAE, MSE, RMSE, R²)
- ⚠️ No statistical significance testing (paired t-test)
- ⚠️ No cross-validated performance reporting for LSTM

---

## 🚨 REMAINING ISSUES & RECOMMENDATIONS

### Issue #1: ⚠️ LSTM Lacks K-Fold Cross-Validation
**Impact**: Medium  
**Location**: `src/lstm_trainer.py` - `grid_search()` method

**Problem**:
- XGBoost uses `TimeSeriesSplit` with 3 folds
- LSTM uses single train/validation split
- Unfair comparison of generalization performance

**Recommendation**:
```python
# Implement time-series CV for LSTM
from sklearn.model_selection import TimeSeriesSplit

def grid_search_with_cv(self, param_grid, cv_folds=3):
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    for params in ParameterGrid(param_grid):
        cv_scores = []
        for train_idx, val_idx in tscv.split(self.X_train):
            X_train_fold = self.X_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            # Train and evaluate
            # ...
        mean_cv_score = np.mean(cv_scores)
```

### Issue #2: ⚠️ Multi-Step Forecasting Inconsistency
**Impact**: Medium  
**Location**: Feature engineering vs LSTM trainer

**Problem**:
- Feature engineering creates sequences differently for multi-step
- LSTM trainer only uses last value from multi-step sequences
- May not match intended task definition

**Recommendation**:
- **Option A**: Change LSTM output to multi-output (predict full sequence)
- **Option B**: Ensure feature engineering only creates single-value targets
- **Option C**: Document and verify current approach is correct

**Verification Needed**:
```python
# In 02_feature_engineering.ipynb
# Check what create_sequences_lstm actually outputs for 30n_30n
# Then verify lstm_trainer.py handles it correctly
```

### Issue #3: ⚠️ Coarse Temporal Resolution (8 intervals/day)
**Impact**: Low to Medium  
**Location**: Feature engineering functions

**Problem**:
- Original data: 96 intervals/day (15-min resolution)
- Current: 8 intervals/day (3-hour resolution)
- May lose important intraday patterns (e.g., tidal effects)

**Recommendation**:
- **Option A**: Keep current approach (faster, less features)
- **Option B**: Use finer resolution (e.g., 24 or 48 intervals/day)
- **Option C**: Make resolution configurable in config.py

**Trade-offs**:
```python
# Current: 30 days × 8 intervals × 6 features = 1,440 features (XGB) or [240, 6] (LSTM)
# With 24/day: 30 days × 24 intervals × 6 features = 4,320 features (XGB) or [720, 6] (LSTM)
# With 96/day: 30 days × 96 intervals × 6 features = 17,280 features (XGB) or [2880, 6] (LSTM)
```

### Issue #4: ⚠️ Missing Advanced Features
**Impact**: Low  
**Location**: Feature engineering

**PRD Suggests**:
- Seasonal encoding (sin/cos of day-of-year)
- Statistical aggregations (rolling mean, std, min, max)
- Rate of change features
- Cross-station differences

**Current State**:
- Only lag features implemented
- No seasonal encoding
- No rolling statistics

**Recommendation**:
```python
# Add to feature engineering
def add_seasonal_features(df):
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df

def add_rolling_stats(df, window=7):
    for col in water_level_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window).std()
    return df
```

---

## ✅ WHAT'S WORKING WELL

### Strengths of Current Implementation:

1. **✅ Proper Temporal Ordering**
   - No shuffle in train/test split
   - Sequential validation splits
   - Time-series CV for XGBoost

2. **✅ No Data Leakage**
   - Scalers fit only on training data
   - No future information in features
   - Proper isolation of test set

3. **✅ Fair Model Comparison**
   - Same features for both models
   - Same train/test splits
   - Metrics on same scale (original meters)

4. **✅ Good Software Engineering**
   - Modular code (separate trainers)
   - Configurable experiments
   - Metadata tracking
   - Reproducible (fixed random seeds)

5. **✅ LSTM Improvements**
   - Target scaling with inverse transform
   - Regularization (dropout, recurrent dropout)
   - Early stopping
   - Proper sequence shapes

---

## 📋 ACTION ITEMS (Priority Ordered)

### 🔴 HIGH PRIORITY
1. **Implement K-fold CV for LSTM** to match XGBoost evaluation rigor
2. **Verify multi-step forecasting** consistency between feature engineering and training

### 🟡 MEDIUM PRIORITY
3. **Document temporal resolution choice** (8 intervals/day) or make configurable
4. **Add seasonal features** (sin/cos encoding) as suggested by PRD
5. **Implement statistical significance testing** for model comparison

### 🟢 LOW PRIORITY
6. **Add rolling statistics** features (mean, std over windows)
7. **Experiment with finer temporal resolution** (24 or 48 intervals/day)
8. **Add cross-validation visualization** for hyperparameter analysis

---

## 🎓 CONCLUSION

### Summary:

Your pipeline has been **significantly improved** and now follows most best practices:

✅ **Major Fixes Applied**:
- Temporal train/test split (no shuffle)
- No data leakage in scaling
- LSTM uses rainfall features
- Target scaling with proper inverse transform
- Sequential validation splits

⚠️ **Remaining Gaps vs PRD**:
- LSTM doesn't use K-fold CV (XGBoost does)
- Multi-step forecasting needs verification
- Missing advanced features (seasonal encoding, rolling stats)
- Coarse temporal resolution (trade-off for efficiency)

### Your Original Question: "Tôi nghĩ có vấn đề gì đó với LSTM khi so sánh với XGBoost"

**Answer**: The LSTM pipeline is now **much better** than before, but there's still an **unfair advantage for XGBoost**:

1. **XGBoost** uses rigorous 3-fold time-series cross-validation
2. **LSTM** uses only single train/validation split
3. This makes XGBoost's hyperparameter selection more robust
4. LSTM may be underfitting due to less thorough validation

**Recommendation**: Implement time-series K-fold CV for LSTM to match XGBoost's evaluation rigor. This is the most important missing piece for fair comparison.

---

## 📚 REFERENCES

All findings are based on:
- Product Requirements Document (PRD) provided
- Scikit-learn best practices for data leakage prevention
- Time-series forecasting methodology
- Deep learning best practices for sequence models

**Review Status**: ✅ Complete  
**Next Steps**: Address high-priority action items above
