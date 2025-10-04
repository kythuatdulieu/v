# Next Steps to Complete Fix #1

## ✅ Completed Edits

### Cell #3 (Feature Definition)
- ✅ `feature_cols_lstm = feature_cols_xgb` (both use 6 features)

### Cell #7 (Unified Scaling) - **JUST EDITED**
- ✅ Removed separate `scaler_lstm` and `scaler_xgb`
- ✅ Created single `scaler` that fits on training data only
- ✅ Created single `train_scaled` and `test_scaled` DataFrames
- ✅ Both contain all 6 features (water level + rainfall)
- ✅ Saved scaler to `../models/scalers/feature_scaler.pkl` (unified)
- ✅ Added clear warnings about target scaling

### Cell #9 (Function Definitions) - **JUST EDITED**
- ✅ Updated `create_sequences_lstm_daily` docstring
- ✅ Changed "CHỈ WATER LEVEL" → "CÙNG features với XGBoost"
- ✅ Added print showing features being used
- ✅ Clarified both functions use gap forecasting (single value at N+M)

### Cell #11 (Data Creation Loop) - **JUST EDITED**  
- ✅ Changed from `train_scaled_lstm` → `train_scaled`
- ✅ Changed from `test_scaled_lstm` → `test_scaled`
- ✅ Updated all print messages to say "SAME features as XGBoost"
- ✅ Updated metadata message to clarify feature parity

## 🔄 Required Actions

### 1. Re-run Notebook 02 (CRITICAL)
```bash
# The notebook MUST be re-executed to regenerate data files
# Current data files still have 3 features for LSTM
```

**Execution order:**
1. Run Cell #1 (imports)
2. Run Cell #2 (constants and EXPERIMENTS)
3. Run Cell #3 (feature definitions) ← Verify 6 features printed
4. Run Cell #5 (old functions - can skip if not used)
5. Run Cell #7 (unified scaling) ← Verify single scaler created
6. Run Cell #8 (experiments config print)
7. Run Cell #9 (updated functions) ← Verify correct docstrings
8. Run Cell #11 (data creation loop) ← **THIS IS THE CRITICAL ONE**

**Expected output from Cell #11:**
```
Creating LSTM daily sequences: 7 days → predict day 8
Features (6): ['Can Tho_water_level', 'Can Tho_rainfall', 'Chau Doc_water_level', ...]
Generated X sequences with shape (samples, 7, 6)
  └─ samples × 7 timesteps × 6 features
```

**Current (wrong) output:**
```
Generated X sequences with shape (samples, 7, 3)  ← ONLY 3 FEATURES!
```

### 2. Verify Data Files
After re-running, check that data files are correct:

```bash
# For each experiment (7n_1n, 30n_1n, etc.)
python3 << 'EOF'
import numpy as np
import json

configs = ['7n_1n', '30n_1n', '30n_7n', '30n_30n', '90n_7n', '90n_30n']

for config in configs:
    lstm_path = f'../data/{config}_lstm/'
    
    # Load LSTM data
    X_train = np.load(f'{lstm_path}X_train.npy')
    
    # Load metadata
    with open(f'{lstm_path}metadata.json', 'r') as f:
        meta = json.load(f)
    
    # Check shape
    expected_features = 6
    actual_features = X_train.shape[2]
    
    status = "✅" if actual_features == expected_features else "❌"
    print(f"{status} {config}: LSTM shape = {X_train.shape} "
          f"(expected {expected_features} features, got {actual_features})")
    
    # Check metadata
    print(f"   Metadata: {meta['feature_info']}")
    print()
EOF
```

**Expected output:**
```
✅ 7n_1n: LSTM shape = (samples, 7, 6) (expected 6 features, got 6)
   Metadata: SAME AS XGB: 6 variables (WL + Rainfall)

✅ 30n_1n: LSTM shape = (samples, 30, 6) (expected 6 features, got 6)
   Metadata: SAME AS XGB: 6 variables (WL + Rainfall)
...
```

### 3. Re-train LSTM Models
Once data is regenerated with correct features:

```bash
# Open notebook 05_train_all_models.ipynb
# Run all cells to retrain LSTM models with correct 6 features
```

**Expected improvements:**
- Current R²: 0.0 to 0.3 (very poor)
- After fix R²: 0.5 to 0.7 (comparable to XGBoost)

### 4. Compare Models
```bash
# Open notebook 06_model_comparison.ipynb  
# Run all cells to compare XGBoost vs LSTM fairly
```

**Expected results:**
- Both models should have similar performance (±10%)
- LSTM should NO LONGER be handicapped by missing features

## 📊 Verification Checklist

Before re-running:
- [ ] Cell #7 creates single `scaler` (not `scaler_xgb` and `scaler_lstm`)
- [ ] Cell #7 creates single `train_scaled` (not `train_scaled_xgb` and `train_scaled_lstm`)
- [ ] Cell #11 uses `train_scaled` for both XGBoost and LSTM
- [ ] Cell #9 docstring says "CÙNG features với XGBoost"

After re-running Cell #11:
- [ ] LSTM data shape is (samples, N, 6) not (samples, N, 3)
- [ ] Metadata shows "SAME AS XGB: 6 variables"
- [ ] All 6 config folders created successfully

After re-training (notebook 05):
- [ ] LSTM R² improved significantly (from ~0.0-0.3 to 0.5-0.7)
- [ ] LSTM and XGBoost have comparable performance

## 🔍 Key Changes Summary

### Before Fix:
```python
# Cell #7 (OLD - WRONG)
scaler_xgb = StandardScaler()
scaler_lstm = StandardScaler()
train_scaled_xgb = scaler_xgb.fit_transform(train_data[feature_cols_xgb])  # 6 features
train_scaled_lstm = scaler_lstm.fit_transform(train_data[feature_cols_lstm])  # 3 features!

# Cell #11 (OLD - WRONG)
create_sequences_lstm_daily(train_scaled_lstm, ...)  # Only 3 features!
```

### After Fix:
```python
# Cell #7 (NEW - CORRECT)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[feature_cols_xgb])  # 6 features
# Single DataFrame used by BOTH models

# Cell #11 (NEW - CORRECT)
create_sequences_lstm_daily(train_scaled, ...)  # All 6 features!
```

## ⚠️ Critical Notes

1. **Must re-run notebook** - Code changes alone are not enough. The data files must be regenerated.

2. **Check kernel variables** - After re-running, verify:
   ```python
   print(train_scaled.shape)  # Should show all 6 features
   print(feature_cols_lstm)   # Should match feature_cols_xgb
   ```

3. **Sequential execution** - Do NOT run cells out of order. Start from Cell #1 and go sequentially.

4. **Disk space** - Each config creates ~2 folders with numpy arrays. Ensure sufficient space.

5. **Time estimate** - Re-running Cell #11 will take ~30-60 seconds per config (6 configs total).

## 📝 User Instructions

Để hoàn tất fix:

1. **Mở notebook** `02_feature_engineering.ipynb`
2. **Restart kernel** để clear old variables
3. **Run All Cells** từ đầu đến cuối
4. **Verify output** của Cell #11:
   - LSTM shape phải là (samples, N, **6**) không phải (samples, N, 3)
5. **Kiểm tra files** trong `../data/`:
   - Mỗi `*_lstm/metadata.json` phải ghi "SAME AS XGB: 6 variables"
6. **Re-train LSTM** bằng notebook 05
7. **So sánh kết quả** bằng notebook 06

## 🎯 Success Criteria

Fix được coi là hoàn tất khi:
- ✅ LSTM input shape: (samples, N, 6) cho tất cả configs
- ✅ Metadata files confirm 6 features
- ✅ LSTM R² >= 0.5 (comparable to XGBoost)
- ✅ Không còn performance gap lớn giữa LSTM và XGBoost
