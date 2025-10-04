# ✅ FAIRNESS CHECKLIST - Feature Engineering

## 📋 7 Điểm Kiểm Tra Công Bằng

### ✅ 1. Chia theo thời gian, không random
**Status:** PASS ✅

**Implementation:**
- Train/test split: 80/20
- Không có random shuffle
- Duy trì thứ tự thời gian

**Code location:** `notebooks/01_data_cleaning_and_eda.ipynb`

---

### ✅ 2. Embargo period để tránh chồng lấn train/test
**Status:** PASS ✅

**Implementation:**
- **Embargo:** 1 ngày = 8 intervals
- **Timeline:** `[Features: N days] → [Embargo: 1 day] → [Target at day M]`
- Features kết thúc TRƯỚC embargo
- Target bắt đầu SAU embargo

**Code:**
```python
embargo_intervals = EMBARGO_DAYS * INTERVALS_PER_DAY  # = 1 * 8 = 8 intervals

# XGBoost & LSTM đều bắt đầu từ:
for i in range(N_intervals + embargo_intervals, len(data) - M_intervals + 1):
    # Features: từ (i - embargo - N) đến (i - embargo)
    # Target: tại (i + M - 1)
```

**Lợi ích:**
- Không data leakage
- Mô phỏng real-world deployment (cần 1 ngày để thu thập data)

---

### ✅ 3. Cùng mục tiêu cho XGB và LSTM
**Status:** PASS ✅

**Implementation:**
- **Target type:** Single value (1 số duy nhất)
- **Target location:** Ngày thứ N + EMBARGO + M
- **KHÔNG phải:** Mean, sequence, hoặc bất kỳ aggregation nào

**XGBoost:**
```python
target_idx = i + M_intervals - 1
y_val = data_sorted.iloc[target_idx][target_col]
```

**LSTM:**
```python
target_idx = i + M_intervals - 1  # GIỐNG XGBoost
y_seq = data_sorted.iloc[target_idx][target_col]  # Single value
```

**Verification:**
```python
# y_train_xgb.shape = (samples,)
# y_train_lstm.shape = (samples,)  ← GIỐNG NHAU
```

---

### ✅ 4. Scaler/encoder fit trên train, áp lên test
**Status:** PASS ✅

**Implementation:**
```python
scaler = StandardScaler()

# Fit ONLY on train
train_features_scaled = scaler.fit_transform(train_data[feature_cols_xgb])

# Transform test (NO FIT)
test_features_scaled = scaler.transform(test_data[feature_cols_xgb])
```

**Lưu ý:**
- Target KHÔNG scale tại feature engineering
- LSTM sẽ scale target riêng trong `lstm_trainer.py`
- XGBoost không cần scale target (tree-based)

---

### ✅ 5. Không dùng đặc trưng có yếu tố tương lai
**Status:** PASS ✅

**Removed features:**
- `WL_Change`: Được tính từ future - past → có leak
- `month`: Có thể leak temporal information

**Retained features:**
```python
feature_cols_xgb = [col for col in train_data.columns 
                    if col not in ['datetime', 'month'] 
                    and 'WL_Change' not in col]
```

**Final features:**
- Water Level: Can Tho, Chau Doc, Dai Ngai
- Rainfall: Can Tho, Chau Doc, Dai Ngai
- **Total: 6 features** (công bằng cho cả XGB và LSTM)

---

### ✅ 6. Sửa target_col thành parameter
**Status:** PASS ✅

**Before:**
```python
def save_data(...):
    # target_col là global variable ❌
    metadata = {
        'target_col': target_col  # Lấy từ global scope
    }
```

**After:**
```python
def save_data(..., target_col, ...):  # ✅ Parameter
    metadata = {
        'target_col': target_col  # Lấy từ parameter
    }
```

**Benefits:**
- Không phụ thuộc global state
- Reusable function
- Clear function signature

---

### ✅ 7. Nhất quán đơn vị interval và nhãn cột
**Status:** PASS ✅

**Configuration:**
```python
INTERVAL_HOURS = 3        # 3 giờ/interval
INTERVALS_PER_DAY = 8     # 24 / 3 = 8
EMBARGO_DAYS = 1          # 1 ngày = 8 intervals
```

**Data verification:**
```csv
datetime
2022-03-12 06:00:00  ← 6h
2022-03-12 09:00:00  ← 9h (+ 3 giờ)
2022-03-12 12:00:00  ← 12h (+ 3 giờ)
2022-03-12 15:00:00  ← 15h (+ 3 giờ)
```

**Lag column naming:**
```python
lag_hours = lag * INTERVAL_HOURS  # lag=1 → 3h, lag=2 → 6h, ...
column_name = f"{col}_lag_{lag_hours}h"
```

**Examples:**
- `Can Tho_Water Level_lag_3h` → 3 giờ trước
- `Can Tho_Water Level_lag_6h` → 6 giờ trước
- `Can Tho_Water Level_lag_168h` → 7 ngày trước (7*24=168h)

**Consistency check:**
- ✅ XGBoost: N*8 intervals
- ✅ LSTM: N*8 timesteps
- ✅ Embargo: 1*8 = 8 intervals
- ✅ All labels use correct hour units

---

## 🎯 Expected Information Parity

### Công thức tổng quát:
```
Total information = N days × INTERVALS_PER_DAY × num_features
                  = N × 8 × 6
```

### Cho mỗi experiment:

| Config | N | Total Intervals | Total Features (XGB) | LSTM Shape |
|--------|---|-----------------|---------------------|------------|
| 7n_1n | 7 | 56 | 56 × 6 = 336 | (56, 6) |
| 30n_1n | 30 | 240 | 240 × 6 = 1440 | (240, 6) |
| 30n_7n | 30 | 240 | 240 × 6 = 1440 | (240, 6) |
| 30n_30n | 30 | 240 | 240 × 6 = 1440 | (240, 6) |
| 90n_7n | 90 | 720 | 720 × 6 = 4320 | (720, 6) |
| 90n_30n | 90 | 720 | 720 × 6 = 4320 | (720, 6) |

**✅ XGBoost và LSTM có CÙNG lượng thông tin!**

---

## 📊 Verification Steps

### 1. Kiểm tra target shape
```python
assert y_train_xgb.shape == y_train_lstm.shape
assert len(y_train_xgb.shape) == 1  # 1D array
```

### 2. Kiểm tra feature count
```python
# XGBoost
assert X_train_xgb.shape[1] == N * INTERVALS_PER_DAY * num_features

# LSTM
assert X_train_lstm.shape[1] == N * INTERVALS_PER_DAY
assert X_train_lstm.shape[2] == num_features
```

### 3. Kiểm tra embargo
```python
# Datetime của target phải cách datetime cuối của features >= 1 ngày
assert (target_datetime - feature_end_datetime).days >= EMBARGO_DAYS
```

### 4. Kiểm tra no data leakage
```python
# Test set không overlap với train set
assert train_data['datetime'].max() < test_data['datetime'].min()
```

---

## 🚀 Run Notebook

Để tạo lại toàn bộ data:

```bash
cd notebooks
# Run notebook 02_feature_engineering.ipynb
# → Nhấn "Run All" là xong!
```

**Output:**
- `../data/{config}_xgb/` - XGBoost data (CSV)
- `../data/{config}_lstm/` - LSTM data (NPY)
- `../data/data_summary.csv` - Tóng kết
- `../models/scalers/feature_scaler.pkl` - Scaler

---

## ✅ Final Checklist Summary

| # | Criterion | Status | Impact |
|---|-----------|--------|---------|
| 1 | Time-based split | ✅ PASS | No future leak |
| 2 | Embargo period | ✅ PASS | No overlap |
| 3 | Same target | ✅ PASS | Fair comparison |
| 4 | Scaler on train | ✅ PASS | No test leak |
| 5 | No future features | ✅ PASS | Valid features |
| 6 | target_col param | ✅ PASS | Clean code |
| 7 | Interval consistency | ✅ PASS | Correct labels |

**Overall Score: 7/7 PASS** ✅

---

## 📝 Notes

### Về interval size:
- Data hiện tại: **3-hour intervals** (6h, 9h, 12h, ...)
- Nếu data là 15-min intervals:
  - Change `INTERVAL_HOURS = 0.25`
  - Change `INTERVALS_PER_DAY = 96`
  - Update lag labels accordingly

### Về target:
- Hiện tại: **Single value** tại ngày N+EMBARGO+M
- Đây là cách CÔNG BẰNG nhất để so sánh XGB vs LSTM
- Nếu cần predict sequence, phải training riêng (không so sánh trực tiếp được)

### Về features:
- XGBoost: Flattened lag features (tabular)
- LSTM: Sequences (3D tensor)
- **Cùng thông tin**, chỉ khác format

---

**Last updated:** 2025-10-04  
**Verified by:** Feature Engineering Pipeline v2.0
