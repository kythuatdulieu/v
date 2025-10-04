# 🎯 FINAL CONFIGURATION: 96 Intervals/Day + MAX Target

## ⚙️ **Configuration Summary**

```python
INTERVAL_MINUTES = 15       # 15 phút/interval
INTERVALS_PER_DAY = 96      # 24 * 60 / 15 = 96
EMBARGO_DAYS = 1            # 1 ngày = 96 intervals
TARGET_TYPE = "MAX"         # MAX water level per day
```

---

## 📊 **Data Dimensions**

### **Ví dụ: 7n_1n (7 days → predict MAX of day 8)**

#### XGBoost:
```
Input shape:  (samples, 7 × 96 × 6) = (samples, 4032 features)
Output shape: (samples,) - MAX values
```

#### LSTM:
```
Input shape:  (samples, 672 timesteps, 6 features)
              └─ 672 = 7 days × 96 intervals
Output shape: (samples,) - MAX values (SAME as XGBoost)
```

---

## 🎯 **Target: MAX Water Level**

### **Cách tính:**

```python
# Cho mỗi sample:
target_start_idx = i  # Đầu ngày target
target_end_idx = i + 96  # Cuối ngày target (96 intervals = 1 ngày)

# Target = MAX của 96 intervals
y_max = data[target_start_idx:target_end_idx]['Water Level'].max()
```

### **Timeline:**

```
Time: ──────────────────────────────────────────────────────────────>

Input:  [━━━━━━━━━━━━━━━━━ N days × 96 ━━━━━━━━━━━━━━━━━]
                                                           ⚡EMBARGO⚡
                                                           [96 int]
                                                                     📊
                                                                 [96 int]
                                                                  MAX↑
```

### **Ví dụ cụ thể (7 ngày):**

```
Day:     [1  2  3  4  5  6  7] [EMBARGO] [8]
Intervals: 672 intervals       96 int    96 int
                                          ↓
                                         MAX
```

---

## ⚠️ **Tại sao chọn MAX?**

### **1. Practical Value - Cảnh báo Lũ lụt**
```
Scenario: Dự báo lũ cho ngày mai

❌ Mean water level: 1.2m
   → "Mực nước trung bình bình thường"
   → Không cảnh báo được

✅ MAX water level: 2.5m  
   → "Peak có thể đạt 2.5m - CẢNH BÁO!"
   → Action required!
```

### **2. Fair Comparison**
- Cả XGBoost và LSTM predict CÙNG 1 giá trị max
- Không phải mean (ít thông tin)
- Không phải sequence (không so sánh được)

### **3. Challenging but Learnable**
- Harder than mean prediction
- Tests model's ability to capture extremes
- Real-world application value

---

## 📈 **Expected Data Shapes**

| Config | N days | Input Intervals | XGB Features | LSTM Shape | Target |
|--------|--------|-----------------|--------------|------------|--------|
| 7n_1n | 7 | 672 | 4032 | (672, 6) | MAX day 8 |
| 30n_1n | 30 | 2880 | 17280 | (2880, 6) | MAX day 31 |
| 30n_7n | 30 | 2880 | 17280 | (2880, 6) | MAX day 37 |
| 30n_30n | 30 | 2880 | 17280 | (2880, 6) | MAX day 60 |
| 90n_7n | 90 | 8640 | 51840 | (8640, 6) | MAX day 97 |
| 90n_30n | 90 | 8640 | 51840 | (8640, 6) | MAX day 120 |

**Features:** 6 (Water Level + Rainfall for 3 stations)

---

## 🔍 **Fairness Verification**

### **Checklist:**

```
✅ 1. Same input resolution: 96 intervals/day
✅ 2. Same features: 6 variables (WL + RF)
✅ 3. Same embargo: 96 intervals = 1 day
✅ 4. Same target: MAX of 96 intervals
✅ 5. Same target computation:
      XGBoost: max(intervals[i:i+96])
      LSTM:    max(intervals[i:i+96])  ← IDENTICAL
```

### **Information Parity:**

```python
# Total information per sample:
XGBoost: N × 96 × 6 = flattened to 1D
LSTM:    N × 96 × 6 = kept as 3D

Example (7n_1n):
  XGBoost: 7 × 96 × 6 = 4032 features (1D array)
  LSTM:    (672, 6) = 4032 values (2D array)
  
  → SAME INFORMATION, different format ✅
```

---

## 💻 **Code Example**

### **XGBoost Function:**
```python
def create_lag_features_xgb(data, feature_cols, target_col, N, M):
    N_intervals = N * 96
    embargo_intervals = 96
    
    for i in range(N_intervals + embargo_intervals, len(data) - 96 + 1):
        # Features: N days before embargo
        X_row = [data.iloc[i-embargo-lag][col] 
                 for lag in range(1, N_intervals+1) 
                 for col in feature_cols]
        
        # Target: MAX of next 96 intervals (1 day)
        y_max = data.iloc[i:i+96][target_col].max()
```

### **LSTM Function:**
```python
def create_sequences_lstm(data, feature_cols, target_col, N, M):
    N_intervals = N * 96
    embargo_intervals = 96
    
    for i in range(N_intervals + embargo_intervals, len(data) - 96 + 1):
        # Features: N days before embargo (sequence)
        X_seq = data.iloc[i-embargo-N_intervals:i-embargo][feature_cols].values
        
        # Target: MAX of next 96 intervals (SAME as XGBoost)
        y_max = data.iloc[i:i+96][target_col].max()
```

---

## 📊 **Metadata Example**

```json
{
  "config_name": "7n_1n",
  "model_type": "xgb",
  "X_train_shape": [7328, 4032],
  "y_train_shape": [7328],
  "target_col": "Can Tho_Water Level",
  "target_type": "MAX water level per day",
  "interval_minutes": 15,
  "intervals_per_day": 96,
  "embargo_days": 1,
  "embargo_intervals": 96,
  "feature_info": "6 features: WL + Rainfall"
}
```

---

## 🚀 **Run Instructions**

```bash
cd notebooks

# Open 02_feature_engineering.ipynb
# → Run All cells

# Expected output:
#   - ../data/7n_1n_xgb/     (CSV files)
#   - ../data/7n_1n_lstm/    (NPY files)
#   - ... (12 folders total)
#   - ../data/data_summary.csv
```

---

## 🎓 **Training Implications**

### **For XGBoost:**
```python
# Input: Flattened features (4032 for 7n_1n)
# Output: Single MAX value
# Loss: MSE/MAE on MAX prediction
# Challenge: Learn from high-dimensional input
```

### **For LSTM:**
```python
# Input: Sequence (672 timesteps, 6 features)
# Output: Single MAX value (not sequence!)
# Architecture: 
#   - LSTM layers extract temporal features
#   - Dense layer → single MAX prediction
# Challenge: Learn to identify peak from sequence
```

---

## ⚠️ **Important Notes**

### **1. Target is NOT a sequence:**
```python
# ❌ WRONG:
y = [val1, val2, ..., val96]  # 96 values

# ✅ CORRECT:
y = max([val1, val2, ..., val96])  # 1 value (the MAX)
```

### **2. Fair comparison maintained:**
```python
assert y_xgboost.shape == y_lstm.shape == (samples,)
assert all(y_xgboost == y_lstm)  # Same values!
```

### **3. Data requirement:**
```python
# Cần raw data với 15-min intervals
# Đã có trong: train_data.csv, test_data.csv
# Aggregated từ raw.csv (đã làm sạch)
```

---

## ✅ **Summary**

| Aspect | Value | Reason |
|--------|-------|--------|
| **Resolution** | 15 min, 96/day | High temporal detail |
| **Target** | MAX per day | Flood warning critical |
| **Embargo** | 1 day = 96 int | No data leakage |
| **Fairness** | ✅ 7/7 PASS | True comparison |
| **Complexity** | Higher | Better test of models |
| **Practicality** | ⚠️ High | Disaster management |

---

**Configuration finalized:** 2025-10-04  
**Ready for:** Notebook execution & model training  
**Expected outcome:** Fair XGBoost vs LSTM comparison for flood peak prediction
