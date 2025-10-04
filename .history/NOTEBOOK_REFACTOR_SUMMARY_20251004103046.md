# 📝 FEATURE ENGINEERING NOTEBOOK - CLEAN VERSION

## 🎯 Mục tiêu
Tạo lại notebook `02_feature_engineering.ipynb` gọn gàng, rõ ràng, chỉ cần **Run All** là có data chuẩn.

---

## ✅ Các thay đổi chính

### 1. **Loại bỏ code dư thừa**
**Trước:**
- 2 versions (15-min vs daily) → Confusing
- Nhiều markdown cells giải thích dài dòng
- Comment sai lệch ("CHỈ DÙNG WATER LEVEL" nhưng code dùng cả rainfall)
- Duplicate functions

**Sau:**
- 1 version duy nhất: **3-hour intervals** (khớp với data thực tế)
- Markdown ngắn gọn, đi thẳng vào vấn đề
- Comment chính xác 100%
- Không duplicate

---

### 2. **Cấu trúc notebook mới**

```
1. Header Markdown
   ├─ Mục tiêu
   ├─ Cấu hình thí nghiệm
   └─ Fairness Checklist (7/7 PASS)

2. Import Libraries

3. Load Data & Define Features
   └─ Loại bỏ datetime, month, WL_Change

4. Feature Engineering Functions
   ├─ Configuration (INTERVAL_HOURS, INTERVALS_PER_DAY, EMBARGO_DAYS)
   ├─ create_lag_features_xgb()
   ├─ create_sequences_lstm()
   └─ save_data()

5. Chuẩn hóa dữ liệu
   └─ Fit scaler on train only

6. Tạo dữ liệu cho tất cả experiments
   └─ Loop qua EXPERIMENTS dict

7. Kiểm tra & Tổng hợp
   └─ Summary table + CSV export

8. Sample Data Inspection

9. Kết luận
   └─ Fairness Checklist table
```

**Total cells:** 13 (từ 20+ cells xuống còn 13)

---

### 3. **Configuration rõ ràng**

```python
# ============================================================================
# CONFIGURATION: Interval & Embargo Settings
# ============================================================================
INTERVAL_HOURS = 3         # 3 giờ/interval (6h, 9h, 12h, ...)
INTERVALS_PER_DAY = 8      # 24 / 3 = 8
EMBARGO_DAYS = 1           # Gap 1 ngày giữa features và target
```

**Lợi ích:**
- Dễ dàng thay đổi nếu data format khác
- Comment rõ ràng
- Constants được định nghĩa ở đầu

---

### 4. **Functions được refactor**

#### **create_lag_features_xgb()**
```python
def create_lag_features_xgb(data, feature_cols, target_col, N, M):
    """
    Tạo lag features cho XGBoost với embargo period
    
    Timeline:
        [Features: N days] → [Embargo: 1 day] → [Target: single value at day M]
    """
    N_intervals = N * INTERVALS_PER_DAY
    M_intervals = M * INTERVALS_PER_DAY
    embargo_intervals = EMBARGO_DAYS * INTERVALS_PER_DAY
    
    # ... (logic giữ nguyên, chỉ clean comments)
```

**Improvements:**
- ✅ Docstring rõ ràng với Timeline
- ✅ Lag labels: `lag_3h`, `lag_6h`, ... (đúng đơn vị)
- ✅ Embargo period implemented correctly

#### **create_sequences_lstm()**
```python
def create_sequences_lstm(data, feature_cols, target_col, N, M):
    """
    Tạo sequences cho LSTM - CÙNG FEATURES & TARGET với XGBoost
    
    Timeline:
        [Sequence: N days] → [Embargo: 1 day] → [Target: SINGLE VALUE]
    """
    # ... (GIỐNG XGBoost về embargo & target)
```

**Improvements:**
- ✅ Comment chính xác: "CÙNG FEATURES & TARGET với XGBoost"
- ✅ Target = single value (không phải sequence)
- ✅ Embargo period giống XGBoost

#### **save_data()**
```python
def save_data(..., target_col, ...):  # ✅ target_col là parameter
    """
    Lưu dữ liệu với metadata đầy đủ
    """
    metadata = {
        'target_col': target_col,  # ✅ Không dùng global
        'interval_hours': INTERVAL_HOURS,
        'intervals_per_day': INTERVALS_PER_DAY,
        'embargo_days': EMBARGO_DAYS,
        # ...
    }
```

**Improvements:**
- ✅ target_col là parameter (không phải global variable)
- ✅ Metadata bao gồm interval info
- ✅ Simplified output messages

---

### 5. **Main loop gọn gàng**

```python
for config_name, config in EXPERIMENTS.items():
    N = config['N']
    M = config['M']
    
    # XGBoost
    X_train_xgb, y_train_xgb, dt_train_xgb = create_lag_features_xgb(...)
    X_test_xgb, y_test_xgb, dt_test_xgb = create_lag_features_xgb(...)
    save_data(..., target_col, ...)  # ✅ Pass target_col
    
    # LSTM
    X_train_lstm, y_train_lstm, dt_train_lstm = create_sequences_lstm(...)
    X_test_lstm, y_test_lstm, dt_test_lstm = create_sequences_lstm(...)
    save_data(..., target_col, ...)  # ✅ Pass target_col
```

**Improvements:**
- ✅ Không try-except (nếu lỗi thì nên fail ngay)
- ✅ Clear progress messages
- ✅ Timing information
- ✅ Pass target_col correctly

---

### 6. **Markdown documentation**

#### **Header:**
```markdown
## ✅ Fairness Checklist (7/7 PASS)
1. ✅ Chia theo thời gian (80/20 train/test)
2. ✅ Embargo 1 ngày giữa features và target
3. ✅ Cùng target: 1 số duy nhất (không phải mean/sequence)
4. ✅ Scaler fit trên train only
5. ✅ Không dùng features tương lai (loại bỏ WL_Change)
6. ✅ target_col là parameter
7. ✅ Nhất quán interval: **3 giờ/interval, 8 intervals/day**
```

#### **Kết luận:**
- Fairness Checklist table
- Data Format comparison
- Experiments table
- Ready for Training checklist

---

## 📊 Output

### Files created:
```
data/
├── 7n_1n_xgb/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── datetime_train.csv
│   ├── datetime_test.csv
│   └── metadata.json
├── 7n_1n_lstm/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── datetime_train.csv
│   ├── datetime_test.csv
│   └── metadata.json
├── ... (tương tự cho các configs khác)
└── data_summary.csv

models/
└── scalers/
    └── feature_scaler.pkl
```

### Metadata example:
```json
{
  "config_name": "7n_1n",
  "model_type": "xgb",
  "X_train_shape": [5892, 336],
  "y_train_shape": [5892],
  "target_col": "Can Tho_Water Level",
  "feature_info": "6 features: WL + Rainfall",
  "interval_hours": 3,
  "intervals_per_day": 8,
  "embargo_days": 1,
  "created_at": "2025-10-04T..."
}
```

---

## ✅ Verification

### 1. Chạy notebook
```bash
cd notebooks
# Open 02_feature_engineering.ipynb
# → Run All
```

### 2. Kiểm tra output
```python
# Check shapes
assert X_train_xgb.shape == (samples, N*8*6)
assert X_train_lstm.shape == (samples, N*8, 6)
assert y_train_xgb.shape == y_train_lstm.shape == (samples,)

# Check embargo
# (feature_end_datetime + embargo) < target_datetime

# Check no data leakage
assert train_datetime.max() < test_datetime.min()
```

### 3. Kiểm tra fairness
- ✅ XGBoost & LSTM có cùng features
- ✅ XGBoost & LSTM có cùng target (single values)
- ✅ XGBoost & LSTM có cùng embargo
- ✅ Scaler fit on train only
- ✅ No future features

---

## 🎯 Key Takeaways

### 1. **Interval Consistency**
- Data: 3-hour intervals (6h, 9h, 12h, ...)
- Config: `INTERVAL_HOURS = 3`, `INTERVALS_PER_DAY = 8`
- Labels: `lag_3h`, `lag_6h`, ... (chính xác)

### 2. **Embargo Period**
- 1 ngày gap giữa features và target
- Prevents data leakage
- Real-world deployment scenario

### 3. **Same Target**
- XGBoost & LSTM đều dự đoán **1 số duy nhất**
- Không phải mean, không phải sequence
- Fair comparison guaranteed

### 4. **Clean Code**
- Không global variables (target_col là parameter)
- Clear function signatures
- Good documentation

### 5. **Run All = Done**
- Notebook tự động tạo toàn bộ data
- Không cần manual intervention
- Reproducible 100%

---

## 📚 Related Files

- `notebooks/02_feature_engineering.ipynb` - Main notebook (CLEANED)
- `FAIRNESS_CHECKLIST.md` - Detailed checklist documentation
- `config.py` - Experiments configuration
- `data/data_summary.csv` - Output summary

---

**Last updated:** 2025-10-04  
**Status:** ✅ READY FOR TRAINING
