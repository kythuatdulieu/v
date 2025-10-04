# ✅ FAIRNESS FIXES - Tóm tắt các Sửa đổi

## 📅 Ngày: 2025-10-04
## 📂 File: `notebooks/02_feature_engineering.ipynb`

---

## 🎯 **Mục tiêu**

Sửa chữa **7 vấn đề** trong Fairness Checklist để đảm bảo:
1. So sánh công bằng giữa XGBoost và LSTM
2. Không có data leakage
3. Kết quả đáng tin cậy và reproducible

---

## 🔧 **Các Sửa đổi Chi tiết**

### 1️⃣ **CRITICAL: Sửa Interval Calculation**

**Vấn đề:**
```python
# SAI:
N_intervals = N * 8  # Comment: "8 intervals/day (3h each)"
# Nhưng dữ liệu thực tế là 15-min intervals → 96 intervals/day!
# → Chỉ lấy 14 giờ thay vì 7 ngày!
```

**Sửa:**
```python
# ĐÚNG:
INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # = 96
N_intervals = N * INTERVALS_PER_DAY  # 7 days = 672 intervals
```

**Impact:** ⚠️ **CRITICAL** - Dữ liệu cũ HOÀN TOÀN SAI!

---

### 2️⃣ **CRITICAL: Thêm Embargo Period**

**Vấn đề:**
```python
# Feature window: [i-N_intervals ... i-1]
# Target window:  [i ... i+M_intervals-1]
#                  ↑ CHỒNG LẤN!
```

**Sửa:**
```python
EMBARGO_DAYS = 1
embargo_intervals = EMBARGO_DAYS * INTERVALS_PER_DAY

# Feature window: [i-N-embargo ... i-embargo-1]
# Embargo gap:    [i-embargo ... i-1]  ← KHÔNG DÙNG
# Target window:  [i ... i+M-1]        ← KHÔNG CHỒNG LẤN
```

**Timeline mới:**
```
[Features: 7 days] → [GAP: 1 day] → [Target: day 1]
```

**Impact:** ⚠️ **CRITICAL** - Loại bỏ data leakage nghiêm trọng!

---

### 3️⃣ **CRITICAL: Consistent Targets**

**Vấn đề:**
```python
# XGBoost: Dự đoán 1 SỐ (mean của M ngày)
y_val = data_sorted.iloc[i:i+M_intervals][target_col].mean()

# LSTM: Dự đoán CHUỖI (M*8 giá trị)
y_seq = data_sorted.iloc[i:i+M_intervals][target_col].values

# → KHÔNG SO SÁNH ĐƯỢC!
```

**Sửa:**
```python
# Cả 2 đều dự đoán 1 SỐ tại thời điểm N+EMBARGO+M
target_idx = i + M_intervals - 1
y_val = data_sorted.iloc[target_idx][target_col]  # Single value
```

**Impact:** ⚠️ **CRITICAL** - Fair comparison giữa XGB và LSTM!

---

### 4️⃣ **MEDIUM: target_col thành Parameter**

**Vấn đề:**
```python
def save_data(...):
    metadata = {'target_col': target_col}  # Global variable!
```

**Sửa:**
```python
def save_data(..., target_col, ...):
    metadata = {'target_col': target_col}  # Parameter
```

**Impact:** 🟢 MEDIUM - Code cleaner và reusable.

---

### 5️⃣ **HIGH: Nhãn Cột Chính xác**

**Vấn đề:**
```python
lag_hours = lag * 0.25  # OK cho 15-min
# Nhưng N_intervals = N*8 → Nhãn sai!
```

**Sửa:**
```python
lag_hours = lag * (INTERVAL_MINUTES / 60)  # Chính xác
# lag=1 → 0.25h
# lag=96 → 24h (1 day)
```

**Impact:** 🟡 HIGH - Metadata chính xác cho debugging.

---

## 📊 **Kết quả Sau Khi Sửa**

### Before:
```
❌ Interval: SAI (8 intervals/day thay vì 96)
❌ Embargo: KHÔNG CÓ → Data leakage
❌ Targets: Không nhất quán (mean vs sequence)
❌ Score: 3/7 PASS
```

### After:
```
✅ Interval: ĐÚNG (96 intervals/day for 15-min data)
✅ Embargo: 1 day gap → NO leakage
✅ Targets: Consistent (both predict single value)
✅ Score: 7/7 PASS
```

---

## 🔍 **Cách Verify**

### 1. Check Metadata:
```bash
cat ../data/7n_1n_xgb/metadata.json
```

Expected output:
```json
{
  "interval_minutes": 15,
  "intervals_per_day": 96,
  "embargo_days": 1,
  "embargo_intervals": 96,
  "y_train_shape": [n_samples],  // NOT [n_samples, sequence_length]
  "target_col": "Can Tho_Water Level"
}
```

### 2. Check Shapes:
```python
# XGBoost
X_train.shape  # (n_samples, N*96*num_features) - Flattened
y_train.shape  # (n_samples,) - Single values

# LSTM
X_train.shape  # (n_samples, N*96, num_features) - Sequences
y_train.shape  # (n_samples,) - Single values (SAME as XGB!)
```

### 3. Check Timeline:
```python
# Sample i=1000
# Features use: indices [1000-672-96 ... 1000-96-1] (7 days, before embargo)
# Embargo gap:  indices [1000-96 ... 1000-1]         (1 day, NOT USED)
# Target uses:  indices [1000 ... 1000+7]             (prediction at day 1)
```

---

## ⚠️ **QUAN TRỌNG**

### Models đã train trước đây:
- **KHÔNG SỬ DỤNG ĐƯỢC** vì dữ liệu cũ SAI!
- Performance cao có thể do:
  1. Interval calculation sai
  2. Data leakage (không có embargo)
  3. Target inconsistency

### Khuyến nghị:
1. ✅ **RE-TRAIN tất cả models** với dữ liệu mới
2. ✅ **So sánh kết quả** với models cũ
3. ✅ **Expect lower performance** (do loại bỏ leakage) - ĐÂY LÀ ĐIỀU TỐT!
4. ✅ **Document changes** trong báo cáo

---

## 📝 **Checklist Công bằng - Final Score**

| # | Tiêu chí | Before | After |
|---|----------|--------|-------|
| 1 | Chia theo thời gian | ✅ PASS | ✅ PASS |
| 2 | Embargo Period | ❌ FAIL | ✅ FIXED |
| 3 | Consistent Targets | ⚠️ PARTIAL | ✅ FIXED |
| 4 | Scaler | ✅ PASS | ✅ PASS |
| 5 | No Future Features | ✅ PASS | ✅ PASS |
| 6 | target_col Parameter | ❌ FAIL | ✅ FIXED |
| 7 | Interval Consistency | ❌ FAIL | ✅ FIXED |
| **Score** | **3/7** | **7/7** ✅ |

---

## 🎓 **Lessons Learned**

1. **Always verify interval calculations** - Comment không phải code!
2. **Embargo is essential** for time series - Prevent temporal leakage
3. **Consistent targets** make fair comparisons possible
4. **Document assumptions** clearly in metadata
5. **Test with small data** before full pipeline

---

## 📚 **References**

- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Embargo in Financial ML](https://mlfinlab.readthedocs.io/en/latest/labeling/labeling_excess_over_mean.html)
- [Fair Model Comparison](https://machinelearningmastery.com/how-to-avoid-data-leakage-when-performing-data-preparation/)

---

**Authored by:** GitHub Copilot  
**Date:** 2025-10-04  
**Status:** ✅ READY FOR TRAINING
