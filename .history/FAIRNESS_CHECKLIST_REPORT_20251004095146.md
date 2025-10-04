# ✅ FAIRNESS CHECKLIST - Phân tích Chi tiết

## 📊 **Tổng quan Checklist**

| # | Tiêu chí | Trạng thái | Mức độ ưu tiên |
|---|----------|------------|----------------|
| 1 | Chia theo thời gian, không random | ✅ PASS | 🔴 CRITICAL |
| 2 | Target_end/start + embargo tránh overlap | ❌ FAIL | 🔴 CRITICAL |
| 3 | Cùng mục tiêu cho XGB và LSTM | ⚠️ PARTIAL | 🟡 HIGH |
| 4 | Scaler fit trên train, transform test | ✅ PASS | 🔴 CRITICAL |
| 5 | Không dùng đặc trưng có yếu tố tương lai | ✅ PASS | 🔴 CRITICAL |
| 6 | Target_col thành tham số trong save_data | ❌ FAIL | 🟢 MEDIUM |
| 7 | Nhất quán đơn vị interval và nhãn cột | ❌ FAIL | 🟡 HIGH |

---

## 📝 **Chi tiết từng tiêu chí**

### 1️⃣ **Chia theo thời gian, không random** ✅ PASS

**Mã hiện tại:**
```python
# File: notebooks/01_data_cleaning_and_eda.ipynb
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
split_idx = int(len(df_clean) * 0.8)
train_data = df_clean.iloc[:split_idx].copy()
test_data = df_clean.iloc[split_idx:].copy()
```

**Kết luận:** ✅ **ĐÚNG** - Chia theo thứ tự thời gian, không shuffle.

---

### 2️⃣ **Target_end/start + embargo tránh overlap** ❌ FAIL

**Vấn đề nghiêm trọng:**

Trong `create_lag_features_xgb`:
```python
for i in range(N_intervals, len(data_sorted) - M_intervals + 1):
    # Features: từ i-N_intervals đến i-1
    # Target: từ i đến i+M_intervals-1
```

**CHỒNG LẤN:** Feature window kết thúc tại `i-1`, Target window bắt đầu tại `i` → **KHÔNG CÓ GAP!**

**Ví dụ cụ thể (N=7 days, M=1 day):**
```
Feature window: index [i-56 ... i-1]  (7 days * 8 intervals)
Target window:  index [i ... i+7]     (1 day * 8 intervals)
                       ↑
                 CHỒNG LẤN NGAY TẠI index i!
```

**Hậu quả:**
- Training: Model học được mối quan hệ "tương lai gần" → Overfit
- Test: Không realistic vì thiếu embargo period
- Kết quả: Performance cao giả tạo

**Giải pháp:**
```python
# Thêm embargo period
EMBARGO_INTERVALS = 8  # 1 ngày = 8 intervals

for i in range(N_intervals + EMBARGO_INTERVALS, len(data_sorted) - M_intervals + 1):
    # Features: [i-N_intervals-EMBARGO_INTERVALS ... i-EMBARGO_INTERVALS-1]
    # Embargo:  [i-EMBARGO_INTERVALS ... i-1] (KHÔNG DÙNG)
    # Target:   [i ... i+M_intervals-1]
```

---

### 3️⃣ **Cùng mục tiêu cho XGB và LSTM** ⚠️ PARTIAL

**Vấn đề không nhất quán:**

Trong `create_lag_features_xgb` (15-min intervals):
```python
if M == 1:
    y_val = data_sorted.iloc[i + M_intervals - 1][target_col]
else:
    y_val = data_sorted.iloc[i:i + M_intervals][target_col].mean()  # TRUNG BÌNH
```

Trong `create_sequences_lstm` (15-min intervals):
```python
if M == 1:
    y_seq = data_sorted.iloc[i + M_intervals - 1][target_col]
else:
    y_seq = data_sorted.iloc[i:i + M_intervals][target_col].values  # CHUỖI
```

**Vấn đề:**
- XGBoost: Dự đoán **1 SỐ** (trung bình M ngày)
- LSTM: Dự đoán **CHUỖI** (M*8 giá trị)
- Không thể so sánh trực tiếp!

**Nhưng trong Daily functions:**
```python
# Cả 2 đều dự đoán 1 SỐ tại ngày N+M
target_value = data_sorted.iloc[i+M-1][target_col]
```

✅ **Daily version ĐÚNG**, nhưng 15-min version **SAI**.

---

### 4️⃣ **Scaler fit trên train, transform test** ✅ PASS

```python
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_data[feature_cols_xgb])
test_features_scaled = scaler.transform(test_data[feature_cols_xgb])
```

✅ **HOÀN HẢO** - Không có data leakage!

---

### 5️⃣ **Không dùng đặc trưng có yếu tố tương lai** ✅ PASS

```python
# Loại bỏ WL_Change (tính từ future data)
feature_cols_xgb = [col for col in train_data.columns 
                    if col not in ['datetime', 'month'] 
                    and 'WL_Change' not in col]
```

✅ **ĐÚNG** - Đã loại bỏ các feature có leak tương lai.

---

### 6️⃣ **Target_col thành tham số trong save_data** ❌ FAIL

**Mã hiện tại:**
```python
def save_data(X_train, y_train, X_test, y_test, datetime_train, datetime_test, 
              config_name, model_type, feature_info=None):
    # ...
    metadata = {
        'target_col': target_col,  # ← Dùng biến global!
        # ...
    }
```

**Vấn đề:**
- `target_col` là biến global, không phải parameter
- Nếu target thay đổi → metadata sai

**Giải pháp:**
```python
def save_data(X_train, y_train, X_test, y_test, datetime_train, datetime_test, 
              config_name, model_type, target_col, feature_info=None):
    #                                              ↑ Thêm parameter
```

---

### 7️⃣ **Nhất quán đơn vị interval và nhãn cột** ❌ FAIL

**VẤN ĐỀ NGHIÊM TRỌNG:**

**Comment nói:** "8 intervals mỗi ngày (max mỗi interval là 3 tiếng)"
```python
N_intervals = N * 8  # 8 intervals/day
```

**Thực tế dữ liệu:** 15 phút/interval → **96 intervals/day!**

**Mâu thuẫn:**
- 8 intervals/day = 3 giờ/interval
- Nhưng dữ liệu là 15 phút/interval
- **N_intervals phải là N*96, KHÔNG PHẢI N*8!**

**Nhãn cột lag sai:**
```python
lag_hours = lag * 0.25  # Đúng cho 15-min
# Nhưng N_intervals = N*8 → SAI!
```

**Ví dụ lỗi:**
```
N=7 days
N_intervals = 7*8 = 56 intervals
→ Chỉ lấy 56*15min = 14 giờ (KHÔNG PHẢI 7 NGÀY!)

Đúng phải là: N_intervals = 7*96 = 672 intervals
```

---

## 🔧 **KHUYẾN NGHỊ SỬA CHỮA**

### Priority 1 (CRITICAL):

1. **Sửa interval calculation:**
```python
# Xác định rõ interval size
INTERVAL_MINUTES = 15  # hoặc 180 nếu dùng 3h
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 96 cho 15min, 8 cho 3h

N_intervals = N * INTERVALS_PER_DAY
M_intervals = M * INTERVALS_PER_DAY
```

2. **Thêm embargo period:**
```python
EMBARGO_DAYS = 1
EMBARGO_INTERVALS = EMBARGO_DAYS * INTERVALS_PER_DAY

for i in range(N_intervals + EMBARGO_INTERVALS, 
               len(data_sorted) - M_intervals + 1):
    # Features: [i-N-EMBARGO ... i-EMBARGO-1]
    # Gap: [i-EMBARGO ... i-1]
    # Target: [i ... i+M-1]
```

3. **Thống nhất target type:**
```python
# Cả XGB và LSTM đều dự đoán 1 SỐ
if M == 1:
    target = data_sorted.iloc[i][target_col]
else:
    # Gap forecasting: predict value at day N+M
    target = data_sorted.iloc[i+M_intervals-1][target_col]
```

### Priority 2 (HIGH):

4. **Sửa save_data:**
```python
def save_data(..., target_col, ...):
    metadata = {'target_col': target_col, ...}
```

5. **Nhãn cột nhất quán:**
```python
if INTERVAL_MINUTES == 15:
    lag_label = f"lag_{lag*0.25:.2f}h"  # 0.25h per 15min
elif INTERVAL_MINUTES == 180:
    lag_label = f"lag_{lag*3:.1f}h"     # 3h per interval
```

---

## 📊 **KẾT LUẬN**

**Score: 3/7 PASS ⚠️**

**Vấn đề nghiêm trọng nhất:**
1. **Interval calculation SAI** → Dữ liệu không đúng temporal window
2. **Không có embargo** → Data leakage nghiêm trọng
3. **Target inconsistency** → Không fair comparison

**Khuyến nghị:**
🔴 **DỪNG training** cho đến khi sửa các vấn đề CRITICAL!

Các model hiện tại có performance cao có thể do data leakage, không phải do model tốt.
