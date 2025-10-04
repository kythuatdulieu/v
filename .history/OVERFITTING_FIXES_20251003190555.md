# 🔴 LSTM Overfitting - Giải pháp

## Vấn đề hiện tại
- Train R²: **0.911** (rất cao)
- Test R²: **0.297** (rất thấp)
- Gap: **-68%** (overfitting nghiêm trọng)
- XGBoost Test R²: **0.760** (cao hơn LSTM 2.5 lần!)

## So sánh với XGBoost
```
XGBoost: Train 0.979 → Test 0.760 (gap -22%, chấp nhận được)
LSTM:    Train 0.911 → Test 0.297 (gap -68%, KHÔNG chấp nhận được)
```

## Nguyên nhân

### 1. Dropout không đủ mạnh
Hiện tại: `dropout = 0.2` (chỉ tắt 20% neurons)
- LSTM có 31,651 parameters (rất nhiều cho 7440 samples)
- Ratio: 0.24 samples/param → dễ overfit

### 2. Model quá phức tạp
```json
"best_params": {
    "units": 50,
    "n_layers": 2,  ← 2 layers có thể quá nhiều
    "dropout": 0.2  ← quá thấp
}
```

### 3. Epochs quá cao
- Trained 85 epochs
- Có thể đã học "thuộc lòng" training patterns
- Early stopping với patience=10 chưa đủ aggressive

### 4. Không có Regularization
- Không có L1/L2 regularization
- Không có RecurrentDropout

## Giải pháp

### Fix #1: Tăng Dropout (QUAN TRỌNG NHẤT)
```python
# File: config.py
LSTM_PARAMS = {
    'units': [50, 100],
    'n_layers': [1, 2],
    'dropout': [0.3, 0.4, 0.5],  # ← Tăng từ [0.1, 0.2] lên [0.3, 0.4, 0.5]
    'batch_size': [256],
    'epochs': [100],
    'patience': [10]
}
```

### Fix #2: Thêm RecurrentDropout
```python
# File: src/lstm_trainer.py, trong build_model()

# Thay vì:
model.add(LSTM(units, return_sequences=True, dropout=dropout))

# Sửa thành:
model.add(LSTM(
    units, 
    return_sequences=True, 
    dropout=dropout,
    recurrent_dropout=0.2  # ← THÊM recurrent_dropout
))
```

### Fix #3: Giảm patience của Early Stopping
```python
# File: config.py
LSTM_PARAMS = {
    ...
    'patience': [5, 7]  # ← Giảm từ [10] xuống [5, 7]
}
```

### Fix #4: Thêm L2 Regularization
```python
# File: src/lstm_trainer.py
from tensorflow.keras.regularizers import l2

# Trong build_model():
model.add(LSTM(
    units,
    return_sequences=True,
    dropout=dropout,
    recurrent_dropout=0.2,
    kernel_regularizer=l2(0.01)  # ← THÊM L2 regularization
))
```

### Fix #5: Batch Normalization
```python
# File: src/lstm_trainer.py
from tensorflow.keras.layers import BatchNormalization

# Sau mỗi LSTM layer:
model.add(LSTM(...))
model.add(BatchNormalization())  # ← THÊM batch normalization
model.add(Dropout(dropout))
```

## Ưu tiên thực hiện

### NGAY LẬP TỨC (High Impact, Low Effort):
1. ✅ **Fix #1: Tăng dropout** - Sửa config.py, re-train
   - Expected: Test R² tăng từ 0.297 → 0.4-0.5

2. ✅ **Fix #3: Giảm patience** - Sửa config.py, re-train
   - Prevent overfitting sớm hơn

### SAU ĐÓ (Medium Impact, Medium Effort):
3. **Fix #2: Recurrent dropout** - Sửa lstm_trainer.py
   - Regularize hidden states

4. **Fix #4: L2 regularization** - Sửa lstm_trainer.py
   - Penalize large weights

### NẾU VẪN CHƯA TỐT (High Impact, High Effort):
5. **Fix #5: Batch normalization** - Sửa lstm_trainer.py
   - Stabilize training

## Kết quả mong đợi

### Sau Fix #1 + #3:
- Test R²: **0.4 - 0.5** (tăng từ 0.297)
- Train-Test gap: **< 40%** (giảm từ 68%)

### Sau tất cả fixes:
- Test R²: **0.5 - 0.65** (so với XGBoost 0.76)
- Train-Test gap: **< 30%**
- LSTM competitive với XGBoost

## Lệnh thực hiện

```bash
# 1. Sửa config.py
# 2. Re-train LSTM
cd /home/duclinh/v
# Open notebook 05_train_all_models.ipynb
# Run all cells

# 3. So sánh kết quả
# Open notebook 06_model_comparison.ipynb
# Run all cells
```

## Kiểm tra thêm

### Data Leakage Check:
```python
# Verify sequential split trong lstm_trainer.py
# Đảm bảo validation data LUÔN SAU training data
```

### Feature Correlation:
```python
# Check multicollinearity giữa các features
# Có thể cần feature selection nếu correlation > 0.9
```

### Temporal Patterns:
```python
# Visualize predictions vs actuals
# Check xem LSTM có học được seasonal patterns không
```

## Debug Training Process

Thêm logging vào lstm_trainer.py:
```python
# Sau mỗi epoch in ra:
print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

Kiểm tra:
- Val_loss có giảm đều không?
- Có điểm nào val_loss bắt đầu tăng trong khi train_loss giảm? (overfitting point)
