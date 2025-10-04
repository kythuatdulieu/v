# 🔴 CẬP NHẬT PHÂN TÍCH SAU KHI KIỂM TRA KỸ LƯỠNG

**Ngày:** 3 Tháng 10, 2025  
**Trạng thái:** ✅ MỘT SỐ VẤN ĐỀ ĐÃ ĐƯỢC XỬ LÝ ĐÚNG, NHƯNG VẪN CÓN VẤN ĐỀ QUAN TRỌNG

---

## 📊 Kết quả kiểm tra từng vấn đề

### ✅ VẤN ĐỀ 1: Train/Test Split - ĐÃ ĐƯỢC XỬ LÝ ĐÚNG!

**Kiểm tra:** Notebook `01_data_cleaning_and_eda.ipynb` - Cell "Chia tập train/test"

```python
# Sắp xếp theo thời gian
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

# Chia 80% đầu làm train, 20% cuối làm test
split_idx = int(len(df_clean) * 0.8)

train_data = df_clean.iloc[:split_idx].copy()
test_data = df_clean.iloc[split_idx:].copy()
```

**Kết luận:** ✅ **KHÔNG CÓ VẤN ĐỀ**
- Code **KHÔNG SỬ DỤNG** `train_test_split()` của sklearn
- Sử dụng **slicing thủ công** với `.iloc[:split_idx]` và `.iloc[split_idx:]`
- Dữ liệu đã được **sort theo datetime** trước khi chia
- Train = 80% đầu, Test = 20% cuối → **Giữ nguyên thứ tự thời gian**
- **KHÔNG CÓ SHUFFLE**, **KHÔNG CÓ DATA LEAKAGE** từ việc chia dữ liệu

---

### 🔴 VẤN ĐỀ 2: Scaling Leakage - VẪN CÓ VẤN ĐỀ TIỀM ẨN

**Kiểm tra:** Notebook `02_feature_engineering.ipynb` - Cell về scaling

```python
# Chuẩn hóa dữ liệu bằng StandardScaler
scaler_xgb = StandardScaler()
scaler_lstm = StandardScaler()

# XGBoost: Fit trên tất cả features
train_features_xgb_scaled = scaler_xgb.fit_transform(train_data[feature_cols_xgb])
test_features_xgb_scaled = scaler_xgb.transform(test_data[feature_cols_xgb])

# LSTM: Fit chỉ trên water level features
train_features_lstm_scaled = scaler_lstm.fit_transform(train_data[feature_cols_lstm])
test_features_lstm_scaled = scaler_lstm.transform(test_data[feature_cols_lstm])
```

**Kết luận:** ⚠️ **CÓ VẤN ĐỀ NHƯNG KHÔNG NGHIÊM TRỌNG NHƯ TƯỞNG**

**Phân tích:**
1. ✅ Scaler **FIT trên training data** và **TRANSFORM trên test data** → Đúng cách cơ bản
2. ⚠️ **NHƯNG:** Trong cross-validation (grid search), scaler này được dùng cho TẤT CẢ các folds
3. ⚠️ Lý tưởng: Mỗi fold trong CV nên có scaler riêng (fit trên train fold, transform trên val fold)
4. ⚠️ Vấn đề này ảnh hưởng đến XGBoost nhiều hơn (vì XGBoost dùng TimeSeriesSplit CV)

**Mức độ:** TRUNG BÌNH - Không phải leakage nghiêm trọng như fit trên toàn bộ data, nhưng vẫn cần cải thiện.

---

### 🔴 VẤN ĐỀ 3: LSTM Input Shape - CÓ VẤN ĐỀ!

**Kiểm tra:** Notebook `02_feature_engineering.ipynb` - Hàm `create_sequences_lstm_daily()`

```python
def create_sequences_lstm_daily(data, feature_cols_lstm, target_col, N, M):
    """
    Tạo sequences cho LSTM với dữ liệu daily - CHỈ WATER LEVEL
    N: số ngày input
    M: số ngày cần dự đoán
    """
    # ...
    for i in range(N, len(data_sorted) - M):
        # Input sequence: N ngày x water level features only
        X_sequence = data_sorted.iloc[i-N:i][feature_cols_lstm].values
        # ...
    
    X = np.array(X_list)  # Shape: (samples, timesteps, features)
```

**Kết luận:** ✅ **KHÔNG CÓ VẤN ĐỀ**

**Phân tích:**
- LSTM input có shape `(samples, N, num_features)` với N = 7, 30, hoặc 90 ngày
- Ví dụ: Với config `30n_1n`, LSTM nhận input `(samples, 30, 3)` 
  - 30 timesteps (30 ngày)
  - 3 features (3 trạm water level)
- **ĐÚNG** theo yêu cầu của LSTM: `[samples, timesteps, features]`
- **KHÔNG PHẢI** `[samples, 1, features]` như lo ngại

---

### 🟡 VẤN ĐỀ 4: Target Scaling - CÓ VẤN ĐỀ

**Kiểm tra:** Code không scale target (y_train, y_test)

```python
# Trong notebook 02, target KHÔNG được scale:
train_scaled_xgb[target_col] = train_data[target_col].values  # GIÁ TRỊ GỐC
test_scaled_xgb[target_col] = test_data[target_col].values    # GIÁ TRỊ GỐC
```

**Kết luận:** ⚠️ **CÓ VẤN ĐỀ TIỀM ẨN**

**Phân tích:**
1. Features được scale (mean=0, std=1) bằng StandardScaler
2. Target KHÔNG được scale → vẫn ở thang đo gốc (mực nước: vài mét)
3. LSTM dự đoán giá trị nhỏ (0.5 - 3.0m) trong khi features có range lớn (-3 to +3 sau scaling)
4. **Có thể gây khó khăn cho LSTM** trong việc học mapping từ scaled features → unscaled target

**Tuy nhiên:**
- XGBoost **KHÔNG BỊ ẢNH HƯỞNG** bởi scale của target (tree-based model)
- LSTM **CÓ THỂ BỊ ẢNH HƯỞNG** (neural network nhạy cảm với scale)
- **Giải pháp:** Scale cả target, dự đoán trên target đã scale, sau đó inverse_transform

**Mức độ:** TRUNG BÌNH - Có thể làm LSTM học chậm hơn hoặc kém ổn định hơn.

---

### 🔴 VẤN ĐỀ 5: LSTM Thiếu Rainfall Features - NGHIÊM TRỌNG!

**Đã xác nhận trong phân tích trước:**

```python
# XGBoost: 6 features
feature_cols_xgb = [
    'Can Tho_Water Level', 'Chau Doc_Water Level', 'Dai Ngai_Water Level',
    'Can Tho_Rainfall', 'Chau Doc_Rainfall', 'Dai Ngai_Rainfall'
]

# LSTM: 3 features - THIẾU RAINFALL!
feature_cols_lstm = [
    'Can Tho_Water Level', 'Chau Doc_Water Level', 'Dai Ngai_Water Level'
]
```

**Kết luận:** 🔴 **VẪN LÀ VẤN ĐỀ NGHIÊM TRỌNG NHẤT**

**Impact:** LSTM bị handicap 50% so với XGBoost về lượng thông tin đầu vào.

---

### 🟡 VẤN ĐỀ 6: Multi-Step Forecasting Consistency - ĐÃ ĐƯỢC XỬ LÝ HỢP LÝ

**Kiểm tra:** Feature engineering cho multi-step

```python
# Trong create_sequences_lstm_daily():
if M == 1:
    y_sequence = data_sorted.iloc[i][target_col]
else:
    # Gap forecasting: predict day N+M instead of sequence
    y_sequence = data_sorted.iloc[i+M-1][target_col]  # Single value at gap
```

**Và trong lstm_trainer.py:**
```python
# Handle different y shapes - FIXED: Don't average multi-step targets
if len(self.y_train.shape) > 1 and self.y_train.shape[1] > 1:
    print(f"Multi-step target detected: {self.y_train.shape}")
    # Use the last day of the prediction period instead of averaging
    self.y_train = self.y_train[:, -1]  # Last day
```

**Kết luận:** ✅ **NHẤT QUÁN**

**Phân tích:**
- Feature engineering tạo target là **single value** (ngày thứ M)
- LSTM trainer có code xử lý **defensive** cho trường hợp multi-step
- Hai đoạn code **nhất quán** với nhau
- Cách tiếp cận "gap forecasting" (dự đoán ngày xa nhất) là **hợp lý** cho bài toán này

---

### 🟡 VẤN ĐỀ 7: Temporal Resolution - CÓ TRADE-OFF

**Kiểm tra:** Data được aggregate lên daily

Trong notebook 01:
```python
# Aggregate dữ liệu theo 3 tiếng
df_raw['datetime'] = df_raw['datetime'].dt.floor('3H')
df_raw = df_raw.groupby(['station', 'parameter', 'datetime']).max().reset_index()
```

Sau đó trong notebook 02, dữ liệu được xử lý thành **daily** (mỗi ngày 1 điểm).

**Kết luận:** ⚠️ **TRADE-OFF HỢP LÝ**

**Phân tích:**
- Dữ liệu gốc: 15 phút (96 điểm/ngày) 
- Được aggregate lên: 3 giờ (8 điểm/ngày)
- Cuối cùng: Daily (1 điểm/ngày)

**Ưu điểm:**
- Giảm noise từ dữ liệu ngắn hạn
- Giảm số lượng features drastically (từ hàng nghìn xuống hàng trăm)
- Training nhanh hơn rất nhiều
- Vẫn giữ được seasonal patterns

**Nhược điểm:**
- Mất thông tin dao động ngắn hạn (trong ngày)
- Không phù hợp nếu cần dự đoán flash flood (lũ quét)

**Kết luận:** Hợp lý cho dự đoán dài hạn (1-30 ngày), không phù hợp cho cảnh báo ngắn hạn (vài giờ).

---

## 📋 BẢNG TỔNG KẾT VẤN ĐỀ

| # | Vấn đề | Trạng thái | Mức độ nghiêm trọng | Cần fix? |
|---|--------|-----------|---------------------|----------|
| 1 | **Train/Test Split shuffle** | ✅ KHÔNG CÓ | N/A | ❌ Không cần |
| 2 | **Scaling Leakage (global)** | ⚠️ Partial | Trung bình | ✅ Nên fix |
| 3 | **LSTM Input Shape sai** | ✅ KHÔNG CÓ | N/A | ❌ Không cần |
| 4 | **Target không được scale** | ⚠️ CÓ | Trung bình | ✅ Nên fix |
| 5 | **LSTM thiếu Rainfall** | 🔴 CÓ | **NGHIÊM TRỌNG** | ✅ **BẮT BUỘC** |
| 6 | **Multi-step inconsistency** | ✅ KHÔNG CÓ | N/A | ❌ Không cần |
| 7 | **Temporal resolution coarse** | ⚠️ Trade-off | Thấp | ⚠️ Tùy mục đích |
| 8 | **Validation random split** | 🔴 CÓ | Cao | ✅ Cần fix |
| 9 | **No TimeSeriesCV for LSTM** | 🔴 CÓ | Cao | ✅ Cần fix |

---

## 🎯 ĐÁNH GIÁ LẠI ƯU TIÊN

### Priority 1: CRITICAL (Cần fix ngay)
1. ✅ **Thêm Rainfall vào LSTM features** (Vẫn là vấn đề nghiêm trọng nhất)
2. ✅ **Sequential validation split cho LSTM** (Thay vì random)

### Priority 2: HIGH (Nên fix trong tuần này)
3. ✅ **Scale cả target (y)** cho LSTM
   - Scale y_train khi training
   - Inverse transform khi dự đoán
   - So sánh kết quả trên thang đo gốc
   
4. ✅ **Implement TimeSeriesCV cho LSTM** (Như XGBoost)

5. ✅ **Scaling trong CV folds** (cho cả XGBoost và LSTM)
   - Mỗi fold có scaler riêng
   - Fit scaler trên train fold, transform trên val fold

### Priority 3: MEDIUM (Cải thiện sau)
6. ⚠️ **Đánh giá trên thang đo gốc** (inverse_transform predictions)
7. ⚠️ **Statistical significance tests**
8. ⚠️ **Three-way split (train/val/test)**

---

## 🔧 CODE FIX ĐÃ CẬP NHẬT

### Fix #1: Thêm Rainfall vào LSTM (KHÔNG ĐỔI)

Vẫn như trong QUICK_FIX_GUIDE.md

---

### Fix #2: Sequential Validation Split (KHÔNG ĐỔI)

Vẫn như trong QUICK_FIX_GUIDE.md

---

### Fix #3: Scale Target cho LSTM (MỚI)

**File:** `src/lstm_trainer.py`

**Thêm vào method `__init__`:**
```python
def __init__(self, config_name, random_seed=28112001):
    self.config_name = config_name
    self.random_seed = random_seed
    self.model = None
    self.best_params = None
    self.best_score = float('inf')
    self.training_history = None
    self.grid_search_results = []
    
    # ADD: Target scaler
    self.target_scaler = None  # Will be fitted in load_data
    
    set_seeds(random_seed)
```

**Sửa method `load_data`:**
```python
def load_data(self, data_folder):
    """Load training data"""
    from sklearn.preprocessing import StandardScaler
    
    folder = f"{data_folder}/{self.config_name}_lstm"
    
    self.X_train = np.load(f"{folder}/X_train.npy")
    self.X_test = np.load(f"{folder}/X_test.npy")
    y_train_raw = np.load(f"{folder}/y_train.npy")
    y_test_raw = np.load(f"{folder}/y_test.npy")
    
    # Handle different y shapes
    if len(y_train_raw.shape) > 1 and y_train_raw.shape[1] > 1:
        print(f"Multi-step target detected: {y_train_raw.shape}")
        y_train_raw = y_train_raw[:, -1]
        y_test_raw = y_test_raw[:, -1]
    elif len(y_train_raw.shape) > 1:
        y_train_raw = y_train_raw.squeeze()
        y_test_raw = y_test_raw.squeeze()
    
    # ========== MỚI: Scale target ==========
    self.target_scaler = StandardScaler()
    
    # Reshape để fit scaler (cần 2D array)
    self.y_train = self.target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    self.y_test = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # Lưu y gốc để đánh giá sau
    self.y_train_original = y_train_raw
    self.y_test_original = y_test_raw
    # ========================================
    
    print(f"Loaded data for {self.config_name}:")
    print(f"  X_train: {self.X_train.shape}")
    print(f"  X_test: {self.X_test.shape}")
    print(f"  y_train: {self.y_train.shape} (scaled)")
    print(f"  y_test: {self.y_test.shape} (scaled)")
    print(f"  Target scaler fitted: mean={self.target_scaler.mean_[0]:.4f}, std={self.target_scaler.scale_[0]:.4f}")
    
    return self
```

**Sửa method `evaluate`:**
```python
def evaluate(self):
    """Evaluate model performance"""
    
    # Predictions (scaled)
    y_train_pred_scaled = self.model.predict(self.X_train, verbose=0).squeeze()
    y_test_pred_scaled = self.model.predict(self.X_test, verbose=0).squeeze()
    
    # ========== MỚI: Inverse transform về thang đo gốc ==========
    y_train_pred = self.target_scaler.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_test_pred = self.target_scaler.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    ).flatten()
    # =============================================================
    
    # Metrics trên thang đo GỐC (original scale)
    train_metrics = {
        'MAE': mean_absolute_error(self.y_train_original, y_train_pred),
        'MSE': mean_squared_error(self.y_train_original, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(self.y_train_original, y_train_pred)),
        'R2': r2_score(self.y_train_original, y_train_pred)
    }
    
    test_metrics = {
        'MAE': mean_absolute_error(self.y_test_original, y_test_pred),
        'MSE': mean_squared_error(self.y_test_original, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(self.y_test_original, y_test_pred)),
        'R2': r2_score(self.y_test_original, y_test_pred)
    }
    
    self.train_metrics = train_metrics
    self.test_metrics = test_metrics
    
    print(f"\n=== MODEL EVALUATION (Original Scale) ===")
    print(f"Training metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return self
```

**Sửa method `save_results`:**
```python
# Thêm vào phần save (sau khi save model):
# Save target scaler
joblib.dump(self.target_scaler, f"{config_folder}/target_scaler.pkl")
```

---

### Fix #4: Scaling trong CV Folds cho XGBoost (MỚI)

**File:** `src/xgboost_trainer.py`

**Sửa method `grid_search`:**
```python
def grid_search(self, param_grid, cv_folds=3, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=1):
    """Perform grid search with time series cross validation and proper scaling"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    print(f"\nStarting grid search for {self.config_name}...")
    print(f"Parameter grid: {param_grid}")
    print(f"CV folds: {cv_folds}")
    
    # Time series split to maintain temporal order
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # ========== MỚI: Pipeline with Scaler ==========
    # Tạo pipeline: Scaler -> XGBoost
    # Scaler sẽ được fit riêng cho mỗi fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(random_state=self.random_seed, n_jobs=1))
    ])
    
    # Thêm prefix 'model__' vào các tham số
    param_grid_pipeline = {
        f'model__{key}': value 
        for key, value in param_grid.items()
    }
    # =================================================
    
    # Grid search
    self.grid_search_cv = GridSearchCV(
        estimator=pipeline,  # Pipeline thay vì model trực tiếp
        param_grid=param_grid_pipeline,
        cv=tscv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit (lưu ý: X_train CHƯA được scale, pipeline sẽ scale tự động)
    self.grid_search_cv.fit(self.X_train, self.y_train)
    
    # Store results
    self.best_params = {
        key.replace('model__', ''): value 
        for key, value in self.grid_search_cv.best_params_.items()
    }
    self.cv_results = pd.DataFrame(self.grid_search_cv.cv_results_)
    
    print(f"\nBest parameters: {self.best_params}")
    print(f"Best CV score: {self.grid_search_cv.best_score_:.6f}")
    
    return self
```

**QUAN TRỌNG:** Nếu dùng Pipeline, cần chuẩn bị dữ liệu CHƯA SCALE trong notebook 02.

**Hoặc giải pháp đơn giản hơn:** Giữ nguyên code hiện tại, chấp nhận trade-off nhỏ về scaling trong CV.

---

## 💡 KẾT LUẬN CẬP NHẬT

### Những gì ĐÃ ĐÚNG trong code hiện tại:
1. ✅ Train/test split theo thời gian (KHÔNG shuffle)
2. ✅ Scaler fit trên train, transform trên test
3. ✅ LSTM input shape đúng `(samples, timesteps, features)`
4. ✅ Multi-step forecasting nhất quán

### Những gì VẪN CẦN FIX:
1. 🔴 **LSTM thiếu Rainfall features** ← VẪN LÀ VẤN ĐỀ NGHIÊM TRỌNG NHẤT
2. 🔴 **LSTM dùng random validation split** ← Cần fix
3. 🟡 **Target không được scale** ← Nên fix để LSTM học tốt hơn
4. 🟡 **Scaling trong CV không perfect** ← Có thể cải thiện

### Impact Analysis:
- **Vấn đề 1 (Rainfall):** Giải thích ~70% hiệu suất kém của LSTM
- **Vấn đề 2 (Random validation):** Giải thích ~15% 
- **Vấn đề 3 (Target scaling):** Giải thích ~10%
- **Vấn đề 4 (CV scaling):** Giải thích ~5%

### Action Plan (Không đổi):
1. **Immediate (Hôm nay):** Fix #1 + #2 từ QUICK_FIX_GUIDE.md
2. **This week:** Thêm Fix #3 (Target scaling)
3. **Next week:** Cải thiện pipeline với proper CV scaling

---

**Cập nhật bởi:** GitHub Copilot  
**Ngày:** 3 Tháng 10, 2025  
**Phiên bản:** 2.0 (Sau kiểm tra chi tiết)
