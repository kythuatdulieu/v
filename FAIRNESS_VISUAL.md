# 🎯 FAIRNESS COMPARISON: XGBoost vs LSTM

## Timeline Visualization

```
Time Axis: ────────────────────────────────────────────────────────────────>
           
Features:  [━━━━━━ N days ━━━━━━]
                                  ⚡EMBARGO⚡  🎯 Target
                                  [1 day gap]   (single value)
           
Position:  i-N-embargo ... i-embargo ... i ... i+M-1
```

### Example: 7n_1n (7 days → predict day 8)

```
Days:      [-7 -6 -5 -4 -3 -2 -1] [EMBARGO] [+1]
           
Features:   ┌─────────────────┐
            │  7 days input   │
            │  (56 intervals) │
            └─────────────────┘
                                 ┌────────┐
                                 │  Gap   │
                                 │ 1 day  │
                                 │(8 int) │
                                 └────────┘
                                            🎯
                                            Target
                                            at day 8
```

---

## Data Structure Comparison

### XGBoost Format
```
Shape: (samples, features)
       (samples, N × intervals_per_day × num_features)
       (samples, N × 8 × 6)
       (samples, 336)  ← for 7n_1n

Structure:
┌─────────────────────────────────────────────────┐
│ Sample 1: [lag_3h_WL1, lag_3h_RF1, ..., lag_168h_RF3] │
│ Sample 2: [lag_3h_WL1, lag_3h_RF1, ..., lag_168h_RF3] │
│ ...                                               │
└─────────────────────────────────────────────────┘
       ↓
   Flattened
   tabular data
```

### LSTM Format
```
Shape: (samples, timesteps, features)
       (samples, N × intervals_per_day, num_features)
       (samples, N × 8, 6)
       (samples, 56, 6)  ← for 7n_1n

Structure:
┌─────────────────────────────────────────────┐
│ Sample 1:                                   │
│   ┌──────────────────────┐                 │
│   │ t=1:  [WL1, WL2, WL3,│                 │
│   │        RF1, RF2, RF3]│                 │
│   │ t=2:  [WL1, WL2, WL3,│                 │
│   │        RF1, RF2, RF3]│                 │
│   │ ...                  │                 │
│   │ t=56: [WL1, WL2, WL3,│                 │
│   │        RF1, RF2, RF3]│                 │
│   └──────────────────────┘                 │
└─────────────────────────────────────────────┘
       ↓
   Sequence
   (3D tensor)
```

---

## Fairness Checklist Visual

```
┌─────────────────────────────────────────────────────────┐
│             FAIRNESS CHECKLIST (7/7 PASS)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ 1. Time-based split                                │
│      └─ Train: 2022-03 to 2024-03 (80%)               │
│      └─ Test:  2024-03 to 2024-09 (20%)               │
│                                                         │
│  ✅ 2. Embargo period                                  │
│      └─ 1 day gap = 8 intervals                       │
│                                                         │
│  ✅ 3. Same target                                     │
│      └─ Both predict single value                     │
│                                                         │
│  ✅ 4. Scaler fairness                                 │
│      └─ Fit on train only                             │
│                                                         │
│  ✅ 5. No future features                              │
│      └─ Removed: WL_Change, month                     │
│                                                         │
│  ✅ 6. target_col parameter                            │
│      └─ No global variables                           │
│                                                         │
│  ✅ 7. Interval consistency                            │
│      └─ 3 hours/interval, 8 intervals/day             │
│                                                         │
└─────────────────────────────────────────────────────────┘

         XGBoost               LSTM
            ↓                   ↓
        ┌────────┐          ┌────────┐
        │ SAME   │          │ SAME   │
        │Features│    =     │Features│
        └────────┘          └────────┘
        ┌────────┐          ┌────────┐
        │ SAME   │          │ SAME   │
        │ Target │    =     │ Target │
        └────────┘          └────────┘
        ┌────────┐          ┌────────┐
        │ SAME   │          │ SAME   │
        │Embargo │    =     │Embargo │
        └────────┘          └────────┘

    ✅ FAIR COMPARISON GUARANTEED!
```

---

**Generated:** 2025-10-04  
**Status:** ✅ PRODUCTION READY
