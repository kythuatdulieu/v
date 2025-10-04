#!/bin/bash
# Script to verify all fixes have been applied correctly

echo "================================================"
echo "🔍 VERIFYING FIXES FOR LSTM IMPROVEMENTS"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
passed=0
failed=0

# Check function
check_fix() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo "Checking: $name"
    
    if eval "$command" | grep -q "$expected"; then
        echo -e "${GREEN}✅ PASS${NC}: $name"
        ((passed++))
    else
        echo -e "${RED}❌ FAIL${NC}: $name"
        echo "  Expected to find: $expected"
        ((failed++))
    fi
    echo ""
}

echo "======================================"
echo "Fix #1: Rainfall Features for LSTM"
echo "======================================"

check_fix \
    "LSTM features include rainfall" \
    "grep -A 2 'feature_cols_lstm' /home/duclinh/v/notebooks/02_feature_engineering.ipynb" \
    "WL_Change"

echo "======================================"
echo "Fix #2: Sequential Validation Split"
echo "======================================"

check_fix \
    "Sequential validation in grid_search()" \
    "grep -A 3 'validation_data' /home/duclinh/v/src/lstm_trainer.py" \
    "X_val_fold"

check_fix \
    "Sequential validation in train_best_model()" \
    "grep -A 3 'validation_data.*X_val_final' /home/duclinh/v/src/lstm_trainer.py" \
    "X_val_final"

echo "======================================"
echo "Fix #3: Target Scaling"
echo "======================================"

check_fix \
    "Target scaler initialization" \
    "grep 'self.target_scaler' /home/duclinh/v/src/lstm_trainer.py | head -1" \
    "target_scaler"

check_fix \
    "Target scaler fitting" \
    "grep 'target_scaler.fit_transform' /home/duclinh/v/src/lstm_trainer.py" \
    "fit_transform"

check_fix \
    "Inverse transform in evaluate" \
    "grep 'inverse_transform' /home/duclinh/v/src/lstm_trainer.py" \
    "inverse_transform"

check_fix \
    "Saving target scaler" \
    "grep 'joblib.dump.*target_scaler' /home/duclinh/v/src/lstm_trainer.py" \
    "target_scaler"

echo "======================================"
echo "Summary"
echo "======================================"
echo -e "✅ Passed: ${GREEN}$passed${NC}"
echo -e "❌ Failed: ${RED}$failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All fixes verified successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run: jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb"
    echo "2. Run: jupyter nbconvert --to notebook --execute notebooks/05_train_all_models.ipynb"
    echo "3. Compare results in notebook 06_model_comparison.ipynb"
    exit 0
else
    echo -e "${RED}⚠️  Some fixes may not be applied correctly.${NC}"
    echo "Please review the failed checks above."
    exit 1
fi
