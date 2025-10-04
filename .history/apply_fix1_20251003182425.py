"""
Script to apply Fix #1: Update feature_cols_lstm in notebook
"""
import json

# Read notebook
notebook_path = '/home/duclinh/v/notebooks/02_feature_engineering.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and update the cell
updated = False
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        # Join lines to search
        code = ''.join(cell['source'])
        
        # Check if this is the cell we need to update
        if 'feature_cols_lstm' in code and 'Water Level' in code and 'feature_cols_xgb' in code:
            print("Found cell to update!")
            print(f"Old code:\n{code[:200]}...")
            
            # Replace the old line
            old_line = "feature_cols_lstm = [col for col in train_data.columns if 'Water Level' in col]"
            new_lines = [
                "# ✅ FIX #1: LSTM dùng CÙNG features với XGBoost (water level + rainfall)\n",
                "# Trước: LSTM chỉ dùng water level → thiếu 50% thông tin\n",
                "# Sau: LSTM dùng cả water level + rainfall → công bằng với XGBoost\n",
                "feature_cols_lstm = [col for col in train_data.columns if col not in ['datetime', 'month'] \n",
                "                     and 'WL_Change' not in col]"
            ]
            
            # Update cell source
            new_source = []
            skip_next = False
            for i, line in enumerate(cell['source']):
                if skip_next:
                    skip_next = False
                    continue
                    
                if old_line in line:
                    # Replace this line and the next line
                    new_source.extend(new_lines)
                    new_source.append('\n')
                    new_source.append('\n')
                    skip_next = True  # Skip the next empty line
                elif "print(f\"LSTM features (chỉ water level)" in line:
                    # Update print statement
                    new_source.append("print(f\"\\nXGBoost features ({len(feature_cols_xgb)}): {feature_cols_xgb}\")\n")
                    new_source.append("print(f\"LSTM features ({len(feature_cols_lstm)}) - UPDATED TO INCLUDE RAINFALL: {feature_cols_lstm}\")\n")
                elif "print(f\"\\nXGBoost features:" in line or "print(f\"Target column:" in line:
                    # Skip redundant prints
                    if "Target column" in line:
                        new_source.append(line)
                        new_source.append("print(f\"\\n✅ LSTM now has access to rainfall data - expected significant improvement!\")\n")
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            updated = True
            print(f"\nNew code:\n{''.join(new_source[:300])}...")
            break

if updated:
    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"\n✅ Notebook updated successfully: {notebook_path}")
else:
    print("\n❌ Could not find the cell to update")
