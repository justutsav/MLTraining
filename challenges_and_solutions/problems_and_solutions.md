# Challenges & Solutions

## 1. Large Model File Exceeds GitHub's 100 MB Limit

**Problem:**
The trained Random Forest model (`crop_model.pkl`) was ~126 MB uncompressed, exceeding GitHub's 100 MB per-file limit.

**Solution:**
Used LZMA compression via `joblib.dump(model, 'crop_model.pkl', compress=3)`.
This shrinks the file to ~25 MB or less, well within GitHub's limit.
No need for Git LFS or auto-training on startup.

## 2. Ensuring Correct Predictions

**Problem:**
Model predictions can be wrong if feature encoding or feature order doesn't match training.

**Solution:**
The model uses a scikit-learn `Pipeline` with a `ColumnTransformer` that handles
`OneHotEncoding` (for Area, Item) and `StandardScaling` (for Year, Rainfall,
Pesticides, Temperature) internally.

This means:
- The app passes raw inputs (e.g. "India", "Maize") directly to `model.predict()`
- The Pipeline handles all encoding/scaling automatically
- Feature columns in the input DataFrame match the exact names used during training
- Error handling in `app.py` catches any mismatch issues
