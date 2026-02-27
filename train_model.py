"""
train_model.py
--------------
Trains the Crop Yield Prediction model and exports (with compression):
  - crop_model.pkl   (compressed sklearn Pipeline, ~25 MB or less)
  - areas_list.pkl   (sorted unique Area values for dropdowns)
  - items_list.pkl   (sorted unique Item values for dropdowns)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load Data
df = pd.read_csv('yield_df.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print("Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")

# 2. Prepare Features & Target
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

categorical_cols = ['Area', 'Item']
numerical_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

# 3. Build Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 4. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
print("Model Training Complete.")

# 5. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Accuracy (R2 Score): {r2:.4f}")
print(f"Average Prediction Error:  {mae:.2f} hg/ha")

# 6. Export with COMPRESSION (compress=3 shrinks ~126 MB -> ~25 MB)
joblib.dump(model, 'crop_model.pkl', compress=3)
print("[OK] Saved crop_model.pkl (compressed)")

areas = sorted(df['Area'].unique().tolist())
items = sorted(df['Item'].unique().tolist())

joblib.dump(areas, 'areas_list.pkl')
print(f"[OK] Saved areas_list.pkl  ({len(areas)} areas)")

joblib.dump(items, 'items_list.pkl')
print(f"[OK] Saved items_list.pkl  ({len(items)} items)")

print("\nAll exports complete!")
