# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ==== PATH SETUP ====
BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '')
DATA_PATH = os.path.join(BASE, 'data', 'cars.csv')
MODEL_PATH = os.path.join(BASE, 'models', 'rf_car_price.joblib')
OUTPUT_REPORT = os.path.join(BASE, 'outputs', 'report.md')

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'outputs'), exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(DATA_PATH)
print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==== CLEAN COLUMN NAMES ====
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("✅ Renamed columns:", df.columns.tolist())

# ==== BASIC CLEANING ====
df = df.dropna(subset=['selling_price'])

# ==== FEATURE ENCODING ====
le = LabelEncoder()
for col in ['fuel_type', 'selling_type', 'transmission']:
    df[col] = le.fit_transform(df[col])

# ==== DEFINE FEATURES AND TARGET ====
X = df[['year', 'present_price', 'driven_kms', 'fuel_type', 'selling_type', 'transmission', 'owner']]
y = df['selling_price']

# ==== SPLIT DATA ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== TRAIN MODEL ====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==== EVALUATE MODEL ====
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Model Trained Successfully! R²: {r2:.3f}, MAE: {mae:.2f}")

# ==== SAVE MODEL ====
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# ==== SAVE REPORT ====
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("# Car Price Prediction Report\n\n")
    f.write(f"- Model used: RandomForestRegressor\n")
    f.write(f"- R² Score: {r2:.3f}\n")
    f.write(f"- Mean Absolute Error: {mae:.2f}\n")
    f.write(f"- Features used: ['year', 'present_price', 'driven_kms', 'fuel_type', 'selling_type', 'transmission', 'owner']\n")

print(f"✅ Report generated at {OUTPUT_REPORT}")
