# 📄 generate_features_from_base.py
# Adds engineered features like driver_form, constructor_form, circuit_encoded, grid_advantage

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# === Config ===
INPUT_CSV = "../data/f1_features_2024_enriched.csv"
OUTPUT_CSV = "../data/f1_features_2024_with_features.csv"

# === Load base/enriched data ===
print("📥 Loading data...")
df = pd.read_csv(INPUT_CSV)
df['date'] = pd.to_datetime(df['date'])

# === Encode circuit names ===
print("🔤 Encoding circuit names...")
circuit_encoder = LabelEncoder()
df['circuit_encoded'] = circuit_encoder.fit_transform(df['circuit'])

# === Driver and Constructor Form ===
print("📊 Calculating driver and constructor form...")
df = df.sort_values(by=['driver', 'date'])
df['driver_form'] = (
    df.groupby('driver')['position']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

df = df.sort_values(by=['constructor', 'date'])
df['constructor_form'] = (
    df.groupby('constructor')['position']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

# === Grid Advantage (lower is better) ===
df['grid_advantage'] = df['grid'] - df['position']

# === Save final feature set ===
print("💾 Saving with engineered features...")
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
