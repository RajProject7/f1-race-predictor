# 📦 simulate_race_2025.py
# Predict the finishing order for Monaco GP (Round 8, 2025)

import pandas as pd
import joblib
import fastf1
from datetime import datetime

# Load trained model
model = joblib.load("../models/f1_position_model_2025.pkl")

# Enable FastF1 cache
fastf1.Cache.enable_cache("../cache")

# === USER PARAMS ===
SEASON = 2025
ROUND = 8  # Monaco GP

# Load sessions
session_q = fastf1.get_session(SEASON, ROUND, 'Q')
session_q.load()

session_r = fastf1.get_session(SEASON, ROUND, 'R')
session_r.load()

# === Qualifying Times ===
laps_q = session_q.laps.pick_quicklaps()
best_laps = laps_q.groupby('Driver')['LapTime'].min().reset_index()
best_laps['qualifying_time'] = best_laps['LapTime'].dt.total_seconds()

# Map driver abbreviation to full name
driver_name_map = session_r.results.set_index('Abbreviation')['FullName'].to_dict()
best_laps['driver'] = best_laps['Driver'].map(driver_name_map)
best_laps = best_laps.dropna(subset=['driver'])
best_laps = best_laps[['driver', 'qualifying_time']]

# === Grid Positions from Race Results ===
grids = session_r.results[['Abbreviation', 'GridPosition']].copy()
grids['driver'] = grids['Abbreviation'].map(driver_name_map)
grids = grids.dropna(subset=['driver'])
grids = grids[['driver', 'GridPosition']].rename(columns={'GridPosition': 'grid'})

# === Merge Inputs ===
race_input = pd.merge(best_laps, grids, on='driver')
race_input['round'] = ROUND

# Add dummy weather
race_input['air_temp'] = 25.0
race_input['track_temp'] = 35.0
race_input['humidity'] = 50.0

# === Predict ===
features = ['grid', 'qualifying_time', 'air_temp', 'track_temp', 'humidity']
race_input['predicted_position'] = model.predict(race_input[features])
race_input = race_input.sort_values('predicted_position')
race_input['simulated_finish'] = range(1, len(race_input) + 1)

# Output
print("\n🏁 Monaco GP 2025 - Predicted Finishing Order:")
print(race_input[['simulated_finish', 'driver', 'grid', 'predicted_position']])
