import fastf1
import pandas as pd
import joblib
from datetime import datetime
from fastf1.ergast import Ergast

# Enable cache for FastF1
fastf1.Cache.enable_cache('../cache')

# Load trained model
model = joblib.load('../models/f1_position_model_2025.pkl')

# Configuration: set the season and round you want to simulate
SEASON = 2025
ROUND = 8  # Monaco GP

# Load qualifying session for weather and lap times
session_q = fastf1.get_session(SEASON, ROUND, 'Q')
session_q.load()

# Get race session for grid positions
session_r = fastf1.get_session(SEASON, ROUND, 'R')
session_r.load()

# ----------------------------
# Step 1: Build driver data
# ----------------------------
driver_lookup = session_r.results.set_index('Abbreviation')['FullName'].to_dict()

# Grid positions
grid_data = session_r.results[['Abbreviation', 'GridPosition']].copy()
grid_data['driver'] = grid_data['Abbreviation'].map(driver_lookup)
grid_data['grid'] = grid_data['GridPosition'].astype(int)
grid_data = grid_data[['driver', 'grid']]

# Best qualifying lap time per driver
laps_q = session_q.laps.pick_quicklaps()
best_laps = laps_q.groupby('Driver')['LapTime'].min().reset_index()
best_laps['qualifying_time'] = best_laps['LapTime'].dt.total_seconds()
best_laps['driver'] = best_laps['Driver'].map(driver_lookup)
best_laps = best_laps[['driver', 'qualifying_time']]

# Merge grid and quali
race_input = pd.merge(grid_data, best_laps, on='driver', how='left')

# ----------------------------
# Step 2: Weather Data
# ----------------------------
weather_row = session_q.weather_data.iloc[0]
race_input['air_temp'] = weather_row['AirTemp']
race_input['track_temp'] = weather_row['TrackTemp']
race_input['humidity'] = weather_row['Humidity']

# ----------------------------
# Step 3: Driver/Constructor Form
# ----------------------------
# Load historical feature set for form calculations
df_hist = pd.read_csv("../data/f1_features_2025_enriched.csv")
df_hist['date'] = pd.to_datetime(df_hist['date'])

# Get race date for filtering
date_row = df_hist[(df_hist['season'] == SEASON) & (df_hist['round'] == ROUND)]['date']
race_date = pd.to_datetime(date_row.values[0]) if not date_row.empty else datetime.today()

# Compute form features
driver_forms = []
constructor_forms = []

for _, row in race_input.iterrows():
    driver = row['driver']
    constructor = df_hist[df_hist['driver'] == driver]['constructor'].values[0] if not df_hist[df_hist['driver'] == driver].empty else None

    past_driver = df_hist[(df_hist['driver'] == driver) & (df_hist['date'] < race_date)].sort_values('date')
    past_constructor = df_hist[(df_hist['constructor'] == constructor) & (df_hist['date'] < race_date)].sort_values('date')

    driver_form = past_driver['position'].rolling(3).mean().iloc[-1] if len(past_driver) >= 3 else 10.0
    constructor_form = past_constructor['position'].rolling(3).mean().iloc[-1] if len(past_constructor) >= 3 else 10.0

    driver_forms.append(driver_form)
    constructor_forms.append(constructor_form)

race_input['driver_form'] = driver_forms
race_input['constructor_form'] = constructor_forms
race_input['grid_advantage'] = race_input['grid'] - race_input['driver_form']

# ----------------------------
# Step 4: Circuit Encoding
# ----------------------------
race_input['circuit_encoded'] = ROUND  # use round number as circuit encoding

# ----------------------------
# Step 5: Prediction
# ----------------------------
features = [
    'grid', 'driver_form', 'constructor_form',
    'circuit_encoded', 'grid_advantage', 'qualifying_time',
    'air_temp', 'track_temp', 'humidity'
]

X = race_input[features]
race_input['predicted_position'] = model.predict(X)

# Sort and assign simulated finish
race_sorted = race_input.sort_values('predicted_position').reset_index(drop=True)
race_sorted['simulated_finish'] = range(1, len(race_sorted) + 1)

# ----------------------------
# Final Output
# ----------------------------
print("\n\U0001F3C1 Monaco GP 2025 - Predicted Finishing Order:")
print(race_sorted[['simulated_finish', 'driver', 'grid', 'predicted_position']])
