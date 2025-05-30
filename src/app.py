import fastf1
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Enable cache for FastF1
fastf1.Cache.enable_cache('../cache')

# Load trained model
model = joblib.load('../models/f1_position_model_2025.pkl')

# Full F1 2025 calendar with round numbers
rounds_map = {
    1: "Australia",
    2: "China",
    3: "Japan",
    4: "Bahrain",
    5: "Saudi Arabia",
    6: "Miami",
    7: "Emilia-Romagna",
    8: "Monaco",
    9: "Spain",
    10: "Canada",
    11: "Austria",
    12: "United Kingdom",
    13: "Belgium",
    14: "Hungary",
    15: "Netherlands",
    16: "Italy",
    17: "Azerbaijan",
    18: "Singapore",
    19: "USA",
    20: "Mexico",
    21: "Brazil",
    22: "Las Vegas",
    23: "Qatar",
    24: "Abu Dhabi"
}

# App title
st.title("\U0001F3C1 F1 Race Outcome Predictor - 2025 Season")

# Reverse map for dropdown
round_names = list(rounds_map.values())
selected_round_name = st.selectbox("Select a completed 2025 race round:", round_names, index=7)

# Get round number
ROUND = [k for k, v in rounds_map.items() if v == selected_round_name][0]
SEASON = 2025

# Simulate button
if st.button("\U0001F52E Simulate Race"):
    try:
        session_q = fastf1.get_session(SEASON, ROUND, 'Q')
        session_q.load()
        session_r = fastf1.get_session(SEASON, ROUND, 'R')
        session_r.load()
    except Exception:
        st.warning("⚠️ Data not available yet. Please select a race after its qualifying session is completed.")
        st.stop()

    # Driver info
    driver_lookup = session_r.results.set_index('Abbreviation')['FullName'].to_dict()

    # Grid
    grid_data = session_r.results[['Abbreviation', 'GridPosition']].copy()
    grid_data['driver'] = grid_data['Abbreviation'].map(driver_lookup)
    grid_data['grid'] = grid_data['GridPosition'].astype(int)
    grid_data = grid_data[['driver', 'grid']]

    # Quali times
    laps_q = session_q.laps.pick_quicklaps()
    best_laps = laps_q.groupby('Driver')['LapTime'].min().reset_index()
    best_laps['qualifying_time'] = best_laps['LapTime'].dt.total_seconds()
    best_laps['driver'] = best_laps['Driver'].map(driver_lookup)
    best_laps = best_laps[['driver', 'qualifying_time']]

    # Merge input
    race_input = pd.merge(grid_data, best_laps, on='driver', how='left')

    # Weather
    weather_row = session_q.weather_data.iloc[0]
    race_input['air_temp'] = weather_row['AirTemp']
    race_input['track_temp'] = weather_row['TrackTemp']
    race_input['humidity'] = weather_row['Humidity']

    # Add form features
    hist_df_2024 = pd.read_csv("../data/f1_features_2024_with_features.csv")
    hist_df_2025 = pd.read_csv("../data/f1_features_2025_enriched.csv")
    hist_df_2024['season'] = 2024
    hist_df_2025['season'] = 2025
    hist_df = pd.concat([hist_df_2024, hist_df_2025], ignore_index=True)
    hist_df['date'] = pd.to_datetime(hist_df['date'], format='mixed')

    date_row = hist_df[(hist_df['season'] == SEASON) & (hist_df['round'] == ROUND)]['date']
    race_date = pd.to_datetime(date_row.values[0]) if not date_row.empty else datetime.today()

    driver_forms = []
    constructor_forms = []

    for _, row in race_input.iterrows():
        driver = row['driver']
        constructor = hist_df[hist_df['driver'] == driver]['constructor'].values[0] if not hist_df[hist_df['driver'] == driver].empty else None
        past_driver = hist_df[(hist_df['driver'] == driver) & (hist_df['date'] < race_date)].sort_values('date')
        past_constructor = hist_df[(hist_df['constructor'] == constructor) & (hist_df['date'] < race_date)].sort_values('date')
        driver_form = past_driver['position'].rolling(3).mean().iloc[-1] if len(past_driver) >= 3 else 10.0
        constructor_form = past_constructor['position'].rolling(3).mean().iloc[-1] if len(past_constructor) >= 3 else 10.0
        driver_forms.append(driver_form)
        constructor_forms.append(constructor_form)

    race_input['driver_form'] = driver_forms
    race_input['constructor_form'] = constructor_forms
    race_input['grid_advantage'] = race_input['grid'] - race_input['driver_form']
    race_input['circuit_encoded'] = ROUND

    # Predict
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

    # Actual positions
    actual_positions = session_r.results[['Abbreviation', 'Position']].copy()
    actual_positions['driver'] = actual_positions['Abbreviation'].map(driver_lookup)
    actual_positions['actual_position'] = actual_positions['Position'].astype(int)
    actual_positions = actual_positions[['driver', 'actual_position']]

    # Merge and calculate error
    race_sorted = pd.merge(race_sorted, actual_positions, on='driver', how='left')
    race_sorted['error'] = race_sorted['simulated_finish'] - race_sorted['actual_position']

    # Display table
    race_name = session_r.event['EventName']
    st.subheader(f"\U0001F3C1 {race_name} GP 2025 - Predicted Finishing Order")
    st.dataframe(race_sorted[['simulated_finish', 'driver', 'grid', 'predicted_position', 'actual_position', 'error']])

    # 📊 Improved Plotting
    st.subheader("\U0001F4CA Prediction vs Actual")
    race_sorted = race_sorted.sort_values('actual_position')
    race_sorted['error_class'] = race_sorted['error'].apply(
        lambda x: 'Accurate (±1)' if abs(x) <= 1 else ('Close (±3)' if abs(x) <= 3 else 'Off (>3)')
    )
    palette = {
        'Accurate (±1)': 'green',
        'Close (±3)': 'orange',
        'Off (>3)': 'red'
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(
        x='predicted_position',
        y='driver',
        data=race_sorted,
        hue='error_class',
        dodge=False,
        palette=palette,
        ax=ax
    )
    ax.scatter(race_sorted['actual_position'], race_sorted['driver'], color='black', label='Actual', zorder=5)
    ax.set_title('Prediction vs Actual Finishing Positions', fontsize=14, weight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(fig)
