# Updated universal simulate_race.py
import pandas as pd
import requests
import joblib
from tqdm import tqdm

# ---------------------- CONFIG ----------------------
SEASON = 2025
ROUND = 8  # Monaco GP

# ----------------------------------------------------

# Load trained model
model = joblib.load('../models/f1_position_model.pkl')

# Load historical dataset
df_hist = pd.read_csv('../data/f1_features_combined.csv')  # Update this to include 2025 if needed

# Parse date
df_hist['date'] = pd.to_datetime(df_hist['date'])

# Encode circuit automatically based on name
def encode_circuit(circuit_name, full_df):
    circuits = full_df['circuit'].astype('category').cat.categories.tolist()
    return circuits.index(circuit_name) if circuit_name in circuits else -1

# Convert quali time to seconds
def time_to_seconds(timestr):
    if timestr is None:
        return None
    try:
        mins, secs = timestr.split(':')
        return int(mins) * 60 + float(secs)
    except:
        return None

# Get race input data from Ergast

def get_race_input(season, round_no, circuit_code):
    url_results = f"https://ergast.com/api/f1/{season}/{round_no}/results.json?limit=100"
    url_quali = f"https://ergast.com/api/f1/{season}/{round_no}/qualifying.json?limit=100"

    results = requests.get(url_results).json()
    quali = requests.get(url_quali).json()

    race = results['MRData']['RaceTable']['Races'][0]
    quali_results = quali['MRData']['RaceTable']['Races'][0]['QualifyingResults']

    data = []
    for r in race['Results']:
        driver_id = r['Driver']['driverId']
        constructor_name = r['Constructor']['name']
        driver_name = f"{r['Driver']['givenName']} {r['Driver']['familyName']}"
        grid = int(r['grid'])

        # Get best qualifying time
        q_entry = next((q for q in quali_results if q['Driver']['driverId'] == driver_id), {})
        q1 = time_to_seconds(q_entry.get('Q1'))
        q2 = time_to_seconds(q_entry.get('Q2'))
        q3 = time_to_seconds(q_entry.get('Q3'))

        valid_times = [t for t in [q1, q2, q3] if t is not None]
        best_time = min(valid_times) if valid_times else None

        data.append({
            'driver': driver_name,
            'driver_id': driver_id,
            'constructor': constructor_name,
            'grid': grid,
            'qualifying_time': best_time,
            'circuit_encoded': circuit_code
        })

    return pd.DataFrame(data)

# Compute form based only on same season

def compute_form(df_input, race_date, df_hist_season):
    forms = []

    for _, row in df_input.iterrows():
        driver = row['driver']
        constructor = row['constructor']

        driver_past = df_hist_season[(df_hist_season['driver'] == driver) & (df_hist_season['date'] < race_date)].sort_values('date')
        constructor_past = df_hist_season[(df_hist_season['constructor'] == constructor) & (df_hist_season['date'] < race_date)].sort_values('date')

        driver_form = driver_past['position'].rolling(3).mean().iloc[-1] if len(driver_past) >= 3 else None
        constructor_form = constructor_past['position'].rolling(3).mean().iloc[-1] if len(constructor_past) >= 3 else None

        forms.append((driver_form, constructor_form))

    df_input['driver_form'] = [f[0] for f in forms]
    df_input['constructor_form'] = [f[1] for f in forms]

    # Fill missing values
    df_input['driver_form'].fillna(10.0, inplace=True)
    df_input['constructor_form'].fillna(10.0, inplace=True)

    fallback_time = df_input['qualifying_time'].max()
    fallback_time = fallback_time + 2 if pd.notnull(fallback_time) else 100.0
    df_input['qualifying_time'].fillna(fallback_time, inplace=True)

    # Recalculate grid advantage
    df_input['grid_advantage'] = df_input['grid'] - df_input['driver_form']

    return df_input

# Predict positions and assign unique finish order

def predict_race(df_features):
    X = df_features[[
        'grid', 'driver_form', 'constructor_form',
        'circuit_encoded', 'grid_advantage', 'qualifying_time'
    ]]

    df_features['predicted_position'] = model.predict(X)
    df_sorted = df_features.sort_values('predicted_position')
    df_sorted['simulated_finish'] = range(1, len(df_sorted) + 1)

    return df_sorted

# -------------- MAIN ----------------------

if __name__ == "__main__":
    # Get race date and circuit
    race_row = df_hist[(df_hist['season'] == SEASON) & (df_hist['round'] == ROUND)]
    race_date = pd.to_datetime(race_row['date'].values[0])
    circuit_name = race_row['circuit'].values[0]
    circuit_code = encode_circuit(circuit_name, df_hist)

    # Filter past races only for the same season
    df_hist_season = df_hist[(df_hist['season'] == SEASON) & (df_hist['date'] < race_date)]

    # Get driver input and compute form
    race_df = get_race_input(SEASON, ROUND, circuit_code)
    race_df = compute_form(race_df, race_date, df_hist_season)
    result_df = predict_race(race_df)

    print(result_df[['simulated_finish', 'driver', 'grid', 'predicted_position']])
