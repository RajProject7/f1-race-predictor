import requests
import pandas as pd
from tqdm import tqdm

# ------------------------
# Generate 2025 F1 Data
# ------------------------

def fetch_2025_race_data(upto_round=7):
    all_data = []

    for rnd in tqdm(range(1, upto_round + 1)):
        url = f"https://ergast.com/api/f1/2025/{rnd}/results.json?limit=100"
        resp = requests.get(url)
        data = resp.json()

        try:
            race = data['MRData']['RaceTable']['Races'][0]
            date = race['date']
            circuit = race['Circuit']['circuitName']
            season = int(race['season'])
            round_no = int(race['round'])

            for result in race['Results']:
                driver_name = f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
                constructor = result['Constructor']['name']
                position = int(result['position'])

                all_data.append({
                    'date': date,
                    'season': season,
                    'round': round_no,
                    'circuit': circuit,
                    'driver': driver_name,
                    'constructor': constructor,
                    'position': position
                })

        except (IndexError, KeyError):
            print(f"⚠️ Could not fetch data for round {rnd}.")
            continue

    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df_2025 = fetch_2025_race_data(upto_round=7)
    df_2025['date'] = pd.to_datetime(df_2025['date'])
    df_2025.to_csv("../data/f1_features_2025.csv", index=False)
    print("✅ Saved 2025 race data to 'f1_features_2025.csv'")
