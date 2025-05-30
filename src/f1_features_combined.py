import pandas as pd

# Load both enriched datasets
df_2024 = pd.read_csv('../data/f1_features_2024_with_features.csv')
df_2025 = pd.read_csv('../data/f1_features_2025_enriched.csv')

# Add season column if not already present
df_2024['season'] = 2024
df_2025['season'] = 2025

# Combine and sort by date
df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
df_combined = df_combined.sort_values(by='date')

# Save to combined file
df_combined.to_csv('../data/f1_features_combined.csv', index=False)
print("✅ Saved combined dataset to f1_features_combined.csv")
