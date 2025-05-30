import joblib
import pandas as pd

# Load the trained model
model = joblib.load('../models/f1_best_model.pkl')

# Example: Realistic test input for Max Verstappen — Saudi GP 2024
live_input = {
    'grid': 2,
    'driver_form': 4.0,         # Assume finished P1 in 3 previous races
    'constructor_form': 3.0,    # Red Bull finishing ~P1.5 recently
    'circuit_encoded': 11,       # Based on your encoding for Jeddah
    'grid_advantage': -1.0,        # He's expected to win from pole
    'qualifying_time': 71.576   # 1:18.576 → in seconds
}

# Convert to DataFrame
X_live = pd.DataFrame([live_input])

# Predict class (podium = 1 or 0)
prediction = model.predict(X_live)[0]
proba = model.predict_proba(X_live)[0][1]

# Output
print(f"🎯 Predicted Podium Outcome: {'YES 🏆' if prediction == 1 else 'NO'}")
print(f"📊 Model Confidence: {proba:.2%}")
