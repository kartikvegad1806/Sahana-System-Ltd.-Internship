# test_model.py

import joblib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_model.pkl")

# =============================================================================
# LOAD MODEL
# =============================================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found. Run Decision_Tree.py first.")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.\n")

# =============================================================================
# RECREATE ENCODERS
# =============================================================================

weather_encoder = LabelEncoder()
weather_encoder.fit(["Sunny", "Overcast", "Rainy"])

humidity_encoder = LabelEncoder()
humidity_encoder.fit(["High", "Normal"])

wind_encoder = LabelEncoder()
wind_encoder.fit(["Weak", "Strong"])

target_encoder = LabelEncoder()
target_encoder.fit(["No", "Yes"])

# =============================================================================
# USER INPUT
# =============================================================================

print("Enter the following details:")

weather = input("Weather (Sunny/Overcast/Rainy): ").strip().capitalize()
humidity = input("Humidity (High/Normal): ").strip().capitalize()
wind = input("Wind (Weak/Strong): ").strip().capitalize()


# Encode input
try:
    encoded_input = pd.DataFrame({
        "Weather": [weather_encoder.transform([weather])[0]],
        "Humidity": [humidity_encoder.transform([humidity])[0]],
        "Wind": [wind_encoder.transform([wind])[0]],
    })

except ValueError:
    print("\nInvalid input provided. Please use the allowed values exactly as shown.")
    exit()

# =============================================================================
# PREDICTION
# =============================================================================

prediction = model.predict(encoded_input)[0]
decoded_prediction = target_encoder.inverse_transform([prediction])[0]

print("\nPrediction:", decoded_prediction)
