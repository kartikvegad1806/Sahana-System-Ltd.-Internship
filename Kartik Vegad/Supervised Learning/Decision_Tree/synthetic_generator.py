# synthetic_generator.py

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_base_tennis_dataset():
    data = {
        "Weather": [
            "Sunny","Sunny","Overcast","Rainy","Rainy","Rainy",
            "Overcast","Sunny","Sunny","Rainy","Sunny",
            "Overcast","Overcast","Rainy"
        ],
        "Humidity": [
            "High","High","High","High","Normal","Normal",
            "Normal","High","Normal","Normal","Normal",
            "High","Normal","High"
        ],
        "Wind": [
            "Weak","Strong","Weak","Weak","Weak","Strong",
            "Strong","Weak","Weak","Weak","Strong",
            "Strong","Weak","Strong"
        ],
        "Play": [
            "No","No","Yes","Yes","Yes","No",
            "Yes","No","Yes","Yes","Yes",
            "Yes","Yes","No"
        ]
    }

    return pd.DataFrame(data)


def generate_synthetic_data(n_samples=1000, save=True):
    print("Inside synthetic generator...")

    base_df = load_base_tennis_dataset()

    synthetic_df = base_df.sample(
        n=n_samples,
        replace=True,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("Generated shape:", synthetic_df.shape)

    if save:
        save_path = os.path.join(DATA_DIR, "synthetic_tennis_dataset.csv")
        print("Saving to:", save_path)
        synthetic_df.to_csv(save_path, index=False)

    return synthetic_df.drop(columns=["Play"]), synthetic_df["Play"]



    return X, y
