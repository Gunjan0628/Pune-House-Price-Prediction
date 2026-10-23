
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_house_price_model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_features.json"


# --------------------------------------------------
# 2. LOAD MODEL AND FEATURE CONFIGURATION
# --------------------------------------------------

def load_model():
    """Load the trained model and feature configuration."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python -m src.train first."
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Feature configuration not found: {CONFIG_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, config


# --------------------------------------------------
# 3. PREPARE USER INPUT
# --------------------------------------------------

def prepare_input(property_data, config):
    """
    Convert property details into the exact feature format
    expected by the trained model.
    """

    features = config["features"]
    numeric_columns = [
        col for col in config.get("numeric_columns", [])
        if col in features
    ]
    categorical_columns = [
        col for col in config.get("categorical_columns", [])
        if col in features
    ]
    amenity_columns = [
        col for col in config.get("amenity_columns", [])
        if col in features
    ]

    # Reject unexpected input fields to catch spelling mistakes.
    unexpected = set(property_data) - set(features)

    if unexpected:
        raise ValueError(
            f"Unexpected input fields: {sorted(unexpected)}"
        )

    # Add every expected feature in the model's original order.
    row = {
        feature: property_data.get(feature, np.nan)
        for feature in features
    }

    input_df = pd.DataFrame([row], columns=features)

    # Convert numeric inputs safely.
    for col in numeric_columns:
        input_df[col] = pd.to_numeric(
            input_df[col], errors="coerce"
        ).astype(float)

    # Standardize amenity inputs.
    yes_values = {
        "yes", "y", "true", "1", "available"
    }
    no_values = {
        "no", "n", "false", "0", "not available",
        "unavailable"
    }

    for col in amenity_columns:
        if col not in input_df.columns:
            continue

        value = input_df.at[0, col]

        if pd.isna(value):
            input_df.at[0, col] = np.nan
            continue

        normalized = str(value).strip().lower()

        if normalized in yes_values:
            input_df.at[0, col] = "Yes"
        elif normalized in no_values:
            input_df.at[0, col] = "No"
        else:
            input_df.at[0, col] = str(value).strip()

    # Missing categorical inputs are left missing.
    # The saved model's imputer will handle them.
    for col in categorical_columns:
        input_df[col] = (
            input_df[col]
            .astype("object")
            .where(input_df[col].notna(), np.nan)
        )

    return input_df


# --------------------------------------------------
# 4. PREDICT PROPERTY PRICE
# --------------------------------------------------

def predict_price(property_data):
    """
    Predict the estimated property price in INR.

    property_data should be a dictionary whose keys match
    the feature names in model_features.json.
    """

    model, config = load_model()

    input_df = prepare_input(property_data, config)

    prediction = float(model.predict(input_df)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return {
        "predicted_price": round(prediction, 2),
        "predicted_price_in_lakhs": round(prediction / 100000, 2),
        "predicted_price_in_crores": round(prediction / 10000000, 3),
        "currency": "INR"
    }


# --------------------------------------------------
# 5. RUN A SAMPLE PREDICTION
# --------------------------------------------------

if __name__ == "__main__":

    # Replace these example values with your own property details.
    sample_property = {
        "Area": 1000,
        "Bedrooms": 2,
        "bathroom": 2,
        "balconies": 1,
        "floor": "2",
        "opensides": 2,
        "Age": "New",
        "addi_rooms": "Study Room",
        "facing": "East",
        "locality": "Wakad",
        "neworold": "New",
        "Residential": 1,
        "Villa": 0,
        "Independent": 0,
        "Penthouse": 0,
    }

    # Amenities are separate model features.
    # Add every amenity column that exists in your config.
    _, feature_config = load_model()

    for amenity in feature_config.get("amenity_columns", []):
        if amenity not in sample_property:
            sample_property[amenity] = "No"

    result = predict_price(sample_property)

    print("\nPUNE HOUSE PRICE PREDICTION")
    print("------------------------------")
    print(f"Estimated price: ₹{result['predicted_price']:,.0f}")
    print(f"In lakhs:       ₹{result['predicted_price_in_lakhs']} lakh")
    print(f"In crores:      ₹{result['predicted_price_in_crores']} crore")
