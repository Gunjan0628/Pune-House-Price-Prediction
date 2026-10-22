
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    DATA_DIR / "pune_cleaned_reference_v2.csv",
    DATA_DIR / "pune_cleaned_reference.csv",
]

CONFIG_PATH = MODEL_DIR / "model_features.json"
TUNING_REPORT_PATH = MODEL_DIR / "tuning_report.json"

FINAL_MODEL_PATH = MODEL_DIR / "best_house_price_model.joblib"
FINAL_REPORT_PATH = MODEL_DIR / "final_model_report.json"


# --------------------------------------------------
# 2. LOAD DATASET AND CONFIGURATION
# --------------------------------------------------

DATA_PATH = next(
    (path for path in DATA_CANDIDATES if path.exists()),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "Cleaned dataset not found in data/processed/. "
        "Check the CSV filename and path."
    )

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        "model_features.json not found in the models folder."
    )

if not TUNING_REPORT_PATH.exists():
    raise FileNotFoundError(
        "tuning_report.json not found. Run Notebook 4 first."
    )

df = pd.read_csv(DATA_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

with open(TUNING_REPORT_PATH, "r", encoding="utf-8") as file:
    tuning_report = json.load(file)

FEATURES = config["features"]
TARGET = config.get("target", "price")

NUMERIC_COLUMNS = [
    col for col in config["numeric_columns"]
    if col in FEATURES
]

CATEGORICAL_COLUMNS = [
    col for col in config["categorical_columns"]
    if col in FEATURES
]

AMENITY_COLUMNS = [
    col for col in config.get("amenity_columns", [])
    if col in FEATURES
]

print("Dataset loaded:", DATA_PATH.name)
print("Original dataset shape:", df.shape)
print("Number of features:", len(FEATURES))


# --------------------------------------------------
# 3. VALIDATE AND PREPARE DATA
# --------------------------------------------------

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

df = df.loc[
    df[TARGET].notna() & (df[TARGET] > 0)
].copy()

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Numeric columns: convert invalid values to missing values.
for col in NUMERIC_COLUMNS:
    X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

# Categorical columns: replace missing values with np.nan.
for col in CATEGORICAL_COLUMNS:
    X[col] = (
        X[col]
        .astype("object")
        .where(X[col].notna(), np.nan)
    )

# Standardize amenity values consistently.
yes_values = {
    "yes", "y", "true", "1", "available"
}

no_values = {
    "no", "n", "false", "0", "not available",
    "unavailable"
}

for col in AMENITY_COLUMNS:
    values = X[col].astype("object")
    normalized = values.astype(str).str.strip().str.lower()

    X[col] = values

    X.loc[normalized.isin(yes_values), col] = "Yes"
    X.loc[normalized.isin(no_values), col] = "No"

    X[col] = X[col].where(X[col].notna(), np.nan)

print("Rows available for training:", len(X))
print("Features used:", len(FEATURES))


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 5. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Not specified"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
    ],
    remainder="drop"
)


# --------------------------------------------------
# 6. READ BEST MODEL AND PARAMETERS
# --------------------------------------------------

model_name = (
    tuning_report.get("selected_model")
    or tuning_report.get("best_model")
    or tuning_report.get("model_name")
)

if not model_name:
    raise ValueError(
        "Model name missing from tuning_report.json. "
        "Check the selected model field created by Notebook 4."
    )

model_name_lower = str(model_name).lower()

if "extra" in model_name_lower:
    model_class = ExtraTreesRegressor
    canonical_model_name = "ExtraTreesRegressor"

elif "random" in model_name_lower or "forest" in model_name_lower:
    model_class = RandomForestRegressor
    canonical_model_name = "RandomForestRegressor"

else:
    raise ValueError(
        f"Unsupported model name in tuning report: {model_name}"
    )

best_params = (
    tuning_report.get("best_parameters")
    or tuning_report.get("best_params")
    or {}
)

# Remove the Pipeline prefix if the parameters came from
# RandomizedSearchCV on a pipeline with a step named "model".
clean_params = {}

for key, value in best_params.items():
    if key.startswith("model__"):
        key = key.replace("model__", "", 1)

    if key in model_class().get_params():
        clean_params[key] = value

model = model_class(
    random_state=42,
    n_jobs=-1,
    **clean_params
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

print("Selected model:", canonical_model_name)
print("Parameters:", clean_params)


# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("------------------------------")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 9. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(pipeline, FINAL_MODEL_PATH)

report = {
    "model_name": canonical_model_name,
    "target": TARGET,
    "features": FEATURES,
    "numeric_columns": NUMERIC_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "amenity_columns": AMENITY_COLUMNS,
    "best_parameters": clean_params,
    "dataset_file": DATA_PATH.name,
    "rows_used": int(len(X)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "random_state": 42,
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    },
    "target_unit": config.get("target_unit", "INR")
}

with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, ensure_ascii=False)

print("\nSaved model:", FINAL_MODEL_PATH)
print("Saved report:", FINAL_REPORT_PATH)


# --------------------------------------------------
# 10. VERIFY SAVED MODEL
# --------------------------------------------------

loaded_model = joblib.load(FINAL_MODEL_PATH)

sample = X_test.iloc[[0]]
sample_prediction = loaded_model.predict(sample)[0]

print("\nMODEL VERIFICATION")
print("------------------------------")
print(f"Actual price    : ₹{y_test.iloc[0]:,.0f}")
print(f"Predicted price : ₹{sample_prediction:,.0f}")
print("\ntrain.py completed successfully!")
