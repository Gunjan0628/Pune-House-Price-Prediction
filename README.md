# Pune House Price Prediction

An end-to-end Machine Learning project to estimate residential property prices in Pune, Maharashtra, using property characteristics, locality information, and available amenities.

The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model training and tuning, a FastAPI prediction backend, and a Streamlit user interface.

## Project Overview

Property prices depend on several factors, including area, number of bedrooms, locality, property age, and available amenities. This project explores these factors and builds a regression model to estimate property prices from user-provided inputs.

**Objective:** Build a practical, end-to-end machine learning application that demonstrates the complete workflow from raw data processing to model deployment interfaces.

## Key Features

- Data cleaning and preprocessing
- Exploratory Data Analysis and data visualization
- Feature engineering and amenity handling
- Regression model comparison and hyperparameter tuning
- Extra Trees Regressor with a Scikit-learn preprocessing pipeline
- Saved model for reuse during prediction
- FastAPI backend for prediction requests
- Streamlit interface for interactive property price estimation

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model persistence | Joblib |
| API backend | FastAPI, Uvicorn |
| User interface | Streamlit |
| Development | Jupyter Notebook, Git, GitHub |

## Machine Learning Workflow

1. **Data preparation:** Load and inspect the property dataset, validate columns, and handle data-quality issues.
2. **Exploratory Data Analysis:** Investigate price distributions, property characteristics, and relationships between features.
3. **Feature engineering:** Prepare numerical and categorical features, including property details and amenities.
4. **Model development:** Compare regression models and tune the selected model's hyperparameters.
5. **Evaluation:** Measure performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
6. **Prediction application:** Load the saved model and use it to generate predictions through the FastAPI backend and Streamlit frontend.

## Model Evaluation

The saved Extra Trees Regressor was evaluated on a 5,540-row test set using the `pune_cleaned_reference.csv` dataset and an 80/20 split with `random_state=42`.

| Metric | Verified result |
|---|---:|
| Model | Extra Trees Regressor |
| Valid dataset rows | 27,700 |
| Features | 50 |
| Training samples | 22,160 |
| Testing samples | 5,540 |
| MAE | ₹19.23 lakh |
| RMSE | ₹48.13 lakh |
| R² | 0.929541 |

**Metric interpretation**

- **MAE:** The average absolute prediction error was approximately ₹19.23 lakh.
- **RMSE:** Larger prediction errors receive more weight; the RMSE was approximately ₹48.13 lakh.
- **R²:** The model explained approximately 92.95% of the target-price variance on this test split. This is not the same as 92.95% prediction accuracy.

These results describe the evaluated test split. Performance on new properties may differ, and the error values indicate that predictions can be substantially off. The results should not be interpreted as a guarantee of market value.

## Project Structure

```text
Pune-House-Price-Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering_model_training.ipynb
│   ├── 04_model_tuning.ipynb
│   └── 05_model_saving.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/
│   ├── best_house_price_model.joblib
│   ├── model_features.json
│   ├── model_report.json
│   ├── final_model_report.json
│   └── tuning_report.json
├── reports/
├── visualizations/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Gunjan0628/Pune-House-Price-Prediction.git
cd Pune-House-Price-Prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```powershell
uvicorn main:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

### 5. Start the Streamlit application

Open a second terminal, activate the virtual environment, and run:

```powershell
streamlit run app.py
```

**Prerequisites:** The required processed dataset, model configuration, and saved model must be present at the paths expected by the application. The commands above assume the entry points and dependency names remain as configured in the repository.

## Limitations

- Predictions depend on the quality and coverage of the available dataset.
- The model may perform differently for localities or property types that are underrepresented in the training data.
- Market prices change over time; the model does not automatically reflect current market conditions.
- A prediction is an estimate, not a professional property valuation or a guaranteed selling price.
- Evaluation on a random holdout set does not by itself establish performance on future market data.

## Future Improvements

- Validate performance on a separate, newer dataset.
- Investigate unusually large prediction errors and outliers.
- Add prediction intervals or other uncertainty estimates.
- Improve input validation and error handling in the API.
- Add model monitoring and a documented model-retraining process.
- Include application screenshots and a live demo, if deployed.

## Author

**Gunjan Jadhav**

GitHub: [Gunjan0628](https://github.com/Gunjan0628)

---

*This project was developed for learning and portfolio demonstration purposes.*
