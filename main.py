
from typing import Any

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse

from src.predict import predict_price


# --------------------------------------------------
# 1. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Pune House Price Prediction API",
    description=(
        "API to estimate residential property prices "
        "in Pune using a trained machine learning model."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 2. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to Pune House Price Prediction API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Pune House Price Prediction API"
    }


# --------------------------------------------------
# 3. PRICE PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(
    property_data: dict[str, Any] = Body(
        ...,
        description=(
            "Property features as a JSON object. "
            "Use the exact feature names from model_features.json."
        )
    )
):
    if not property_data:
        raise HTTPException(
            status_code=400,
            detail="Property data cannot be empty."
        )

    try:
        result = predict_price(property_data)

        return {
            "success": True,
            "prediction": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error
