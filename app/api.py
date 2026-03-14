from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Literal

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a customer is likely to churn using a tuned LightGBM model.",
    version="1.0.0"
)

# ────────────────────────────────────────────────
# Load model once at startup (very efficient)
# ────────────────────────────────────────────────
MODEL_PATH = "models/churn_best_lightgbm_pipeline.joblib"

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ────────────────────────────────────────────────
# Pydantic model – matches Telco dataset columns exactly
# ────────────────────────────────────────────────
class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="0 = No, 1 = Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: float = Field(..., ge=0, le=72, description="Months with company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=18.25, le=118.75)
    TotalCharges: float = Field(..., ge=0.0, description="Can be 0 for new customers")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.6,
                "TotalCharges": 787.2
            }
        }

# ────────────────────────────────────────────────
# Prediction endpoint
# ────────────────────────────────────────────────
@app.post("/predict", response_model=dict)
async def predict_churn(customer: Customer):
    try:
        # Convert input to DataFrame (exactly what the pipeline expects)
        input_df = pd.DataFrame([customer.model_dump()])

        # Predict
        prob_churn = model_pipeline.predict_proba(input_df)[0][1]  # probability of class 1 (churn)
        pred_churn = model_pipeline.predict(input_df)[0]           # 0 or 1

        # Simple risk categories (you can tune these thresholds)
        if prob_churn >= 0.65:
            risk_level = "High"
            message = "High risk of churn – recommend immediate retention action."
        elif prob_churn >= 0.35:
            risk_level = "Medium"
            message = "Moderate risk – consider proactive outreach."
        else:
            risk_level = "Low"
            message = "Low risk of churn – customer likely to stay."

        return {
            "churn_probability": round(float(prob_churn), 4),
            "predicted_churn": bool(pred_churn),
            "risk_level": risk_level,
            "message": message,
            "model_version": "LightGBM-tuned-v1"
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")


# ────────────────────────────────────────────────
# Health check (useful for deployment / monitoring)
# ────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}