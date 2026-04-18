from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn based on their profile",
    version="1.0.0"
)

MODEL_PATH = "../models/best_model.pkl"
METADATA_PATH = "../models/model_metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    tenure_group: str
    charges_per_tenure: float
    total_services: int

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    model_used: str

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "model": metadata["model_name"],
        "metrics": metadata["metrics"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if probability >= 0.7:
            risk = "High"
        elif probability >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=round(float(probability), 4),
            risk_level=risk,
            model_used=metadata["model_name"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return metadata