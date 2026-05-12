from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="NIDS Prediction API")

# Load models
rf_model = joblib.load("../Model/rf_model.pkl")
xgb_model = joblib.load("../Model/xgb_model.pkl")
label_encoder = joblib.load("../Model/label_encoder.pkl")
feature_columns = joblib.load("../Model/feature_columns.pkl")
# feature_columns = 78 #For testing


class InputData(BaseModel):
    features: list


# Validate feature length
def validate_input(features):
    if len(features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Expected len(feature_columns) features, got {len(features)}"
        )


@app.get("/")
def home():
    return {
        "message": "NIDS API running",
        "expected_features": len(feature_columns)
    }


@app.post("/predict/rf")
def predict_rf(data: InputData):

    validate_input(data.features)

    x = np.array(data.features).reshape(1, -1)
    pred = rf_model.predict(x)

    return {
        "prediction": label_encoder.inverse_transform(pred)[0]
    }


@app.post("/predict/xgb")
def predict_xgb(data: InputData):

    validate_input(data.features)

    x = np.array(data.features).reshape(1, -1)
    pred = xgb_model.predict(x)

    return {
        "prediction": label_encoder.inverse_transform(pred)[0]
    }