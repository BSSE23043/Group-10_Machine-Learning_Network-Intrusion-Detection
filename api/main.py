from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="NIDS Prediction API")

# Load models
rf_model = joblib.load("../Model/rf_model.pkl")
xgb_model = joblib.load("../Model/xgb_model.pkl")
label_encoder = joblib.load("../Model/label_encoder.pkl")


class InputData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "NIDS API is running"}


@app.post("/predict/rf")
def predict_rf(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    pred = rf_model.predict(x)
    return {"prediction": label_encoder.inverse_transform(pred)[0]}


@app.post("/predict/xgb")
def predict_xgb(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    pred = xgb_model.predict(x)
    return {"prediction": label_encoder.inverse_transform(pred)[0]}