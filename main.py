from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Ignore warnings (optional)
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()


class PatientFeatures(BaseModel):
    PRG: float
    PL: float
    PR : float
    SK: float
    TS: float
    M11: float
    BD2: float
    AGE: float
    Insurance: float


gradient_boosting_pipeline = joblib.load("models/gradient_boosting_pipeline.joblib")
random_forest_pipeline = joblib.load("models/random_forest_pipeline.joblib")
encoder = joblib.load("models/encoder.joblib")


@app.get("/")
async def root():
    return {"Welcome to LP5": "fastapi for sepsis prediction"}


@app.post("/predict_random_forest")
def random_forest_prediction(data: PatientFeatures):
    df = pd.DataFrame([data.dict()])
    prediction = random_forest_pipeline.predict(df)
    prediction = int(prediction[0])
    prediction =encoder.inverse_transform([prediction])[0]
    probability =  random_forest_pipeline.predict_proba(df)
    probabilities =probability.tolist()
    return {'prediction': prediction,'probabilities':probabilities}


@app.post("/predict_gradient_boosting")
def gradient_boosting_predict(data: PatientFeatures):
    df = pd.DataFrame([data.dict()])
    prediction = gradient_boosting_pipeline.predict(df)
    prediction = int(prediction[0])
    prediction =encoder.inverse.transform([prediction])[0]
    probability =  gradient_boosting_pipeline.predict_proba(df)
    probabilities =probability.tolist()
    return{'prediction': prediction,'probabilities':probabilities}


@app.get('/documents')
def documentation():
    return{'description': 'Dcouments'}

            
            