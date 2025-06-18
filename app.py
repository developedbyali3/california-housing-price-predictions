import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Fixed column indices (based on standard California Housing dataset)
rooms_ix = 2       # "AveRooms"
pop_ix = 4         # "Population"

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X):
        rooms_per_person = X[:, rooms_ix] / (X[:, pop_ix] + 1e-5)  # avoid zero division
        return np.c_[X, rooms_per_person]

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your final pipeline (preprocessing + model)
final_pipeline = joblib.load("final_pipeline.pkl")

st.title("🏡 California Housing Price Predictor")

# Input sliders for all features
MedInc = st.slider("Median Income", 0.0, 20.0, 3.0)
HouseAge = st.slider("House Age", 1, 50, 10)
AveRooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 100, 35000, 1000)
AveOccup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# Prepare input DataFrame
input_df = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude]
})

if st.button("Predict"):
    pred = final_pipeline.predict(input_df)[0]
    st.success(f"Predicted Median House Value: ${pred * 100000:.2f}")
