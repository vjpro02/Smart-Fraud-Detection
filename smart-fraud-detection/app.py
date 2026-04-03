import streamlit as st
import joblib
import pandas as pd

st.title("Smart Fraud Detection System")

model = joblib.load("models/fraud_model.pkl")

features = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

inputs = {}
for f in features:
    inputs[f] = st.number_input(f, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    if pred == 1:
        st.error("Fraud Detected")
    else:
        st.success("Genuine Transaction")
