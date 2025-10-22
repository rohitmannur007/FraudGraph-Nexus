import streamlit as st
import coremltools as ct
import numpy as np
import json

st.title("Fraud Engine CoreML Demo (On M2)")

model = ct.models.MLModel('fraud_detector.mlmodel')

amt = st.slider("Transaction Amount", 0.0, 10000.0, 100.0)
if st.button("Score Fraud"):
    feats = np.array([[np.log1p(amt), 0.5, 0.0]*10], dtype=np.float32)  # Pad feats
    prediction = model.predict({'features': feats})
    score = prediction['variable'].item()  # Adapt to output name
    st.write(f"Fraud Score: {score:.4f}")
    st.balloons() if score > 0.5 else None

# Load sample
with open('coreml_demo/sample_input.json', 'r') as f:
    sample = json.load(f)
st.json(sample)