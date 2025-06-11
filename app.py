# app.py - Streamlit interface for anomaly detection
import streamlit as st
import os
from model import run_pipeline_from_csv

st.set_page_config(page_title="Anomaly Detection with GAT-GCN", layout="centered")
st.title("🚀 Anomaly Detection in E-commerce Transactions")

st.markdown("""
Upload a CSV file **without labels**. The system will:
- Preprocess and extract features with autoencoder
- Build a graph with k-NN
- Detect anomalies using a hybrid GAT-GCN model
- Return a CSV with `anomaly_score` and `is_anomaly` columns
""")

uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    input_path = "input_temp.csv"
    output_path = "anomaly_output.csv"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing... this may take a minute.")
    result_csv = run_pipeline_from_csv(input_path, output_csv_path=output_path)

    st.success("Done! Download the result below:")
    with open(result_csv, "rb") as f:
        st.download_button("Download Anomaly Results CSV", f, file_name="anomalies_detected.csv")
