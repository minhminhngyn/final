# app.py
import streamlit as st
from model import run_pipeline
import tempfile
import os

st.set_page_config(page_title="Fraud Detection GNN", layout="centered")
st.title("🔍 Fraud Detection on E-commerce Transactions")
st.markdown("Upload a `.mat` file with transaction data to detect suspicious activities.")

uploaded_file = st.file_uploader("Upload .mat file", type=["mat"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Running model. This may take a moment...")
    try:
        output_path = run_pipeline(tmp_path)
        with open(output_path, "rb") as f:
            st.success("✅ Prediction complete! Download results below.")
            st.download_button(
                label="📥 Download Predicted Fraud Transactions (.mat)",
                data=f,
                file_name="fraud_predictions.mat",
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
    finally:
        os.remove(tmp_path)
