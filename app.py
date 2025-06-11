import streamlit as st
import pandas as pd
import torch
from your_model_module import load_data, create_graph, HybridGATGCN  # sửa đúng tên nếu khác

def predict_on_uploaded_file(uploaded_file, model_path="trained_model.pt"):
    # Lưu file tạm thời
    temp_csv_path = "uploaded_input.csv"
    with open(temp_csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load dữ liệu
    features, labels, feature_names = load_data(temp_csv_path)
    data, _ = create_graph(features, labels, feature_names=feature_names)

    # Load model
    device = torch.device("cpu")
    model = HybridGATGCN(
        in_dim=features.shape[1],
        hidden_dim=256,
        out_dim=len(set(labels)),
        heads=8,
        dropout=0.27
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dự đoán
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        prob = torch.exp(out).cpu().numpy()[:, 1] if out.shape[1] > 1 else torch.exp(out).cpu().numpy()[:, 0]

    # Kết quả
    df_result = pd.DataFrame(features, columns=feature_names)
    df_result['TrueLabel'] = labels
    df_result['PredictedLabel'] = pred
    df_result['FraudProbability'] = prob
    return df_result

# Giao diện Streamlit
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("🕵️ Fraud Detection Prediction")
st.write("Upload a **preprocessed CSV file** to get predictions from the Hybrid GAT-GCN model.")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    with st.spinner("Predicting..."):
        df_result = predict_on_uploaded_file(uploaded_file)
    st.subheader("🔍 Prediction Results (Top 20)")
    st.dataframe(df_result.head(20))

    # Tải về kết quả
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Predictions",
        data=csv,
        file_name='fraud_predictions.csv',
        mime='text/csv',
    )
