import streamlit as st
import pandas as pd
import torch
from io import BytesIO
from tempfile import NamedTemporaryFile
from model import run_pipeline, load_data, create_graph, HybridGATGCN

st.set_page_config(page_title="Fraud Detection GNN", layout="wide")
st.title("🚨 Fraud Transaction Detection with GNN")

st.markdown("Upload a CSV file containing transaction data (including a 'label' column). The model will predict fraudulent transactions.")

uploaded_file = st.file_uploader("Upload MAT", type="mat")

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mat") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.success("✅ File uploaded successfully. Running the model...")

    # Chạy pipeline
    model, importance_df = run_pipeline(tmp_path, model_path="trained_model.pt", train_new=False)

    # Dự đoán kết quả đầu ra
    features, labels, feature_names = load_data(tmp_path)
    data, _ = create_graph(features, labels)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()

    df_result = pd.read_csv(tmp_path)
    df_result['Predicted_Label'] = pred

    st.markdown("### 📊 Prediction Results")
    st.dataframe(df_result.head(30))

    # Tải kết quả về
    csv_output = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as MAT",
        data=csv_output,
        file_name='predicted_results.csv',
        mime='text/csv'
    )

    # Hiển thị kết quả phân tích độ quan trọng đặc trưng (nếu có)
    if importance_df is not None:
        st.markdown("### 🔍 Top 10 Important Features")
        st.dataframe(importance_df.head(10))

        st.markdown("#### 📈 Feature Importance Bar Chart")
        st.bar_chart(importance_df.set_index('Feature').head(10))
else:
    st.info("👈 Please upload a CSV file to begin.")
