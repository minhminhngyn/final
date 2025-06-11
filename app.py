# Thêm giao diện Streamlit cho pipeline dự đoán giao dịch gian lận
import streamlit as st

# Gọi pipeline bên trên
from fraud_model import run_pipeline, HybridGATGCN, load_data, create_graph

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("🔍 Fraud Detection on E-commerce Transactions")

st.markdown("""
#### Ứng dụng mô hình GAT-GCN lai để dự đoán giao dịch gian lận
Tải dữ liệu đầu vào và xem kết quả phân tích đặc trưng và độ chính xác mô hình.
""")

uploaded_file = st.file_uploader("Tải file CSV dữ liệu giao dịch:", type=["csv"])

if uploaded_file is not None:
    with open("user_uploaded_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🚀 Đang chạy mô hình và phân tích... Vui lòng chờ..."):
        model, importance_df = run_pipeline("user_uploaded_data.csv", train_new=True)

    st.success("✅ Phân tích hoàn tất!")

    st.subheader("🎯 Top 10 Đặc trưng Quan trọng (SHAP hoặc Permutation)")
    st.dataframe(importance_df.head(10))

    st.subheader("📈 Kết quả Dự đoán")
    st.markdown("""
    - **Accuracy:** %.4f  
    - **F1 Score:** %.4f  
    - **Recall:** %.4f  
    - **ROC-AUC:** %.4f
    """ % (
        evaluate(model, create_graph(*load_data("user_uploaded_data.csv"))[0],
                  create_graph(*load_data("user_uploaded_data.csv"))[0].test_mask)[i]
        for i in [0, 4, 5, 1]  # Acc, F1, Recall, ROC-AUC
    ))

    st.subheader("🖼️ Biểu đồ Độ Quan Trọng Đặc Trưng")
    st.image("shap_summary.png")
    st.image("shap_bar.png")
else:
    st.info("Vui lòng tải lên tập dữ liệu để bắt đầu.")
