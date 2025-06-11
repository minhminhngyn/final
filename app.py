import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import scipy.io
import os

# ======================== MODEL ========================
class HybridGATGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, dropout=0.27):
        super(HybridGATGCN, self).__init__()
        self.dropout = dropout

        # Layer 1
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Layer 2
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)

        # Layer 2
        x = torch.nn.functional.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return torch.nn.functional.log_softmax(x, dim=1)
# ======================== STREAMLIT UI ========================
st.set_page_config(page_title="Batch Transaction Detection with GAT-GCN", layout="wide")
st.title("📊 Batch Transaction Anomaly Detection")

uploaded_mat = st.file_uploader("Upload your .mat file (containing 'features' and 'label')", type=["mat"])

if uploaded_mat is not None:
    st.success("📁 File uploaded successfully.")

    if st.button("🔍 Analyze"):
        with open("temp_data.mat", "wb") as f:
            f.write(uploaded_mat.read())
def load_test_data(file_path="temp_data.mat"):
    try:
        mat = scipy.io.loadmat(file_path)
        st.write("📂 Biến trong file .mat:", list(mat.keys()))

        if "features" not in mat:
            raise KeyError("Không tìm thấy 'features' trong file .mat")

        features = mat["features"]

        if "label" in mat:
            labels = mat["label"].ravel()
            st.info("✅ Đã tìm thấy nhãn 'label'.")
        else:
            st.warning("⚠️ Không tìm thấy nhãn 'label'. Gán mặc định tất cả là 0.")
            labels = np.zeros(features.shape[0], dtype=int)  # giả định không có gian lận

        return features, labels
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu từ file .mat: {e}")
        raise
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try:
        features, labels = load_test_data()
        print(f"Loaded test data: {features.shape[0]} samples, {features.shape[1]} features")
    except FileNotFoundError:
        print("Preprocessed data file not found. Please run the preprocessing step first.")
        return

        model_path = "trained_model.pt"
    try:
        num_classes = 2 # Assuming binary classification
        model = HybridGATGCN(
            in_dim=features.shape[1],  # Update in_dim to features.shape[1]
            hidden_dim=256,  # Correct hidden dimension
            out_dim=num_classes,
            heads=8,  # Correct heads
            dropout=0.27  # Correct dropout
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Không tìm thấy tệp mô hình {model_path}. Sử dụng mô hình chưa huấn luyện.")
        num_classes = len(np.unique(labels))
        model = HybridGATGCN(
            in_dim=features.shape[1],
            hidden_dim=128,
            out_dim=num_classes,
            heads=8,
            dropout=0.27
        ).to(device)

    results = evaluate_model(model, data)
    class_names = [f'Class {i}' for i in range(results['confusion_matrix'].shape[0])]
    visualize_results(results, class_names)

    results_df = pd.DataFrame({
        'predicted_label': results['predictions']
    })
# Nếu có nhãn thực sự
    if results['true_labels'] is not None and len(set(results['true_labels'])) > 1:
        results_df['true_label'] = results['true_labels']

    for i in range(results['probabilities'].shape[1]):
        results_df[f'prob_class_{i}'] = results['probabilities'][:, i]

    # Lưu file
    results_df.to_csv('test_predictions.csv', index=False)

    # ✅ Gọi Streamlit UI hiển thị
    st.success("✅ Prediction Completed")
    st.dataframe(results_df.head(20))

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction Results", data=csv, file_name="prediction_results.csv", mime='text/csv')

# Gọi main trong Streamlit
if __name__ == "__main__":
    main()
