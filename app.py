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
    st.button("🔍 Analyze")
def load_test_data(file_path="temp_data.mat"):
    """
    Tải dữ liệu kiểm thử từ file CSV
    """
    try:
        df = pd.read_csv(file_path)
        if 'label' not in df.columns:
            raise ValueError("Không tìm thấy cột 'label' trong dữ liệu")

        labels = df.pop('label').values
        features = df.values

        return features, labels
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        raise
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   try:
        features, labels = load_test_data(test_file_path)
    except FileNotFoundError:
        print("Preprocessed data file not found. Please run the preprocessing step first.")
        return
    data = create_test_graph(features, labels)
    data = data.to(device)
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
        'true_label': results['true_labels'],
        'predicted_label': results['predictions']
    })
    if results['true_labels'] is not None and len(set(results['true_labels'])) > 1:
        results_df['true_label'] = results['true_labels']
    for i in range(results['probabilities'].shape[1]):
        results_df[f'prob_class_{i}'] = results['probabilities'][:, i]
    results_df.to_csv('test_predictions.csv', index=False)
    st.success("✅ Prediction Completed")
    st.dataframe(results_df.head(20))
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction Results", data=csv, file_name="prediction_results.csv", mime='text/csv')
if __name__ == "__main__":
    main()
