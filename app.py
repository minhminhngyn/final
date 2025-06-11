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
class HybridGATGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2, dropout=0.5):
        super(HybridGATGCN, self).__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

def create_graph(features, labels, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
    knn_adj = nbrs.kneighbors_graph(features, mode='connectivity')
    edge_index = np.array(knn_adj.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    num_nodes = x_tensor.shape[0]
    train_ratio, val_ratio = 0.7, 0.15
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)

    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# ======================== STREAMLIT UI ========================
st.set_page_config(page_title="Batch Transaction Detection with GAT-GCN", layout="wide")
st.title("📊 Batch Transaction Anomaly Detection")

uploaded_mat = st.file_uploader("Upload your .mat file (containing 'features' and 'label')", type=["mat"])

if uploaded_mat is not None:
    st.success("📁 File uploaded successfully.")

    if st.button("🔍 Analyze"):
        with open("temp_data.mat", "wb") as f:
            f.write(uploaded_mat.read())

        mat = scipy.io.loadmat("temp_data.mat")
        features = mat["features"]
        labels = mat["label"].ravel()
        feature_names = [f"feat_{i}" for i in range(features.shape[1])]
        data = create_graph(features, labels)

        model = HybridGATGCN(
            in_dim=features.shape[1],
            hidden_dim=256,
            out_dim=len(np.unique(labels)),
            heads=8,
            dropout=0.27
        )

        model.load_state_dict(torch.load("trained_model.pt", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1).numpy()

        df_result = pd.DataFrame(features, columns=feature_names)
        df_result['True Label'] = labels
        df_result['Predicted Label'] = preds

        st.success("✅ Prediction Completed")
        st.dataframe(df_result.head(20))

        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction Results", data=csv, file_name="prediction_results.csv", mime='text/csv')
