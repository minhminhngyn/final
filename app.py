import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error
import warnings
import shap
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# ========================
# Define HybridGATGCN model
# ========================
class HybridGATGCN(nn.Module):
    """Mô hình lai giữa GCN và GAT"""
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2, dropout=0.5):
        super(HybridGATGCN, self).__init__()
        self.dropout = dropout

        # Lớp 1: GCN + BatchNorm + GAT
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Lớp 2: GCN + BatchNorm + GAT
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Lớp 1
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)

        # Lớp 2
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)

def augment_graph(data, noise_level=0.1, drop_edge_rate=0.2):
    """Tăng cường dữ liệu đồ thị bằng cách thêm nhiễu và loại bỏ cạnh"""
    # Thêm nhiễu Gaussian vào đặc trưng
    x_aug = data.x + noise_level * torch.randn_like(data.x)

    # Loại bỏ ngẫu nhiên một số cạnh
    edge_index = data.edge_index.clone()
    num_edges = edge_index.shape[1]
    num_drop = int(drop_edge_rate * num_edges)

    if num_drop > 0:
        perm = torch.randperm(num_edges, device=edge_index.device)
        keep_edges = perm[num_drop:]
        edge_index_aug = edge_index[:, keep_edges]
    else:
        edge_index_aug = edge_index

    return Data(x=x_aug, edge_index=edge_index_aug, y=data.y,
                train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask)

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Batch Transaction Detection with GAT-GCN", layout="wide")
st.title("📊 Batch Transaction Anomaly Detection")

uploaded_mat = st.file_uploader("Upload your .mat file (containing 'features' and 'label')", type=["mat"])

if uploaded_mat is not None:
   def load_data(file_path):
    """Tải dữ liệu từ file CSV"""
    df = pd.read_csv(file_path)
    labels = df.pop('label').values
    feature_names = df.columns.tolist()
    features = df.values
    return features, labels, feature_names

def create_graph(features, labels, n_neighbors=5, feature_names=None):
    """Tạo đồ thị từ dữ liệu dạng bảng sử dụng k-NN"""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
    knn_adj = nbrs.kneighbors_graph(features, mode='connectivity')
    edge_index = np.array(knn_adj.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Phân chia tập dữ liệu
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

    # Tạo tên đặc trưng nếu không được cung cấp
    if feature_names is None:
        feature_names = [f'feat_{i}' for i in range(features.shape[1])]

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask), feature_names

        # Load model
    model = HybridGATGCN(
        in_dim=features.shape[1],
        hidden_dim=256,
        out_dim=len(np.unique(labels)),
        heads=8,
        dropout=0.27)

    if model_path and not train_new:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print("Training new model...")
            model, _ = train(model, data, epochs=50, lr=0.01, alpha=0.1, weight_decay=5e-4)
            torch.save(model.state_dict(), "trained_model.pt")
    else:
        print("Training new model...")
        model, _ = train(model, data, epochs=50, lr=0.01, alpha=0.1, weight_decay=5e-4)
        torch.save(model.state_dict(), "trained_model.pt")

        df_result = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(features.shape[1])])
        df_result['True Label'] = labels
        df_result['Predicted Label'] = preds
        st.success("✅ Prediction Completed")
        st.dataframe(df_result.head(20))

        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction Results", data=csv, file_name="prediction_results.csv", mime='text/csv')
