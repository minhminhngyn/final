# model.py (updated for unlabeled CSV input -> .mat processing -> CSV output)
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.neighbors import NearestNeighbors

# Step 1: Load CSV (no labels assumed)
def load_csv_and_convert_to_mat(csv_path, mat_path="converted_input.mat"):
    df = pd.read_csv(csv_path)
    features = df.values.astype(np.float32)
    scipy.io.savemat(mat_path, {"features": features})
    return features

# Step 2: Load .mat file for internal pipeline
def load_mat_features(mat_path):
    mat = scipy.io.loadmat(mat_path)
    features = mat.get("features")
    return features

# Step 3: Autoencoder for feature extraction
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(features, encoding_dim=10):
    model = Autoencoder(features.shape[1], encoding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    data = torch.FloatTensor(features)
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(data), data)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model.encoder(data).numpy()

# Step 4: Preprocess data

def preprocess_data(features):
    features = KNNImputer().fit_transform(features)
    features = FunctionTransformer(np.log1p).fit_transform(np.clip(features, 1e-6, None))
    features = RobustScaler(quantile_range=(5, 95)).fit_transform(features)
    features = train_autoencoder(features, encoding_dim=16)
    return features

# Step 5: Create graph for GNN

def create_graph(features, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    edge_index = torch.tensor(np.array(nbrs.kneighbors_graph(mode="connectivity").nonzero()), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index)

# Step 6: GAT-GCN hybrid model

class HybridGATGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.bn2(self.gcn2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat2(x, edge_index)

# Step 7: Detect anomalies by distance from center

def detect_anomalies(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        center = z.mean(dim=0)
        scores = torch.norm(z - center, dim=1)
        threshold = torch.quantile(scores, 0.95)
        preds = (scores > threshold).int().numpy()
    return preds, scores.numpy()

# Step 8: Full pipeline

def run_pipeline_from_csv(csv_path, output_csv_path="anomalies_output.csv"):
    features = load_csv_and_convert_to_mat(csv_path)
    features = preprocess_data(features)
    data = create_graph(features)
    model = HybridGATGCN(in_dim=features.shape[1], hidden_dim=128, out_dim=32)
    model = train(model, data, epochs=30)
    preds, scores = detect_anomalies(model, data)

    # Save output CSV
    df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(features.shape[1])])
    df["anomaly_score"] = scores
    df["is_anomaly"] = preds
    df.to_csv(output_csv_path, index=False)
    print(f"Saved anomaly results to {output_csv_path}")
    return output_csv_path

def train(model, data, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.mean(torch.norm(out - out.mean(dim=0), dim=1))  # simple representation loss
        loss.backward()
        optimizer.step()
    return model
