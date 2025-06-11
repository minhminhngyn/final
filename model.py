# model.py
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error

# Load .mat file (user uploaded)
def load_mat_file(mat_path):
    mat = scipy.io.loadmat(mat_path)
    mat_data = {}
    for key in mat:
        if not key.startswith('__'):
            data = mat[key]
            if hasattr(data, "toarray"):
                data = data.toarray()
            mat_data[key] = data

    features = mat_data.get("features", None)
    labels = mat_data.get("label", None)
    if labels is not None and labels.shape[0] == 1:
        labels = labels.ravel()
    return features, labels

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(features, encoding_dim=10, epochs=100, lr=0.001, patience=5):
    model = Autoencoder(features.shape[1], encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = torch.FloatTensor(features)
    best_loss, no_improve = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.eval()
    with torch.no_grad():
        return model.encoder(data).numpy()

def preprocess_data(features, labels, encoding_dim=10):
    features = KNNImputer(n_neighbors=5).fit_transform(features)
    features = FunctionTransformer(np.log1p).fit_transform(np.clip(features, 1e-6, None))
    features = RobustScaler(quantile_range=(5, 95)).fit_transform(features)
    features = train_autoencoder(features, encoding_dim)
    return features, labels

def create_graph(features, labels, k=5):
    edge_index = np.array(NearestNeighbors(n_neighbors=k).fit(features).kneighbors_graph(mode='connectivity').nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    num_nodes = x.shape[0]
    idx = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:int(0.7*num_nodes)]] = True
    val_mask[idx[int(0.7*num_nodes):int(0.85*num_nodes)]] = True
    test_mask[idx[int(0.85*num_nodes):]] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

class HybridGATGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=heads, concat=False, dropout=dropout)
        self.dropout = dropout

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
        return F.log_softmax(self.gat2(x, edge_index), dim=1)

def train(model, data, epochs=30, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

def predict_and_export(model, data, output_path="fraud_output.mat"):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        features = data.x.cpu().numpy()
        fraud_indices = np.where(preds == 1)[0]
        scipy.io.savemat(output_path, {
            "fraud_indices": fraud_indices,
            "fraud_features": features[fraud_indices],
            "all_predictions": preds
        })
    return output_path

def run_pipeline(mat_file):
    features, labels = load_mat_file(mat_file)
    features, labels = preprocess_data(features, labels)
    data = create_graph(features, labels)
    model = HybridGATGCN(in_dim=features.shape[1], hidden_dim=128, out_dim=len(np.unique(labels)))
    model = train(model, data)
    return predict_and_export(model, data)
