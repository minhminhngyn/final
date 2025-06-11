
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# Define the HybridGATGCN model
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

# Streamlit App UI
st.set_page_config(page_title="Transaction Anomaly Detection System", layout="wide")
st.title("Transaction Anomaly Detection System (GAT-GCN Based)")

st.header("Required Transaction Information")
actual_price = st.number_input("Original Price (USD)", min_value=0.0)
discounted_price = st.number_input("Discounted Price (USD)", min_value=0.0)

discount_percentage = 0.0
if actual_price > 0:
    discount_percentage = round((actual_price - discounted_price) / actual_price * 100, 2)
    st.markdown(f"**Calculated Discount Percentage:** {discount_percentage}%")
else:
    st.warning("Please enter an original price greater than 0 to calculate discount percentage.")

rating = st.selectbox("Product Rating (stars)", options=[1, 2, 3, 4, 5])
rating_count = st.number_input("Number of Ratings", min_value=0)

# Optional Fields (not used in model but can be logged)
st.header("Optional Additional Information")
_ = st.text_input("Product Category")
_ = st.text_area("Product Description")
_ = st.text_input("Review Title")
_ = st.text_area("Review Content")
_ = st.text_input("Product Link")

# Inference Logic
if st.button("Analyze Transaction"):
    with st.spinner("Running GAT-GCN model..."):
        # Prepare input
        input_features = np.array([[discounted_price, actual_price, discount_percentage, rating, rating_count]])
        x_tensor = torch.tensor(input_features, dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop

        # Load model
        model = HybridGATGCN(in_dim=5, hidden_dim=256, out_dim=2, heads=8, dropout=0.27)
        model.load_state_dict(torch.load("trained_model.pt", map_location=torch.device("cpu")))
        model.eval()

        # Predict
        with torch.no_grad():
            output = model(x_tensor, edge_index)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.exp(output[0])[pred].item()

        # Display result
        if pred == 1:
            st.error(f"🚨 Suspicious Transaction Detected! (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ Transaction Appears Normal (Confidence: {confidence:.2%})")
