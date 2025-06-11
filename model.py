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

class ContrastiveLoss(nn.Module):
    """Hàm mất mát đối nghịch cho học biểu diễn"""
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(sim, labels)

class EarlyStopping:
    """Cơ chế dừng sớm để tránh overfitting"""
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def step(self, score, model=None):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def evaluate(model, data, mask):
    """Đánh giá mô hình với nhiều metric khác nhau"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        correct = pred[mask] == data.y[mask]
        accuracy = correct.sum().item() / mask.sum().item()

        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        try:
            roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        except:
            roc_auc = 0.0

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, roc_auc, rmse, mae, f1, recall

def train(model, data, epochs=50, lr=0.01, alpha=0.1, weight_decay=5e-4, patience=10, verbose=True):
    """Huấn luyện mô hình với học đối nghịch và tăng cường dữ liệu"""
    device = data.x.device
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    criterion = nn.CrossEntropyLoss()
    contrastive_loss_fn = ContrastiveLoss(temperature=0.5)
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    history = {
        'loss': [], 'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'train_recall': [], 'val_recall': []
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # Dự đoán trên dữ liệu gốc
            out = model(data.x, data.edge_index)
            loss_cls = criterion(out[data.train_mask], data.y[data.train_mask])

            # Học đối nghịch với dữ liệu tăng cường
            data_aug = augment_graph(data)
            z1 = model(data.x, data.edge_index)
            z2 = model(data_aug.x, data_aug.edge_index)
            loss_contrastive = contrastive_loss_fn(z1[data.train_mask], z2[data.train_mask])

            # Tổng hợp hàm mất mát
            loss = loss_cls + alpha * loss_contrastive

        # Cập nhật tham số với mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Đánh giá hiệu suất
        train_acc, train_roc_auc, train_rmse, train_mae, train_f1, train_recall = evaluate(model, data, data.train_mask)
        val_acc, val_roc_auc, val_rmse, val_mae, val_f1, val_recall = evaluate(model, data, data.val_mask)

        # Lưu lịch sử huấn luyện
        history['loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)

        # Cập nhật learning rate và kiểm tra early stopping
        scheduler.step(val_f1)
        if early_stopping.step(val_f1, model):
            if verbose:
                print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(early_stopping.best_model_state)
            break

        # In thông tin huấn luyện
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f'Epoch {epoch}/{epochs-1}:')
            print(f'  Loss: {loss.item():.4f}')
            print(f'  Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}')
            print(f'  Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}')

    return model, history

def create_simplified_graph(data, max_nodes=30):
    """Tạo đồ thị đơn giản hóa cho phân tích SHAP"""
    cpu_data = data.cpu()

    # Ưu tiên chọn các nút từ tập test
    if hasattr(cpu_data, 'test_mask') and cpu_data.test_mask.sum() > 0:
        test_indices = torch.where(cpu_data.test_mask)[0]
        if len(test_indices) > max_nodes:
            indices = test_indices[:max_nodes]
        else:
            # Bổ sung các nút từ tập huấn luyện
            remaining = max_nodes - len(test_indices)
            train_indices = torch.where(cpu_data.train_mask)[0][:remaining]
            indices = torch.cat([test_indices, train_indices])
    else:
        indices = torch.arange(min(max_nodes, cpu_data.x.shape[0]))

    x_subset = cpu_data.x[indices].clone()
    y_subset = cpu_data.y[indices].clone()

    # Tạo đồ thị k-NN ổn định hơn
    features_np = x_subset.numpy()
    n_neighbors = max(1, min(3, len(indices)-1))

    # Sử dụng metric khoảng cách mạnh mẽ hơn
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(features_np)
    knn_adj = nbrs.kneighbors_graph(features_np, mode='connectivity')
    edge_index_new = np.array(knn_adj.nonzero())
    edge_index_new = torch.tensor(edge_index_new, dtype=torch.long)

    # Đảm bảo đồ thị liên thông bằng cách thêm self-loops
    self_loops = torch.stack([torch.arange(len(indices)), torch.arange(len(indices))], dim=0)
    edge_index_new = torch.cat([edge_index_new, self_loops], dim=1)

    # Tạo mask
    test_mask = torch.ones(len(indices), dtype=torch.bool)

    simplified_data = Data(
        x=x_subset,
        edge_index=edge_index_new,
        y=y_subset,
        test_mask=test_mask
    )

    return simplified_data

def analyze_with_shap(model, data, feature_names):
    """Phân tích mô hình với SHAP (phiên bản mạnh mẽ)"""
    print("Analyzing model with SHAP...")

    model = model.cpu()
    model.eval()

    # Tạo đồ thị đơn giản hóa
    simple_data = create_simplified_graph(data.cpu(), max_nodes=50)

    total_nodes = simple_data.x.shape[0]
    if total_nodes < 5:
        print("Not enough nodes for SHAP analysis.")
        return False, None

    # Đảm bảo không vượt quá số lượng nút có sẵn
    n_background = min(20, total_nodes)
    n_sample = min(10, total_nodes)

    background_indices = torch.randperm(total_nodes)[:n_background]
    sample_indices = torch.randperm(total_nodes)[:n_sample]

    background_data = simple_data.x[background_indices].numpy()
    sample_features = simple_data.x[sample_indices].numpy()

    def model_predict(x):
        with torch.no_grad():
            if isinstance(x, np.ndarray) and len(x) > 0:
                x_tensor = torch.tensor(x, dtype=torch.float32)
                edge_index = simple_data.edge_index
                out = model(x_tensor, edge_index)
                probs = torch.exp(out).cpu().numpy()
                return [probs[:, i] for i in range(probs.shape[1])]
            else:
                return np.array([])

    try:
        explainer = shap.KernelExplainer(model_predict, background_data, nsamples=100)
        shap_values = explainer.shap_values(sample_features, nsamples=200)

        # Vẽ biểu đồ SHAP
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_features, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=300)
        plt.show()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_features, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar)")
        plt.tight_layout()
        plt.savefig("shap_bar.png", dpi=300)
        plt.show()

        # Tính độ quan trọng của đặc trưng
        if isinstance(shap_values, list):
            feature_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            feature_importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

        print("\nSHAP Feature Importance Table:")
        print(importance_df.head(20))
        importance_df.to_csv("shap_importance.csv", index=False)

        print("SHAP analysis completed successfully.")
        return True, importance_df

    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        return False, None

def calculate_feature_importance(model, data, feature_names):
    """Tính độ quan trọng của đặc trưng bằng phương pháp hoán vị"""
    print("Using permutation feature importance method...")

    # Chuyển mô hình và dữ liệu về CPU
    cpu_model = model.cpu()
    cpu_data = data.cpu()
    cpu_model.eval()

    # Tạo đồ thị đơn giản hóa
    simple_data = create_simplified_graph(cpu_data)

    # Tính độ chính xác cơ sở
    with torch.no_grad():
        out = cpu_model(simple_data.x, simple_data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[simple_data.test_mask] == simple_data.y[simple_data.test_mask]
        baseline_acc = correct.sum().item() / simple_data.test_mask.sum().item()

    # Tính độ quan trọng của đặc trưng
    importances = []

    for i in range(simple_data.x.shape[1]):
        # Lưu giá trị gốc
        original_values = simple_data.x[:, i].clone()

        # Hoán vị đặc trưng
        perm_idx = torch.randperm(simple_data.x.shape[0])
        simple_data.x[:, i] = simple_data.x[perm_idx, i]

        # Đánh giá với đặc trưng đã hoán vị
        with torch.no_grad():
            out = cpu_model(simple_data.x, simple_data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[simple_data.test_mask] == simple_data.y[simple_data.test_mask]
            perm_acc = correct.sum().item() / simple_data.test_mask.sum().item()

        # Khôi phục giá trị gốc
        simple_data.x[:, i] = original_values

        # Tính độ quan trọng (giảm độ chính xác)
        importance = baseline_acc - perm_acc
        importances.append(importance)

    # Tạo DataFrame và vẽ biểu đồ
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    # Tạo biểu đồ
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.title("Feature Importance (Permutation Method)")
    plt.xlabel("Decrease in Accuracy when Feature is Permuted")
    plt.tight_layout()
    plt.savefig("feature_importance_permutation.png", dpi=300, bbox_inches='tight')
    plt.show()

    # In bảng
    print("\nFeature Importance Table (Permutation Method):")
    print(importance_df.head(20))
    importance_df.to_csv("feature_importance_permutation.csv", index=False)

    return importance_df

def run_pipeline(data_path, model_path=None, train_new=False):
    """Pipeline hoàn chỉnh từ tải dữ liệu đến phân tích SHAP"""
    print("Running GNN pipeline with SHAP...")

    # 1. Tải dữ liệu và tạo đồ thị
    print("Loading data...")
    features, labels, feature_names = load_data(data_path)
    data, feature_names = create_graph(features, labels)

    # 2. Khởi tạo hoặc tải mô hình
    print("Setting up model...")
    device = torch.device('cpu')  # Sử dụng CPU để tránh lỗi CUDA với SHAP

    model = HybridGATGCN(
        in_dim=features.shape[1],
        hidden_dim=256,
        out_dim=len(np.unique(labels)),
        heads=8,
        dropout=0.27
    )
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

    # 3. Đánh giá mô hình
    print("Evaluating model...")
    test_acc, test_roc_auc, test_rmse, test_mae, test_f1, test_recall = evaluate(model, data, data.test_mask)
    print("\nTest results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"Recall: {test_recall:.4f}")

    # 4. Phân tích SHAP với cơ chế thử lại
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\nAttempting SHAP analysis (attempt {attempt+1}/{max_retries})...")
        success, importance_df = analyze_with_shap(model, data, feature_names)
        if success:
            break
        elif attempt < max_retries - 1:
            print("Retrying with different parameters...")

    # 5. Sử dụng phương pháp dự phòng nếu SHAP thất bại
    if not success:
        print("SHAP analysis failed after multiple attempts.")
        print("Using permutation feature importance method as fallback...")
        importance_df = calculate_feature_importance(model, data, feature_names)

    print("Pipeline completed!")
    return model, importance_df

if __name__ == "__main__":
    run_pipeline("preprocessed_data.csv", model_path="model.pt")
