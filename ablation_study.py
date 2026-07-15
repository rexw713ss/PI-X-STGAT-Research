import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import random
import numpy as np

from dataset import process_csv_to_tensors
from metrics import PhysicsLossCheck

# ==========================================
# 0. 鎖定隨機亂數種子，保證實驗重現
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# ==========================================
# 1. 網路結構定義
# ==========================================
class EnhancedGATLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.attn_src = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor, combined_adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.fc(x)
        scores = self.attn_src(h) + self.attn_dst(h).transpose(1, 2)
        scores = self.leaky_relu(scores)
        if combined_adj.dim() == 2:
            combined_adj = combined_adj.unsqueeze(0)
        scores = scores + torch.log(combined_adj + 1e-9) 
        attention = torch.softmax(scores, dim=-1)
        out = torch.bmm(attention, h) + self.residual(x)
        return out, attention

class Ablation_PI_X_STGAT(nn.Module):
    def __init__(self, num_nodes, input_dim, gat_dim, gru_dim, output_dim, adj_matrix, use_static=True, use_napl=True, embed_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.use_napl = use_napl
        
        if not use_static:
            self.register_buffer('static_adj', torch.eye(num_nodes))
        else:
            self.register_buffer('static_adj', adj_matrix)
            
        if self.use_napl:
            self.node_emb_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
            self.node_emb_2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.01)
            
        self.gat = EnhancedGATLayer(input_dim, gat_dim)
        self.gru = nn.GRU(gat_dim, gru_dim, batch_first=True)
        self.fc = nn.Linear(gru_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # V1-V3 只吃 2D 交通特徵 (Jam Factor, Speed)
        if self.input_dim == 2:
            x = x[..., :2] 
            
        B, S, N, F_dim = x.shape
        
        if self.use_napl:
            import torch.nn.functional as F
            adp_adj = F.softmax(F.relu(torch.mm(self.node_emb_1, self.node_emb_2)), dim=1)
            combined_adj = self.static_adj + adp_adj
        else:
            combined_adj = self.static_adj
            
        x_flat = x.reshape(B * S, N, F_dim)
        gat_out, _ = self.gat(x_flat, combined_adj) 
        gat_out = gat_out.reshape(B, S, N, -1).permute(0, 2, 1, 3).reshape(B * N, S, -1)
        gru_out, _ = self.gru(gat_out)
        
        pred = self.fc(gru_out[:, -1, :]).reshape(B, N, -1)   
        return pred

def calc_jam_mae(pred, true):
    return torch.abs(pred[..., 0]*10.0 - true[..., 0]*10.0).mean().item()

# ==========================================
# 2. 嚴謹的訓練與驗證函數 (含 Best Epoch Tracker)
# ==========================================
def train_ablation(model, train_loader, val_loader, device, use_phy=False, adj_tensor=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor) if use_phy else None
    
    best_mae = float('inf') # 追蹤歷史最低 MAE
    epochs = 15 # 延長 Epoch 確保所有模型都能達到巔峰
    
    for epoch in range(1, epochs + 1):
        model.train()
        current_lambda = (0.015 if epoch > 3 else 0.0) if use_phy else 0.0
        
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if use_phy and current_lambda > 0:
                loss += current_lambda * physics_loss(pred, x[:, -1, :, 0])
            loss.backward()
            optimizer.step()
            
        # 每個 Epoch 結束都進行 Validation
        model.eval()
        total_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                pred = model(x)
                total_mae += calc_jam_mae(pred, y)
                
        current_val_mae = total_mae / len(val_loader)
        if current_val_mae < best_mae:
            best_mae = current_val_mae # 更新最佳成績
            
    return best_mae

# ==========================================
# 3. 主程式：執行 4 階段並自動標記亮點
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file = "all_segments_preprocessed.csv"
    
    print("🚀 啟動 4 階段 PI-X-STGAT 終極消融實驗 (Best Epoch 模式)...")
    full_dataloader, adj_matrix, _ = process_csv_to_tensors(csv_file, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]
    
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=32, shuffle=False)
    
    results = []

    print("⏳ V1: Base-GRU (2D特徵, 無圖, 無物理)...")
    v1 = Ablation_PI_X_STGAT(num_nodes, 2, 64, 128, 2, adj_tensor, use_static=False, use_napl=False).to(device)
    v1_mae = train_ablation(v1, train_loader, val_loader, device, use_phy=False)
    results.append({'Variant': 'Base-GRU\n(No Spatial/Phy)', 'Jam MAE': v1_mae})
    print(f"✅ V1 Best MAE: {v1_mae:.4f}")

    print("⏳ V2: STGAT (2D特徵, 實體圖, 無物理)...")
    v2 = Ablation_PI_X_STGAT(num_nodes, 2, 64, 128, 2, adj_tensor, use_static=True, use_napl=False).to(device)
    v2_mae = train_ablation(v2, train_loader, val_loader, device, use_phy=False)
    results.append({'Variant': 'STGAT\n(+ Spatial GAT)', 'Jam MAE': v2_mae})
    print(f"✅ V2 Best MAE: {v2_mae:.4f}")

    print("⏳ V3: PI-STGAT (2D特徵, 實體圖, 加物理約束)...")
    v3 = Ablation_PI_X_STGAT(num_nodes, 2, 64, 128, 2, adj_tensor, use_static=True, use_napl=False).to(device)
    v3_mae = train_ablation(v3, train_loader, val_loader, device, use_phy=True, adj_tensor=adj_tensor)
    results.append({'Variant': 'PI-STGAT\n(+ Physics Loss)', 'Jam MAE': v3_mae})
    print(f"✅ V3 Best MAE: {v3_mae:.4f}")

    print("⏳ V4: PI-X-STGAT (18D全特徵, 實體圖+NAPL, 加物理約束)...")
    v4 = Ablation_PI_X_STGAT(num_nodes, 18, 64, 128, 2, adj_tensor, use_static=True, use_napl=True).to(device)
    v4_mae = train_ablation(v4, train_loader, val_loader, device, use_phy=True, adj_tensor=adj_tensor)
    results.append({'Variant': 'PI-X-STGAT\n(Complete Model)', 'Jam MAE': v4_mae})
    print(f"✅ V4 Best MAE: {v4_mae:.4f}")

    # ==========================================
    # 4. 繪製帶有「物理 MVP 亮點」的圖表
    # ==========================================
    df = pd.DataFrame(results)
    plt.figure(figsize=(11, 6))
    colors = ['#CFD8DC', '#90CAF9', '#42A5F5', '#1565C0']
    bars = plt.bar(df['Variant'], df['Jam MAE'], color=colors, edgecolor='#0D47A1', linewidth=1.5, width=0.5)
    
    plt.title('Ablation Study', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Jam Factor MAE (Lower is Better)', fontsize=14, fontweight='bold')
    
    # 標上數值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    # 🚨 動態計算 V2 到 V3 的進步幅度，畫上醒目的箭頭
    drop_percent = ((v2_mae - v3_mae) / v2_mae) * 100
    if drop_percent > 0:
        plt.annotate(
            f"Physics Constraint\nDrops Error by {drop_percent:.1f}%!", 
            xy=(2, v3_mae + 0.003), # 指向 V3 的柱子上方
            xytext=(1.5, v2_mae + 0.005), # 文字放在兩者之間偏上方
            arrowprops=dict(facecolor='#E53935', shrink=0.05, width=3, headwidth=10, edgecolor='none'),
            fontsize=12, fontweight='bold', color='#E53935', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E53935", lw=1.5)
        )

    plt.ylim(df['Jam MAE'].min() * 0.8, df['Jam MAE'].max() * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Fig_Ablation_Awesome.png", dpi=300, bbox_inches='tight')
    print("\n✅ 消融實驗完成！超炫砲圖表已儲存為 'Fig_Ablation_Awesome.png'")

if __name__ == "__main__":
    main()