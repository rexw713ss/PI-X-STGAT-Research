import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

from dataset import process_csv_to_tensors
# 🚨 引入最新的強力模型
from model import AGCRN_Baseline, GraphWaveNet_Baseline, LSTMBaseline, MTGNN_Baseline, PI_X_STGAT
from metrics import PhysicsLossCheck

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def calc_jam_mae(pred, true):
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    return torch.abs(pred_jam - true_jam).mean().item()

def train_and_eval(model, train_loader, val_loader, device, model_type="lstm", adj_tensor=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 🚨 換成 SmoothL1Loss (Huber Loss)，避免極端值干擾
    criterion = nn.SmoothL1Loss(beta=1.0) 
    
    if model_type == "pix":
        physics_loss = PhysicsLossCheck(adj_tensor)
        
    model.train()
    epochs = 8 # 稍微加長一點讓自適應圖矩陣(AGCRN/PI-X)能收斂
    
    for epoch in range(1, epochs + 1):
        # PI-X-STGAT 的小暖身 (Warm-up) 機制
        if model_type == "pix":
            current_lambda = 0.0 if epoch <= 3 else 0.015
        else:
            current_lambda = 0.0
            
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            
            # 根據不同模型的回傳值解包
            if model_type in ["pix", "agcrn", "gwnet", "mtgnn"]:
                pred, _ = model(x)
            else:
                pred = model(x)
                
            loss = criterion(pred, y)
            
            # 加入物理約束
            if model_type == "pix" and current_lambda > 0:
                loss += current_lambda * physics_loss(pred, x[:, -1, :, 0])
                
            loss.backward()
            optimizer.step()
            
    # === 驗證階段 ===
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            
            if model_type in ["pix", "agcrn", "gwnet", "mtgnn"]:
                pred, _ = model(x)
            else:
                pred = model(x)
                
            total_mae += calc_jam_mae(pred, y)
            
    return total_mae / len(val_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file = "all_segments_preprocessed.csv"
    
    horizons = [1, 2, 3, 4]
    horizon_labels = ['15 min', '30 min', '45 min', '60 min']
    
    results = []
    
    print(f"🚀 啟動多步預測分析 (設備: {device})")
    
    for h_idx, h in enumerate(horizons):
        print(f"\n⏳ 正在測試預測未來 {horizon_labels[h_idx]} (Horizon = {h})...")
        # 獲取 num_nodes
        full_dataloader, adj_matrix, nodes = process_csv_to_tensors(csv_file, seq_len=12, horizon=h)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        num_nodes = adj_matrix.shape[0]
        
        full_dataset = full_dataloader.dataset
        train_size = int(len(full_dataset) * 0.8)
        train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=32, shuffle=False)
        
        # 1. 歷史平均 (HA)
        ha_mae = 0.0
        for batch in val_loader:
            ha_pred = batch["x"][..., :2].mean(dim=1)
            ha_mae += calc_jam_mae(ha_pred, batch["y"])
        ha_mae /= len(val_loader)
        results.append({'Horizon': horizon_labels[h_idx], 'Model': 'HA', 'Jam MAE': ha_mae})
        
        # 2. LSTM
        lstm = LSTMBaseline(18, 64, 2).to(device)
        lstm_mae = train_and_eval(lstm, train_loader, val_loader, device, model_type="lstm")
        results.append({'Horizon': horizon_labels[h_idx], 'Model': 'LSTM', 'Jam MAE': lstm_mae})
        
        # 3. AGCRN (大魔王 Baseline)
        agcrn = AGCRN_Baseline(num_nodes=num_nodes, input_dim=18, hidden_dim=128, output_dim=2, embed_dim=10).to(device)
        agcrn_mae = train_and_eval(agcrn, train_loader, val_loader, device, model_type="agcrn")
        results.append({'Horizon': horizon_labels[h_idx], 'Model': 'AGCRN', 'Jam MAE': agcrn_mae})
        
        # 4. PI-X-STGAT (我們的模型)
        gwnet = GraphWaveNet_Baseline(
            num_nodes=num_nodes,
            input_dim=18,
            hidden_dim=64,
            output_dim=2,
            adj_matrix=adj_tensor,
            embed_dim=10,
        ).to(device)
        gwnet_mae = train_and_eval(gwnet, train_loader, val_loader, device, model_type="gwnet")
        results.append({'Horizon': horizon_labels[h_idx], 'Model': 'Graph WaveNet', 'Jam MAE': gwnet_mae})

        mtgnn = MTGNN_Baseline(
            num_nodes=num_nodes,
            input_dim=18,
            hidden_dim=64,
            output_dim=2,
            embed_dim=10,
            top_k=6,
        ).to(device)
        mtgnn_mae = train_and_eval(mtgnn, train_loader, val_loader, device, model_type="mtgnn")
        results.append({'Horizon': horizon_labels[h_idx], 'Model': 'MTGNN', 'Jam MAE': mtgnn_mae})

        pix_stgat = PI_X_STGAT(num_nodes=num_nodes, input_dim=18, gat_dim=64, gru_dim=128, output_dim=2, adj_matrix=adj_tensor, embed_dim=10).to(device)
        pix_mae = train_and_eval(pix_stgat, train_loader, val_loader, device, model_type="pix", adj_tensor=adj_tensor)
        print(f"   [New Baselines] Graph WaveNet: {gwnet_mae:.4f} | MTGNN: {mtgnn_mae:.4f}")
        results.append({'Horizon': horizon_labels[h_idx], 'Model': 'PI-X-STGAT (Ours)', 'Jam MAE': pix_mae})
        
        print(f"   [結果] HA: {ha_mae:.4f} | LSTM: {lstm_mae:.4f} | AGCRN: {agcrn_mae:.4f} | PI-X-STGAT: {pix_mae:.4f}")

    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    # 加入了 AGCRN 的專屬橘色
    sns.lineplot(data=df, x='Horizon', y='Jam MAE', hue='Model', marker='o', linewidth=3, markersize=10, 
                 palette=['#95a5a6', '#e74c3c', '#f39c12', '#8e44ad', '#0097a7', '#1565C0'])
    
    plt.title('Multi-step Forecasting Performance Degradation', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Jam Factor MAE', fontsize=14, fontweight='bold')
    plt.xlabel('Forecasting Horizon', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("Fig_Multistep_Forecasting_Updated.png", dpi=300, bbox_inches='tight')
    print("\n✅ 實驗完成！折線圖已儲存為 'Fig_Multistep_Forecasting_Updated.png'")

if __name__ == "__main__":
    main()
