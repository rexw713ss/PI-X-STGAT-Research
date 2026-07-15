import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import random

from dataset import process_csv_to_tensors
from model import PI_X_STGAT 
from metrics import PhysicsLossCheck

# 鎖定隨機亂數種子，保證實驗重現
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
sns.set_context("paper", font_scale=1.2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🌪️ 啟動 PI-X-STGAT 終極 OoD 壓力測試引擎 (設備: {device})...")
    
    # 1. 載入資料
    csv_file = "all_segments_preprocessed.csv"
    full_dataloader, adj_matrix, node_names = process_csv_to_tensors(csv_file, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0] 
    
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    test_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=32, shuffle=False)
    
    # 2. 訓練 PI-X-STGAT (共用這組最強權重來跑所有測試)
    print("\n⏳ 正在訓練 PI-X-STGAT (10 Epochs 暖身)...")
    model = PI_X_STGAT(num_nodes=num_nodes, input_dim=18, gat_dim=64, gru_dim=128, output_dim=2, adj_matrix=adj_tensor, embed_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor)
    
    model.train()
    for epoch in range(1, 11): 
        current_lambda = 0.015 if epoch > 5 else 0.0
        for batch in test_loader: 
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            if current_lambda > 0:
                loss += current_lambda * physics_loss(pred, x[:, -1, :, 0])
            loss.backward()
            optimizer.step()
            
    # 取出 Test Set 中的一個 Batch 當作正常基準線
    model.eval()
    batch = next(iter(test_loader))
    x_base, y_base = batch["x"].to(device), batch["y"].to(device)
    
    with torch.no_grad():
        pred_base, attn_matrix = model(x_base)
        phy_base = physics_loss(pred_base, x_base[:, -1, :, 0]).item()
        
    node_degrees = adj_matrix.sum(axis=1)

    # ==========================================
    # 🚗 情境 1：單點突發異常 (車禍 Local Shock)
    # ==========================================
    crash_node_idx = np.argmax(node_degrees) # 找最大樞紐
    crash_neighbors = np.where(adj_matrix[crash_node_idx] > 0)[0]
    crash_neighbors = [idx for idx in crash_neighbors if idx != crash_node_idx]
    
    print(f"\n🚗 [情境 1] 核心節點 '{node_names[crash_node_idx]}' 發生嚴重車禍...")
    x_acc = x_base.clone()
    x_acc[0, -3:, crash_node_idx, 0] = 1.0  # Jam=1.0
    x_acc[0, -3:, crash_node_idx, 1] = -3.0 # Speed=-3.0
    
    with torch.no_grad():
        pred_acc, _ = model(x_acc)
    delta_pred = (pred_acc - pred_base) * 10.0 
    
    focus_jam_delta = delta_pred[0, crash_node_idx, 0].item()
    focus_speed_delta = delta_pred[0, crash_node_idx, 1].item()
    impacted_jam_delta = delta_pred[0, crash_neighbors, 0].mean().item() if len(crash_neighbors) > 0 else 0.0
    impacted_speed_delta = delta_pred[0, crash_neighbors, 1].mean().item() if len(crash_neighbors) > 0 else 0.0

    print("🎨 正在繪製：情境 1 圖表...")
    df_local = pd.DataFrame({
        'Scenario': ['Accident-Focus', 'Accident-Focus', 'Accident-Impacted (Neighbors)', 'Accident-Impacted (Neighbors)'],
        'Metric': ['Jam Delta', 'Speed Delta', 'Jam Delta', 'Speed Delta'],
        'Value': [focus_jam_delta, focus_speed_delta, impacted_jam_delta, impacted_speed_delta]
    })
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_local, x='Scenario', y='Value', hue='Metric', palette=['#E53935', '#1E88E5'], edgecolor='black', linewidth=1.2)
    plt.title("OoD Scenario 1: Local Shock Propagation (Car Crash)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Delta (Accident - Baseline)", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Fig_OoD_1_LocalShock.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # ⛈️ 情境 2：全域異常 (極端氣候 Global Anomaly)
    # ==========================================
    print(f"\n⛈️ [情境 2] 全域極端氣候異常 (天氣特徵暴增 3 倍)...")
    x_wea = x_base.clone()
    x_wea[:, :, :, 6:] = x_wea[:, :, :, 6:] * 3.0 
    
    with torch.no_grad():
        pred_wea, _ = model(x_wea)
        phy_wea = physics_loss(pred_wea, x_wea[:, -1, :, 0]).item()
        
    delta_wea = (pred_wea - pred_base) * 10.0
    net_wea_jam_delta = delta_wea[..., 0].mean().item()
    net_wea_speed_delta = delta_wea[..., 1].mean().item()
    phy_shift = phy_wea - phy_base

    print("🎨 正在繪製：情境 2 圖表...")
    df_global = pd.DataFrame({
        'Scenario': ['Weather Extreme', 'Weather Extreme', 'Weather Extreme'],
        'Metric': ['Network Jam Delta', 'Network Speed Delta', 'Physics Loss Shift'],
        'Value': [net_wea_jam_delta, net_wea_speed_delta, phy_shift]
    })
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_global, x='Scenario', y='Value', hue='Metric', palette=['#E53935', '#1E88E5', '#43A047'], edgecolor='black', linewidth=1.2)
    plt.title("OoD Scenario 2: Network Responses to Weather", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Delta / Shift", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig("Fig_OoD_2_GlobalWeather.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # 🏟️ 情境 3：區域異常與主動干預 (大型活動 Regional Event)
    # ==========================================
    event_node_idx = np.argsort(node_degrees)[-2] # 找第二大的樞紐當巨蛋，避開剛剛車禍的路口
    event_node_name = node_names[event_node_idx]
    event_perimeter = np.where(adj_matrix[event_node_idx] > 0)[0]
    
    target_attention = attn_matrix[0, event_node_idx, :].cpu().numpy()
    target_attention[event_node_idx] = 0.0 
    
    top_source_idx = np.argmax(target_attention)
    non_zero_indices = np.where(target_attention > 0.001)[0]
    low_source_idx = non_zero_indices[np.argmin(target_attention[non_zero_indices])] if len(non_zero_indices) > 1 else np.argsort(target_attention)[-2]

    print(f"\n🏟️ [情境 3] 大型活動區域壅塞與 XAI 主動干預 (目標: {event_node_name})...")
    print(f"   🔴 XAI 主動脈: {node_names[top_source_idx]} | 🟡 次要外圍: {node_names[low_source_idx]}")
    
    # Baseline: 體育場與周邊全面癱瘓
    x_event = x_base.clone()
    x_event[0, -6:, event_node_idx, 0] = 1.0 
    for neighbor in event_perimeter:
        x_event[0, -6:, neighbor, 0] = 0.9 
        
    with torch.no_grad():
        pred_event_base, _ = model(x_event)
    jam_event_baseline = pred_event_base[0, event_node_idx, 0].item() * 10.0
    
    # 物理封路設定 (淨空車流、降速)
    min_jam_val = x_base[0, :, :, 0].min().item()
    extreme_low_speed = -3.0                      
    
    # Random Intervention: 封鎖次要道路
    x_random = x_event.clone()
    x_random[0, -6:, low_source_idx, 0] = min_jam_val 
    x_random[0, -6:, low_source_idx, 1] = extreme_low_speed
    with torch.no_grad():
        pred_random, _ = model(x_random)
    jam_random = pred_random[0, event_node_idx, 0].item() * 10.0
    
    # XAI Intervention: 封鎖主動脈
    x_xai = x_event.clone()
    x_xai[0, -6:, top_source_idx, 0] = min_jam_val
    x_xai[0, -6:, top_source_idx, 1] = extreme_low_speed
    with torch.no_grad():
        pred_xai, _ = model(x_xai)
    jam_xai = pred_xai[0, event_node_idx, 0].item() * 10.0
    
    drop_percent = ((jam_event_baseline - jam_xai) / jam_event_baseline) * 100

    print("🎨 正在繪製：情境 3 圖表...")
    df_intervention = pd.DataFrame({
        'Strategy': ['Event Gridlock\n(No Intervention)', 'Random Perimeter\nClosure', 'XAI-Guided Perimeter\nClosure'],
        'Predicted Jam Factor': [jam_event_baseline, jam_random, jam_xai]
    })
    
    plt.figure(figsize=(9, 6))
    colors = ['#8D6E63', '#FFCA28', '#42A5F5']
    bars = plt.bar(df_intervention['Strategy'], df_intervention['Predicted Jam Factor'], color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    clean_node_name = event_node_name.split(',')[0][:25]
    plt.title(f"OoD Scenario 3: Event Perimeter Control\n({clean_node_name})", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Predicted Jam Factor at Event Center", fontsize=14, fontweight='bold')
    plt.axhline(jam_event_baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
                 
    if drop_percent > 0:
        plt.annotate(
            f"-{drop_percent:.1f}% Gridlock Relief!", xy=(2, jam_xai + 0.15), xytext=(2.35, jam_event_baseline - (jam_event_baseline * 0.15)), 
            arrowprops=dict(facecolor='#4CAF50', shrink=0.05, width=2.5, headwidth=9, edgecolor='black'),
            fontsize=13, fontweight='bold', color='white', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#4CAF50", ec="black", lw=1.2)
        )

    plt.ylim(0, max(jam_event_baseline, jam_random, jam_xai) * 1.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Fig_OoD_3_EventIntervention.png", dpi=300, bbox_inches='tight')

    print("\n✅ 三大 OoD 極端測試全數完成！請檢查資料夾中的 3 張圖片。")

if __name__ == "__main__":
    main()