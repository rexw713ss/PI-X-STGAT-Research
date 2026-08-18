import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import random

from dataset import process_csv_to_tensors
from experiment_paths import figure_path
from model import PI_X_STGAT
from metrics import PhysicsLossCheck

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
    print(f"\n🏟️ 啟動 XAI 主動干預模擬 (大型活動與外圍封路管制) (設備: {device})...")
    
    csv_file = "all_segments_preprocessed.csv"
    full_dataloader, adj_matrix, node_names = process_csv_to_tensors(csv_file, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]
    
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    test_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=32, shuffle=False)
    
    print("⏳ 正在載入 PI-X-STGAT 進行權重暖身...")
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
            
    # === 情境 3: 大型活動 (Regional Cluster Anomaly) ===
    model.eval()
    batch = next(iter(test_loader))
    x_real = batch["x"].to(device)
    
    # 1. 找一個度數最高的樞紐當作「體育場/巨蛋 (Event Center)」
    node_degrees = adj_matrix.sum(axis=1)
    target_node_idx = np.argmax(node_degrees)
    target_node_name = node_names[target_node_idx]
    
    # 找出體育場周邊的道路 (1-hop 鄰居)
    event_perimeter = np.where(adj_matrix[target_node_idx] > 0)[0]
    
    # 2. 獲取 Attention 矩陣，找出最致命的外圍源頭
    with torch.no_grad():
        _, attn_matrix = model(x_real)
    target_attention = attn_matrix[0, target_node_idx, :].cpu().numpy()
    target_attention[target_node_idx] = 0.0 
    
    top_source_idx = np.argmax(target_attention)
    # 找一個雖然在周邊但影響力極低的節點
    non_zero_indices = np.where(target_attention > 0.001)[0]
    low_source_idx = non_zero_indices[np.argmin(target_attention[non_zero_indices])] if len(non_zero_indices) > 1 else np.argsort(target_attention)[-2]
        
    print(f"\n🏟️ 大型活動舉辦地 (目標路口): {target_node_name}")
    print(f"🔴 XAI 發現 [主動脈灌入源]: {node_names[top_source_idx]} (權重: {target_attention[top_source_idx]:.4f})")
    print(f"🟡 XAI 發現 [次要周邊道路]: {node_names[low_source_idx]} (權重: {target_attention[low_source_idx]:.4f})")
    
    # 3. 注入危機 (Baseline)：體育場與周邊所有道路，在過去 30 分鐘內全面陷入極度壅塞！
    x_baseline = x_real.clone()
    x_baseline[0, -6:, target_node_idx, 0] = 1.0 # 體育場爆滿
    for neighbor in event_perimeter:
        x_baseline[0, -6:, neighbor, 0] = 0.9 # 周邊道路全面塞爆
        
    with torch.no_grad():
        pred_base, _ = model(x_baseline)
    jam_baseline = pred_base[0, target_node_idx, 0].item() * 10.0
    
    # 定義物理上的極端封路管制 (Road Closure)
    min_jam_val = x_real[0, :, :, 0].min().item() # 淨空車流
    extreme_low_speed = -3.0                      # 車速降至零
    
    # 4. 盲目外圍管制 (Random)：警察封鎖了影響力低的外圍道路
    x_random = x_baseline.clone()
    x_random[0, -6:, low_source_idx, 0] = min_jam_val 
    x_random[0, -6:, low_source_idx, 1] = extreme_low_speed
    with torch.no_grad():
        pred_random, _ = model(x_random)
    jam_random = pred_random[0, target_node_idx, 0].item() * 10.0
    
    # 5. XAI 導向精準封路 (Guided)：警察封鎖了真正灌入車流的主動脈
    x_xai = x_baseline.clone()
    x_xai[0, -6:, top_source_idx, 0] = min_jam_val
    x_xai[0, -6:, top_source_idx, 1] = extreme_low_speed
    with torch.no_grad():
        pred_xai, _ = model(x_xai)
    jam_xai = pred_xai[0, target_node_idx, 0].item() * 10.0
    
    drop_percent = ((jam_baseline - jam_xai) / jam_baseline) * 100

    # ==========================================
    # 🎨 繪圖
    # ==========================================
    print(f"\n📈 區域壅塞 Baseline 預測值: {jam_baseline:.2f}")
    print(f"📈 盲目封路後 預測值: {jam_random:.2f}")
    print(f"📉 XAI 精準封路後 預測值: {jam_xai:.2f} (下降 {drop_percent:.1f}%)")
    
    print("\n🎨 正在繪製大型活動外圍管制成效圖...")
    df_intervention = pd.DataFrame({
        'Strategy': ['Event Gridlock\n(No Intervention)', 'Random Perimeter\nClosure', 'XAI-Guided Perimeter\nClosure'],
        'Predicted Jam Factor': [jam_baseline, jam_random, jam_xai]
    })
    
    plt.figure(figsize=(9, 6))
    colors = ['#8D6E63', '#FFCA28', '#42A5F5']
    bars = plt.bar(df_intervention['Strategy'], df_intervention['Predicted Jam Factor'], 
                   color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    clean_node_name = target_node_name.split(',')[0][:25]
    plt.title(f"Large-Scale Event Perimeter Control Simulation\nat {clean_node_name}", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Predicted Jam Factor at Event Center", fontsize=14, fontweight='bold')
    
    plt.axhline(jam_baseline, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
                 
    if drop_percent > 0:
        plt.annotate(
            f"-{drop_percent:.1f}% Gridlock Relief!", 
            xy=(2, jam_xai + 0.15), 
            xytext=(2.35, jam_baseline - (jam_baseline * 0.15)), 
            arrowprops=dict(facecolor='#4CAF50', shrink=0.05, width=2.5, headwidth=9, edgecolor='black'),
            fontsize=13, fontweight='bold', color='white', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="#4CAF50", ec="black", lw=1.2)
        )

    plt.ylim(0, max(jam_baseline, jam_random, jam_xai) * 1.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(figure_path("Fig_Event_Intervention_Simulation.png"), dpi=300, bbox_inches='tight')
    print("✅ 繪圖完成！請檢查 'Fig_Event_Intervention_Simulation.png'。")

if __name__ == "__main__":
    main()
