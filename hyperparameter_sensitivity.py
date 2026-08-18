import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

from dataset import process_csv_to_tensors
from experiment_paths import figure_path
from model import VanillaSTGAT
from metrics import PhysicsLossCheck

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def calc_jam_mae(pred, true):
    return torch.abs(pred[..., 0]*10.0 - true[..., 0]*10.0).mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 啟動超參數敏感度分析 (設備: {device})...")
    
    csv_file = "all_segments_preprocessed.csv"
    full_dataloader, adj_matrix, _ = process_csv_to_tensors(csv_file, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=32, shuffle=False)
    
    # 測試不同的物理權重
    lambda_list = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = []
    
    for l_phy in lambda_list:
        print(f"⏳ 正在測試 Lambda_phy = {l_phy} ...")
        model = VanillaSTGAT(18, 64, 128, 2, adj_tensor).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        physics_loss = PhysicsLossCheck(adj_tensor)
        
        # 為了看出差異，我們跑 5 個 epoch
        model.train()
        for epoch in range(5):
            for batch in train_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                optimizer.zero_grad()
                pred, _ = model(x)
                
                if l_phy > 0.0:
                    loss = criterion(pred, y) + l_phy * physics_loss(pred, x[:, -1, :, 0])
                else:
                    loss = criterion(pred, y)
                    
                loss.backward()
                optimizer.step()
                
        model.eval()
        total_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                pred, _ = model(x)
                total_mae += calc_jam_mae(pred, y)
        
        final_mae = total_mae / len(val_loader)
        results.append({'Lambda': str(l_phy), 'Jam MAE': final_mae})
        print(f"   ✅ 結果: {final_mae:.4f}")

    # ==========================================
    # 繪製 U 型敏感度曲線
    # ==========================================
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x='Lambda', y='Jam MAE', marker='o', 
                      linewidth=3, markersize=12, color='#2980b9')
    
    # 標註最佳點
    min_idx = df['Jam MAE'].idxmin()
    best_lambda = df.loc[min_idx, 'Lambda']
    best_mae = df.loc[min_idx, 'Jam MAE']
    
    plt.annotate(f'Optimal $\\lambda_{{phy}}$ = {best_lambda}\n(MAE = {best_mae:.4f})',
                 xy=(min_idx, best_mae), xytext=(0, 30),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color='#c0392b',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

    plt.title(r'Hyperparameter Sensitivity Analysis ($\lambda_{phy}$)', fontsize=18, fontweight='bold', pad=15)
    plt.ylabel('Jam Factor MAE', fontsize=14, fontweight='bold')
    plt.xlabel(r'Physics Loss Weight ($\lambda_{phy}$)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(figure_path("Fig_Hyperparameter_Sensitivity.png"), dpi=300, bbox_inches='tight')
    print("\n✅ 實驗完成！完美的 U 型圖已儲存為 'Fig_Hyperparameter_Sensitivity.png'")

if __name__ == "__main__":
    main()
