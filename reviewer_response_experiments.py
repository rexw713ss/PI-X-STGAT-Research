import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

# 記得確保這兩個 import 能對應到你專案中的檔案
from dataset import process_csv_to_tensors
from model import VanillaSTGAT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動 Reviewer 防禦實驗 (設備: {device})...")

    # ==========================================
    # 1. 載入測試集
    # ==========================================
    csv_file = "all_segments_preprocessed.csv"
    full_dataloader, adj_matrix, node_names = process_csv_to_tensors(csv_file, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    test_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ==========================================
    # 2. 載入模型 (這裡需要你的兩種模型權重)
    # ==========================================
    # 模型 A: 完全沒有物理約束 (Lambda = 0)
    model_base = VanillaSTGAT(18, 64, 128, 2, adj_tensor).to(device)
    # model_base.load_state_dict(torch.load("model_lambda_0.pth")) # 請替換為真實權重
    model_base.eval()

    # 模型 B: 我們的最佳模型 (Lambda = 0.01)
    model_ours = VanillaSTGAT(18, 64, 128, 2, adj_tensor).to(device)
    # model_ours.load_state_dict(torch.load("best_model_lambda_0.01.pth")) # 請替換為真實權重
    model_ours.eval()

    # ==========================================
    # 3. 收集實驗數據：誤差分佈 & 物理殘差
    # ==========================================
    errors_ours = []      # 紀錄絕對誤差 (|True - Pred|)
    residuals_base = []   # 紀錄 Base 模型的物理殘差
    residuals_ours = []   # 紀錄 Ours 模型的物理殘差

    print("📊 正在測試集上運算物理殘差與誤差分佈...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y_true = batch["y"].to(device)
            
            # --- 模型 A (無物理約束) 推論 ---
            y_pred_base, _ = model_base(x)
            
            # --- 模型 B (我們的模型) 推論 ---
            y_pred_ours, _ = model_ours(x)
            
            # --- 計算 Absolute Error (針對 Jam Factor) ---
            abs_err = torch.abs(y_pred_ours[:, :, 0] - y_true[:, :, 0])
            errors_ours.extend(abs_err.cpu().numpy().flatten())
            
            # --- 計算物理守恆殘差 L_phy (簡化版計算式，需對齊你論文的公式 12) ---
            # q_i = sigmoid(v) * relu(rho)
            # Base 殘差
            v_base = torch.sigmoid(y_pred_base[:, :, 1])
            rho_base = torch.relu(y_pred_base[:, :, 0] / 10.0)
            q_base = v_base * rho_base
            q_in_base = torch.matmul(q_base, adj_tensor.T)
            q_out_base = torch.matmul(q_base, adj_tensor)
            delta_j_base = y_pred_base[:, :, 0] - x[:, -1, :, 0] # 預測值 - 當下最後一筆
            res_base = torch.abs((q_in_base - q_out_base) - delta_j_base)
            residuals_base.extend(res_base.cpu().numpy().flatten())

            # Ours 殘差
            v_ours = torch.sigmoid(y_pred_ours[:, :, 1])
            rho_ours = torch.relu(y_pred_ours[:, :, 0] / 10.0)
            q_ours = v_ours * rho_ours
            q_in_ours = torch.matmul(q_ours, adj_tensor.T)
            q_out_ours = torch.matmul(q_ours, adj_tensor)
            delta_j_ours = y_pred_ours[:, :, 0] - x[:, -1, :, 0]
            res_ours = torch.abs((q_in_ours - q_out_ours) - delta_j_ours)
            residuals_ours.extend(res_ours.cpu().numpy().flatten())

    # ==========================================
    # 4. 繪製防禦圖表 A：物理殘差分佈 (證明 L_phy 有效)
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(residuals_base, ax=axes[0], color='#FF7043', fill=True, label='Baseline ($\lambda_{phy}=0$)')
    sns.kdeplot(residuals_ours, ax=axes[0], color='#1976D2', fill=True, label='PI-X-STGAT ($\lambda_{phy}=0.01$)')
    axes[0].set_title('Distribution of Surrogate Conservation Residuals', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Conservation Violation Magnitude ($|Q_{in} - Q_{out} - \Delta J|$)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_xlim(0, np.percentile(residuals_base, 95)) # 裁掉極端長尾讓圖好看
    axes[0].legend()

    # ==========================================
    # 5. 繪製防禦圖表 B：絕對誤差長尾分佈 (解釋 MAE vs RMSE)
    # ==========================================
    sns.histplot(errors_ours, ax=axes[1], bins=50, color='#2E7D32', kde=False)
    axes[1].set_title('Absolute Error Distribution (Jam Factor)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Absolute Prediction Error ($|J_{true} - \hat{J}|$)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    
    # 畫線標示 MAE 和 RMSE 的差異
    mae_val = np.mean(errors_ours)
    rmse_val = np.sqrt(np.mean(np.square(errors_ours)))
    axes[1].axvline(mae_val, color='blue', linestyle='--', linewidth=2, label=f'MAE ({mae_val:.3f})')
    axes[1].axvline(rmse_val, color='red', linestyle='--', linewidth=2, label=f'RMSE ({rmse_val:.3f})')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("Fig_Reviewer_Defense.png", dpi=300, bbox_inches='tight')
    print("🎨 成功生成防禦圖表：Fig_Reviewer_Defense.png")

if __name__ == "__main__":
    main()