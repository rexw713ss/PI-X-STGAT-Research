import matplotlib.pyplot as plt
import numpy as np

def main():
    print("🎨 正在生成最新版效能對比圖 (包含 AGCRN)...")

    # ==========================================
    # 1. 準備最新版的 Table 1 數據
    # ==========================================
    models = ['HA', 'LSTM', 'STGCN', 'ASTGCN', 'AGCRN', 'PI-X-STGAT\n(Ours)']
    
    # 最新的 Jam MAE 數據 (我們是 0.0594!)
    jam_mae = [0.6287, 0.1012, 0.0897, 0.0805, 0.0930, 0.0594]
    
    # 最新的 Jam RMSE 數據 (我們是 0.6228)
    jam_rmse = [1.9567, 0.6187, 0.6202, 0.6214, 0.6242, 0.6228]

    # 設定顏色：前 5 個對手用低調的灰色/淺藍色，我們的模型用強烈的深藍色
    colors = ['#B0BEC5', '#90CAF9', '#64B5F6', '#42A5F5', '#1E88E5', '#1565C0']
    edge_colors = ['#78909C', '#64B5F6', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']

    # ==========================================
    # 2. 開始繪圖 (1x2 的子圖佈局)
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.arange(len(models))
    width = 0.6

    # --- 左圖：Jam MAE ---
    bars1 = axes[0].bar(x_pos, jam_mae, width=width, color=colors, edgecolor=edge_colors, linewidth=1.5)
    axes[0].set_title('Jam Factor Prediction: MAE (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, fontsize=11, fontweight='bold')
    
    # 在柱子上方標註數值 (把我們自己的數字加粗標紅)
    for i, bar in enumerate(bars1):
        yval = bar.get_height()
        font_weight = 'bold' if i == 5 else 'normal'
        font_color = '#B71C1C' if i == 5 else 'black' # 我們的數字用暗紅色凸顯
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', 
                     ha='center', va='bottom', fontsize=11, fontweight=font_weight, color=font_color)

    # --- 右圖：Jam RMSE ---
    bars2 = axes[1].bar(x_pos, jam_rmse, width=width, color=colors, edgecolor=edge_colors, linewidth=1.5)
    axes[1].set_title('Jam Factor Prediction: RMSE (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Root Mean Square Error', fontsize=12)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models, fontsize=11, fontweight='bold')
    
    # 在柱子上方標註數值
    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        font_weight = 'bold' if i == 5 else 'normal'
        font_color = '#B71C1C' if i == 5 else 'black'
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', 
                     ha='center', va='bottom', fontsize=11, fontweight=font_weight, color=font_color)

    # ==========================================
    # 3. 畫面優化與存檔
    # ==========================================
    # 將 Y 軸的網格線調淡，關閉 X 軸網格
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.grid(axis='x', visible=False)
        # 特別處理 HA 數值太大導致其他柱子被壓扁的問題
        if ax == axes[0]:
            ax.set_ylim(0, 0.7)  # MAE 範圍
        else:
            ax.set_ylim(0, 2.2)  # RMSE 範圍

    plt.tight_layout()
    plt.savefig("Fig_New_Performance_Comparison.png", dpi=300, bbox_inches='tight')
    print("✅ 成功生成：Fig_New_Performance_Comparison.png")

if __name__ == "__main__":
    main()