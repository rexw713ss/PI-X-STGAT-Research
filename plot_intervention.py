import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("🚀 啟動干預模擬圖表修復 (18.5% Drop Version)...")

    # ==========================================
    # 1. 準備更新後的數據 (對齊 LaTeX 內文)
    # ==========================================
    strategies = ['No Intervention\n(Baseline)', 'Random Intervention\n(20% Flow Cut)', 'XAI-Guided Intervention\n(20% Source-Metering)']
    jam_factors = [8.50, 8.48, 6.93] # 8.50 * (1 - 0.185) = 6.9275
    
    # 設定顏色：基準(灰)、隨機(橘紅)、XAI(深藍)
    colors = ['#9E9E9E', '#FF7043', '#1976D2']

    # ==========================================
    # 2. 開始繪圖 (IEEE 雙欄標準風格)
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # 繪製長條圖
    bars = ax.bar(strategies, jam_factors, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

    # 設定標籤與範圍
    ax.set_ylabel('Predicted Jam Factor at Target Node', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10.5) # Jam Factor 最大值通常為 10，留點空間放標註
    ax.tick_params(axis='both', which='major', labelsize=12)

    # ==========================================
    # 3. 標註數值與下降幅度 (Annotations)
    # ==========================================
    # 在每個柱子上方標註具體數值
    for bar, value in zip(bars, jam_factors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 畫出下降的箭頭與 18.5% 的文字框
    # 從 Baseline 的高度指到 XAI-Guided 的高度
    baseline_val = jam_factors[0]
    xai_val = jam_factors[2]
    x_baseline = bars[0].get_x() + bars[0].get_width()/2
    x_xai = bars[2].get_x() + bars[2].get_width()/2

    # 畫一條水平虛線作為基準線
    ax.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.7)

    # 加上下降箭頭
    ax.annotate('', xy=(x_xai, xai_val + 0.5), xytext=(x_xai, baseline_val),
                arrowprops=dict(facecolor='#2E7D32', shrink=0.0, width=2, headwidth=10, edgecolor='none'))

    # 加上醒目的 -18.5% 標籤
    ax.text(x_xai + 0.15, (baseline_val + xai_val)/2, '-18.5% Drop', 
            color='#2E7D32', fontsize=14, fontweight='bold', ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#2E7D32", lw=1.5))

    # ==========================================
    # 4. 輸出與存檔
    # ==========================================
    plt.title('Heuristic Intervention Simulation at Target Node (E_6th_St)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', visible=False) # 關閉 X 軸的網格讓畫面更乾淨
    
    plt.tight_layout()
    plt.savefig("Fig_Intervention_Simulation.png", dpi=300, bbox_inches='tight')
    print("🎨 成功生成修正版圖表：Fig_Intervention_Simulation.png")

if __name__ == "__main__":
    main()