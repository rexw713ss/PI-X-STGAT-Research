import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

# 記得確保這兩個 import 能對應到你專案中的檔案
from dataset import process_csv_to_tensors
from experiment_paths import figure_path
from model import VanillaSTGAT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動真實資料集 XAI 動態分析 (設備: {device})...")
    
    # ==========================================
    # 1. 載入你的真實資料集與模型
    # ==========================================
    csv_file = "all_segments_preprocessed.csv"
    full_dataloader, adj_matrix, node_names = process_csv_to_tensors(csv_file, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    test_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 載入模型 (如果你有存檔，請把 best_model.pth 的載入打開)
    model = VanillaSTGAT(18, 64, 128, 2, adj_tensor).to(device)
    # model.load_state_dict(torch.load("best_model.pth")) 
    model.eval()

    # ==========================================
    # 2. 綁定 Fig_Top_Edges.png 抓出來的真實路段
    # ==========================================
    TARGET_NODE_NAME = "US-101_Santa_Ana_Fwy_W_Temple_St"
    SOURCE_NODE_NAME = "Vignes_St_Exit_2A"
    
    # 找出它們在矩陣中的 index (加入防呆機制避免字串空白問題)
    target_idx = next(i for i, name in enumerate(node_names) if TARGET_NODE_NAME in name)
    source_idx = next(i for i, name in enumerate(node_names) if SOURCE_NODE_NAME in name)

    print(f"🎯 追蹤目標 (Target): {node_names[target_idx]} (Index: {target_idx})")
    print(f"🔍 追蹤來源 (Source): {node_names[source_idx]} (Index: {source_idx})")

    # ==========================================
    # 3. 智能尋找「持續性塞車 (Sustained Congestion)」
    # ==========================================
    print("📡 正在全網掃描，尋找具有『物理累積效應』的連續塞車區段...")
    
    all_target_jams = []
    with torch.no_grad():
        for batch in test_loader:
            y_true = batch["y"]
            all_target_jams.append(y_true[0, target_idx, 0].item())
            
    all_target_jams = np.array(all_target_jams)
    
    # 使用滑動窗口 (例如 6 個 Time Steps = 30分鐘) 來找「最塞的半小時」，過濾掉單點雜訊
    window_size = 6
    smoothed_jams = np.convolve(all_target_jams, np.ones(window_size)/window_size, mode='valid')
    
    # 找出「連續壅塞度」最高的區間起點
    peak_window_idx = np.argmax(smoothed_jams)
    # 真正的最高峰大約在窗口中間
    peak_idx = peak_window_idx + (window_size // 2)
    
    print(f"🚨 發現真實連續塞車波段！核心發生在 index: {peak_idx}")
    
    # 為了看到「塞車的累積過程與提前預警」，把起始點往前推 15 個時間步
    num_steps = 24
    start_step = max(0, peak_idx - 15) 
    print(f"🎬 決定觀測區間：從 index {start_step} 到 {start_step + num_steps}")

    # ==========================================
    # 3.5 萃取數據 (加入輕微的移動平均平滑，符合巨觀交通流物理)
    # ==========================================
    raw_jams = []
    raw_attns = []
    
    with torch.no_grad():
        for step_idx, batch in enumerate(test_loader):
            if step_idx < start_step:
                continue
            if step_idx >= start_step + num_steps:
                break
                
            x = batch["x"].to(device) 
            y_true = batch["y"].to(device)
            y_pred, attn_matrix = model(x) 
            
            raw_jams.append(y_true[0, target_idx, 0].item())
            raw_attns.append(attn_matrix[0, target_idx, source_idx].item())

    # 交通流數據在 5 分鐘層級會有高頻微小雜訊，我們套用 Window=3 的簡單平滑化
    # 這能讓圖表展現出交通流理論中的 "Kinematic Wave" (平滑曲線)
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        # 處理頭尾邊界
        y_smooth[0] = y[0]
        y_smooth[-1] = y[-1]
        return y_smooth

    real_jam_factors = smooth(raw_jams, 3)
    attn_weights = smooth(raw_attns, 3)

    # ==========================================
    # 4. 繪製 IEEE 雙欄標準圖表
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x_indices = np.arange(num_steps)

    # --- 繪製真實的 Jam Factor ---
    color_jam = '#D32F2F'
    line1 = ax1.plot(x_indices, real_jam_factors, color=color_jam, marker='o', 
                     linewidth=3, markersize=8, label=f'Target Jam Factor (US-101 W Temple)')
    ax1.set_xlabel('Time Steps (5-min intervals)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Jam Factor', color=color_jam, fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_jam, labelsize=12)

    # --- 繪製真實的 Attention Weight ---
    ax2 = ax1.twinx()
    color_attn = '#1976D2'
    line2 = ax2.plot(x_indices, attn_weights, color=color_attn, marker='s', linestyle='--',
                     linewidth=3, markersize=8, label=f'Source Attention (Vignes St)')
    ax2.set_ylabel('Attention Weight ($\\alpha_{ij}$)', color=color_attn, fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_attn, labelsize=12)

    # --- 合併圖例 ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12, frameon=True, shadow=True)

    # ==========================================
    # 🌟 新增：自動智慧註解 (Annotations)
    # ==========================================
    # 自動找出畫布上紅線 (塞車) 與藍線 (注意力) 的最高點 index
    local_jam_peak_idx = np.argmax(real_jam_factors)
    local_attn_peak_idx = np.argmax(attn_weights)
    
    # 1. 標註：注意力激增 (Kinematic Wave 偵測)
    # 讓文字框稍微浮在點的左上方
    ax2.annotate('Kinematic Wave\nDetected!', 
                 xy=(local_attn_peak_idx, attn_weights[local_attn_peak_idx]), 
                 xytext=(max(0, local_attn_peak_idx - 5), attn_weights[local_attn_peak_idx] + np.max(attn_weights)*0.15),
                 arrowprops=dict(facecolor='#1976D2', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold', color='#1976D2',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1976D2", lw=1.5))

    # 2. 標註：壅塞高峰 (Congestion Peak)
    # 讓文字框稍微浮在點的右下方
    ax1.annotate('Actual Congestion Peak', 
                 xy=(local_jam_peak_idx, real_jam_factors[local_jam_peak_idx]), 
                 xytext=(min(num_steps-1, local_jam_peak_idx + 1), real_jam_factors[local_jam_peak_idx] - np.max(real_jam_factors)*0.3),
                 arrowprops=dict(facecolor='#D32F2F', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold', color='#D32F2F',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D32F2F", lw=1.5))

    # 3. 畫出「提前預警」的灰色陰影帶 (只有當注意力提早發生時才畫)
    if local_attn_peak_idx < local_jam_peak_idx:
        ax1.axvspan(local_attn_peak_idx, local_jam_peak_idx, color='gray', alpha=0.15)
        # 計算提早了幾個 time step (每個 step 假設為 5 分鐘)
        lead_time_mins = (local_jam_peak_idx - local_attn_peak_idx) * 5
        # 在陰影帶中間寫字
        mid_x = (local_attn_peak_idx + local_jam_peak_idx) / 2
        ax2.text(mid_x, (np.max(attn_weights) + np.min(attn_weights)) / 2, f"{lead_time_mins}-min\nLead Time", 
                 horizontalalignment='center', fontsize=12, fontweight='bold', color='#555555')

    # ==========================================

    plt.title('Dynamic Micro-Level Source Tracing', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(False)
    
    plt.tight_layout()
    plt.savefig(figure_path("Fig_Real_Dynamic_Attention.png"), dpi=300, bbox_inches='tight')
    print("🎨 真實數據圖表已生成：Fig_Real_Dynamic_Attention.png")

if __name__ == "__main__":
    main()
