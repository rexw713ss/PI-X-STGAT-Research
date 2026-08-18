# dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from constants import BASE_FEATURES, TRAFFIC_TARGETS

def build_adjacency_matrix(latitudes: np.ndarray, longitudes: np.ndarray, k_neighbors: int = 6) -> np.ndarray:
    """利用經緯度計算 Haversine 距離，建立 KNN 空間鄰接矩陣"""
    num_nodes = len(latitudes)
    if num_nodes <= 1:
        return np.eye(1, dtype=np.float32)

    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)

    # 計算兩兩節點的距離矩陣
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, a_min=1e-9, a_max=None)))
    distance_km = 6371.0 * c

    # 建立 KNN 矩陣
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        # 找出距離最近的 k 個鄰居 (排除自己)
        neighbor_idx = np.argsort(distance_km[i])[1:k_neighbors+1]

        local_dist = distance_km[i, neighbor_idx]
        dist_scale = np.std(local_dist) + 1e-3
        weights = np.exp(-(local_dist / dist_scale)) # 距離越近，權重越接近 1

        adjacency[i, neighbor_idx] = weights
        adjacency[i, i] = 1.0 # 加上自我連接 (Self-loop)

    # 行歸一化 (Row normalization)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.clip(row_sum, a_min=1e-6, a_max=None)

    return adjacency.astype(np.float32)

class TrafficDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int, horizon: int):
        """
        features: shape (Num_Timestamps, Num_Nodes, Num_Features)
        targets: shape (Num_Timestamps, Num_Nodes, Num_Targets)
        """
        self.features = features
        self.targets = targets
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_samples = len(features) - seq_len - horizon + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # X: 過去的歷史序列
        x = self.features[idx : idx + self.seq_len]
        # Y: 未來要預測的目標 (這裡取第 horizon 步)
        target_idx = idx + self.seq_len + self.horizon - 1
        y = self.targets[target_idx]

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }

def process_csv_to_tensors(csv_path: str, seq_len: int = 12, horizon: int = 3) -> Tuple[DataLoader, np.ndarray, List[str]]:
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 確保依照時間與路段排序
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df = df.sort_values(by=['Date_Time', 'Road_File']).reset_index(drop=True)

    # 抓取唯一的路段與時間戳記
    unique_nodes = sorted(df['Road_File'].unique())
    unique_times = sorted(df['Date_Time'].unique())

    num_nodes = len(unique_nodes)
    num_times = len(unique_times)
    num_features = len(BASE_FEATURES)
    num_targets = len(TRAFFIC_TARGETS)

    print(f"Found {num_nodes} nodes and {num_times} timestamps.")

    # 建立映射表
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    time_to_idx = {time: i for i, time in enumerate(unique_times)}

    # 初始化空的 Numpy 陣列
    feature_tensor = np.zeros((num_times, num_nodes, num_features), dtype=np.float32)
    target_tensor = np.zeros((num_times, num_nodes, num_targets), dtype=np.float32)

    # 填充數據 (這步可能需要幾秒鐘)
    print("Building Spatio-Temporal Tensors...")
    time_indices = df['Date_Time'].map(time_to_idx).values
    node_indices = df['Road_File'].map(node_to_idx).values

    for f_idx, feature_name in enumerate(BASE_FEATURES):
        if feature_name in df.columns:
            feature_tensor[time_indices, node_indices, f_idx] = df[feature_name].values

    for t_idx, target_name in enumerate(TRAFFIC_TARGETS):
        if target_name in df.columns:
            target_tensor[time_indices, node_indices, t_idx] = df[target_name].values

    # 【關鍵修正】：Jam_Factor 在 CSV 中是 0~10，但其他特徵是負數的標準化值
    # 我們將 Jam_Factor 縮放到 0~1，避免神經網路在計算 Loss 時被 Jam_Factor 主導
    jam_idx_in_target = TRAFFIC_TARGETS.index("Jam_Factor")
    target_tensor[:, :, jam_idx_in_target] = target_tensor[:, :, jam_idx_in_target] / 10.0

    # 建立鄰接矩陣 (取出所有節點的經緯度)
    node_meta = df.drop_duplicates(subset=['Road_File']).sort_values('Road_File')
    lats = node_meta['Latitude'].values
    lons = node_meta['Longitude'].values
    adjacency = build_adjacency_matrix(lats, lons)

    # 建立 PyTorch DataLoader
    dataset = TrafficDataset(feature_tensor, target_tensor, seq_len=seq_len, horizon=horizon)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # 測試階段不打亂

    return dataloader, adjacency, unique_nodes

# ==========================================
# 🛑 第一階段健康檢查 (Sanity Check)
# ==========================================
if __name__ == "__main__":
    # 請將 "your_data.csv" 換成你截圖的那份 CSV 檔案名稱
    CSV_FILE = "all_segments_preprocessed.csv"

    try:
        dataloader, adj_matrix, nodes = process_csv_to_tensors(CSV_FILE)

        print("\n✅ 資料預處理成功！健康檢查報告：")
        print("-" * 40)
        print(f"路網節點數量 (Nodes): {len(nodes)}")
        print(f"空間鄰接矩陣形狀 (Adjacency): {adj_matrix.shape}")

        # 抓取一個 Batch 的資料來檢查
        batch = next(iter(dataloader))
        x = batch["x"]
        y = batch["y"]

        print(f"輸入特徵 X 形狀 (Batch, Seq_len, Nodes, Features): {x.shape}")
        print(f"預測目標 Y 形狀 (Batch, Nodes, Targets): {y.shape}")

        # 檢查是否有 NaN 壞資料
        if torch.isnan(x).any() or torch.isnan(y).any():
            print("⚠️ 警告：張量中包含 NaN (空值)！請檢查 CSV 是否有缺漏。")
        else:
            print("✅ 張量數值健康，沒有 NaN。")

        print("-" * 40)
        print("太棒了！地基已經打好了，隨時可以準備蓋神經網路！")

    except FileNotFoundError:
        print(f"❌ 找不到檔案 {CSV_FILE}，請確認檔名是否正確。")
    except KeyError as e:
        print(f"❌ CSV 中找不到欄位: {e}，請檢查 constants.py 中的 BASE_FEATURES 是否與 CSV 標題完全一致。")