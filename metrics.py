# metrics.py
import torch
import torch.nn as nn

class PhysicsLossCheck(nn.Module):
    def __init__(self, adj_matrix: torch.Tensor):
        super().__init__()
        # 將鄰接矩陣註冊為 Buffer，方便 GPU 運算
        self.register_buffer("adj", adj_matrix)

    def forward(self, pred: torch.Tensor, last_real_jam: torch.Tensor) -> torch.Tensor:
        """
        pred: (Batch, Nodes, 2) -> 預測的 [Jam_Factor, Speed_kmh]
        last_real_jam: (Batch, Nodes) -> 上一個時間步的真實 Jam_Factor
        """
        # 1. 取出預測的密度 (Jam Factor 已經在 dataset 縮放到 0~1)
        # 為了避免負數，加上 ReLU 確保密度大於等於 0
        density_pred = torch.relu(pred[..., 0]) 
        
        # 2. 取出預測的速度 (因為是 Z-score 會有負數)
        # 巧妙解法：用 Sigmoid 將相對車速映射到 0~1 之間，作為流速係數
        speed_pred = torch.sigmoid(pred[..., 1]) 
        
        # 3. 計算預測的相對流量 (Flow = Speed * Density)
        flow_pred = speed_pred * density_pred # (Batch, Nodes)
        
        # 4. 透過空間鄰接矩陣計算流出 (Q_out) 與流入 (Q_in)
        # flow_pred 形狀轉換為 (Batch, Nodes, 1) 來做矩陣乘法
        flow_unsqueeze = flow_pred.unsqueeze(-1)
        
        # Q_out = A * Flow (鄰居吸收了我的流量)
        q_out = torch.matmul(self.adj, flow_unsqueeze).squeeze(-1)
        
        # Q_in = A^T * Flow (我吸收了鄰居的流量)
        q_in = torch.matmul(self.adj.t(), flow_unsqueeze).squeeze(-1)
        
        # 5. 計算密度的變化量 (Delta Jam = 預測未來密度 - 現在真實密度)
        delta_jam = density_pred - last_real_jam
        
        # 6. 物理守恆稽核：淨流量 (流入 - 流出) 應該與密度變化成正比
        # 也就是： (Q_in - Q_out) - Delta_Jam 應該要趨近於 0
        residual = torch.abs((q_in - q_out) - delta_jam)
        
        # 回傳平均的物理誤差
        return residual.mean()