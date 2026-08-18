# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

    

def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj / row_sum


class DiffusionGraphConv(nn.Module):
    def __init__(self, channels: int, supports: int, order: int = 2, dropout: float = 0.1):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.mlp = nn.Conv2d(channels * (supports * order + 1), channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, support_list: list[torch.Tensor]) -> torch.Tensor:
        outputs = [x]
        for support in support_list:
            x_k = torch.einsum("bcnt,nm->bcmt", x, support)
            outputs.append(x_k)
            for _ in range(2, self.order + 1):
                x_k = torch.einsum("bcnt,nm->bcmt", x_k, support)
                outputs.append(x_k)
        h = torch.cat(outputs, dim=1)
        h = self.mlp(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class GraphWaveNet_Baseline(nn.Module):
    """
    Compact Graph WaveNet baseline with gated dilated temporal convolutions and
    static plus adaptive graph diffusion.
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        adj_matrix: torch.Tensor,
        embed_dim: int = 10,
        blocks: int = 2,
        layers: int = 3,
        kernel_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.register_buffer("static_adj", _normalize_adj(adj_matrix))
        self.node_emb_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.node_emb_2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.1)

        self.start_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.graph_convs = nn.ModuleList()

        for _ in range(blocks):
            for layer in range(layers):
                dilation = 2 ** layer
                self.filter_convs.append(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation))
                )
                self.gate_convs.append(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation))
                )
                self.residual_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(hidden_dim))
                self.graph_convs.append(DiffusionGraphConv(hidden_dim, supports=2, order=2, dropout=dropout))

        self.end_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=(1, 1))

    def _supports(self) -> list[torch.Tensor]:
        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_emb_1, self.node_emb_2)), dim=1)
        return [self.static_adj, adaptive_adj]

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 2, 1)
        h = self.start_conv(x)
        skip = 0
        supports = self._supports()

        for i in range(self.blocks * self.layers):
            residual = h
            dilation = self.filter_convs[i].dilation[1]
            h_padded = F.pad(h, (dilation * (self.kernel_size - 1), 0, 0, 0))
            filter_out = torch.tanh(self.filter_convs[i](h_padded))
            gate_out = torch.sigmoid(self.gate_convs[i](h_padded))
            h = filter_out * gate_out

            skip_out = self.skip_convs[i](h)
            skip = skip_out if isinstance(skip, int) else skip[..., -skip_out.size(3):] + skip_out

            h = self.graph_convs[i](h, supports)
            h = self.residual_convs[i](h)
            h = h + residual[..., -h.size(3):]
            h = self.bn[i](h)

        h = F.relu(skip)
        h = F.relu(self.end_conv_1(h))
        h = self.end_conv_2(h)
        pred = h[..., -1].permute(0, 2, 1)
        return pred, None


class MixProp(nn.Module):
    def __init__(self, channels: int, depth: int = 2, alpha: float = 0.05, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.dropout = dropout
        self.mlp = nn.Conv2d(channels * (depth + 1), channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj = _normalize_adj(adj + torch.eye(adj.size(0), device=adj.device))
        h = x
        outputs = [h]
        for _ in range(self.depth):
            h = self.alpha * x + (1.0 - self.alpha) * torch.einsum("bcnt,nm->bcmt", h, adj)
            outputs.append(h)
        h = self.mlp(torch.cat(outputs, dim=1))
        return F.dropout(h, p=self.dropout, training=self.training)


class MTGNN_Baseline(nn.Module):
    """
    Compact MTGNN baseline with learned graph construction, dilated temporal
    convolutions, and mix-hop graph propagation.
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        embed_dim: int = 10,
        layers: int = 3,
        kernel_size: int = 2,
        top_k: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = layers
        self.kernel_size = kernel_size
        self.top_k = min(top_k, num_nodes)
        self.dropout = dropout

        self.node_emb_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.node_emb_2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.lin_1 = nn.Linear(embed_dim, embed_dim)
        self.lin_2 = nn.Linear(embed_dim, embed_dim)

        self.start_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.mix_props = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        for layer in range(layers):
            dilation = 2 ** layer
            self.filter_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation))
            )
            self.gate_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation))
            )
            self.mix_props.append(MixProp(hidden_dim, depth=2, alpha=0.05, dropout=dropout))
            self.residual_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(hidden_dim))

        self.end_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=(1, 1))

    def _learned_adj(self) -> torch.Tensor:
        emb_1 = torch.tanh(self.lin_1(self.node_emb_1))
        emb_2 = torch.tanh(self.lin_2(self.node_emb_2))
        scores = F.relu(torch.mm(emb_1, emb_2.transpose(0, 1)) - torch.mm(emb_2, emb_1.transpose(0, 1)))

        if self.top_k < self.num_nodes:
            values, indices = torch.topk(scores, self.top_k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, indices, values)
            scores = mask

        return F.softmax(scores, dim=1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 2, 1)
        h = self.start_conv(x)
        skip = 0
        adj = self._learned_adj()

        for i in range(self.layers):
            residual = h
            dilation = self.filter_convs[i].dilation[1]
            h_padded = F.pad(h, (dilation * (self.kernel_size - 1), 0, 0, 0))
            filter_out = torch.tanh(self.filter_convs[i](h_padded))
            gate_out = torch.sigmoid(self.gate_convs[i](h_padded))
            h = filter_out * gate_out

            skip_out = self.skip_convs[i](h)
            skip = skip_out if isinstance(skip, int) else skip[..., -skip_out.size(3):] + skip_out

            h = self.mix_props[i](h, adj)
            h = self.residual_convs[i](h)
            h = h + residual[..., -h.size(3):]
            h = self.bn[i](h)

        h = F.relu(skip)
        h = F.relu(self.end_conv_1(h))
        h = self.end_conv_2(h)
        pred = h[..., -1].permute(0, 2, 1)
        return pred, None


# ==========================================
# 0. 基礎 Baseline: LSTM
# ==========================================
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # batch_first=True 代表輸入的維度是 (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 接收到的 x 形狀: (Batch, Seq_len, Nodes, Features)
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # 為了讓 LSTM 處理，我們把 Nodes 融合到 Batch 維度裡
        # 變成: (Batch * Nodes, Seq_len, Features)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, num_features)
        
        # LSTM 輸出
        lstm_out, _ = self.lstm(x_reshaped)
        
        # 只取最後一個時間步 (預測未來) 的隱藏狀態
        last_hidden = lstm_out[:, -1, :] # 形狀: (Batch * Nodes, Hidden_Dim)
        
        # 全連接層預測
        pred = self.fc(last_hidden)      # 形狀: (Batch * Nodes, Output_Dim)
        
        # 轉換回原本的空間形狀: (Batch, Nodes, Output_Dim)
        pred = pred.reshape(batch_size, num_nodes, -1)
        
        # Baseline 沒有 Attention 矩陣，為了格式統一回傳 None
        return pred

# ==========================================
# 1. 升級版 GAT Layer (加入殘差連接)
# ==========================================
class EnhancedGATLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.attn_src = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # 【優化 1】殘差連接 (Residual Connection)
        # 如果輸入維度與輸出維度不同，用一個 Linear 對齊；若相同則直接相加
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor, combined_adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch * Seq_len, Nodes, Features)
        # combined_adj: (Batch, Nodes, Nodes) 或 (Nodes, Nodes)
        B, N, _ = x.shape
        h = self.fc(x)

        src_score = self.attn_src(h)
        dst_score = self.attn_dst(h)

        scores = src_score + dst_score.transpose(1, 2)
        scores = self.leaky_relu(scores)

        # 這裡我們不使用硬性的 0/1 Mask，而是將 combined_adj 作為一個 soft bias 加進去
        # combined_adj 包含了實體圖與自適應圖的資訊
        if combined_adj.dim() == 2:
            combined_adj = combined_adj.unsqueeze(0) # 廣播到 Batch 維度
            
        # 讓原本不相連的地方注意力極低，有連接（或隱藏連接）的地方保留
        scores = scores + torch.log(combined_adj + 1e-9) 
        attention = torch.softmax(scores, dim=-1)

        out = torch.bmm(attention, h)
        
        # 加上殘差連接，防止過度平滑 (Oversmoothing)
        out = out + self.residual(x)
        return out, attention

# ==========================================
# 終極版 PI-X-STGAT (支援消融實驗開關)
# ==========================================
class PI_X_STGAT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        gat_dim: int,
        gru_dim: int,
        output_dim: int,
        adj_matrix: torch.Tensor,
        embed_dim: int = 10,
        use_napl: bool = True,
        node_embed_scale: float = 1.0,
    ):
        super().__init__()
        self.register_buffer('static_adj', adj_matrix)
        self.use_napl = use_napl # 👈 加入開關
        
        if self.use_napl:
            self.node_emb_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * node_embed_scale)
            self.node_emb_2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * node_embed_scale)
        
        self.gat = EnhancedGATLayer(input_dim, gat_dim)
        self.gru = nn.GRU(gat_dim, gru_dim, batch_first=True)
        self.fc = nn.Linear(gru_dim, output_dim)

    def forward(self, x: torch.Tensor):
        B, S, N, F_dim = x.shape
        
        # 👈 根據開關決定是否使用自適應圖
        if self.use_napl:
            import torch.nn.functional as F
            adp_adj = F.softmax(F.relu(torch.mm(self.node_emb_1, self.node_emb_2)), dim=1)
            combined_adj = self.static_adj + adp_adj
        else:
            combined_adj = self.static_adj
            
        x_flat = x.reshape(B * S, N, F_dim)
        gat_out, attention = self.gat(x_flat, combined_adj) 
        
        gat_out = gat_out.reshape(B, S, N, -1).permute(0, 2, 1, 3).reshape(B * N, S, -1)
        gru_out, _ = self.gru(gat_out)
        
        last_hidden = gru_out[:, -1, :]
        pred = self.fc(last_hidden)     
        pred = pred.reshape(B, N, -1)   
        
        attention = attention.reshape(B, S, N, N)[:, -1, :, :]
        return pred, attention


class VanillaSTGAT(PI_X_STGAT):
    """Backward-compatible STGAT variant used by archived analysis scripts.

    Historical scripts used a five-argument constructor without node-adaptive
    parameter learning. This wrapper keeps those scripts runnable without
    changing the final PI-X-STGAT implementation.
    """

    def __init__(
        self,
        input_dim: int,
        gat_dim: int,
        gru_dim: int,
        output_dim: int,
        adj_matrix: torch.Tensor,
    ):
        super().__init__(
            num_nodes=adj_matrix.shape[0],
            input_dim=input_dim,
            gat_dim=gat_dim,
            gru_dim=gru_dim,
            output_dim=output_dim,
            adj_matrix=adj_matrix,
            use_napl=False,
        )

# ==========================================
# 3. 準備給你的強力 Baseline：AGCRN (簡化相容版)
# ==========================================
class AGCRN_Baseline(nn.Module):
    """
    AGCRN 的精神：完全不依賴你給的 adj_matrix，
    全部依靠內部學習的 Node Embeddings 來生成權重進行卷積。
    這裡設計成與 PI-X-STGAT 的輸入形狀完全相容，方便你直接切換做實驗。
    """
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, output_dim: int, embed_dim: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # AGCRN 的靈魂：NAPL (Node Adaptive Parameter Learning)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        
        # 利用 Embeddings 來生成 GRU/GCN 的權重矩陣
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, input_dim, hidden_dim))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, hidden_dim))
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Seq_len, Nodes, Features)
        B, S, N, F = x.shape
        x_flat = x.reshape(B * S, N, F)
        
        # 動態生成權重: E * W_pool
        # weight 形狀: (Nodes, Input_dim, Hidden_dim)
        weights = torch.einsum('ne, eih -> nih', self.node_embeddings, self.weights_pool)
        bias = torch.matmul(self.node_embeddings, self.bias_pool)
        
        # 自適應圖卷積
        # x_flat: (BS, N, F), weights: (N, F, H) -> 產出 (BS, N, H)
        gcn_out = torch.einsum('bnf, nfh -> bnh', x_flat, weights) + bias
        gcn_out = torch.relu(gcn_out)
        
        # 丟入時間模型
        gcn_out = gcn_out.reshape(B, S, N, -1).permute(0, 2, 1, 3).reshape(B * N, S, -1)
        gru_out, _ = self.gru(gcn_out)
        
        last_hidden = gru_out[:, -1, :]
        pred = self.fc(last_hidden)
        pred = pred.reshape(B, N, -1)
        
        # AGCRN 沒有 Attention 矩陣，回傳 None
        return pred, None
    
# ==========================================
# 4. 經典時空圖 Baseline: STGCN
# ==========================================
class STGCN_Baseline(nn.Module):
    """
    標準的時空圖卷積網路 (GCN + GRU)
    使用靜態的物理鄰接矩陣進行特徵聚合。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, adj_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer('adj', adj_matrix)
        
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Seq_len, Nodes, Features)
        B, S, N, F_dim = x.shape
        
        # --- 1. 計算標準化的圖卷積鄰接矩陣 (A_norm) ---
        # 加上單位矩陣 (Self-loop)，並做列歸一化 (Row normalization)
        adj_with_loop = self.adj + torch.eye(N, device=x.device)
        row_sum = adj_with_loop.sum(dim=1, keepdim=True)
        adj_norm = adj_with_loop / row_sum  # (N, N)
        
        # --- 2. 空間圖卷積 (GCN) ---
        x_flat = x.reshape(B * S, N, F_dim)
        h = self.fc(x_flat) # 特徵轉換: (BS, N, hidden_dim)
        
        # GCN 矩陣乘法: A_norm * H
        gcn_out = torch.bmm(adj_norm.unsqueeze(0).expand(B * S, N, N), h)
        gcn_out = torch.relu(gcn_out)
        
        # --- 3. 時間特徵提取 (GRU) ---
        gcn_out = gcn_out.reshape(B, S, N, -1).permute(0, 2, 1, 3).reshape(B * N, S, -1)
        gru_out, _ = self.gru(gcn_out)
        
        # 取最後一個時間步
        last_hidden = gru_out[:, -1, :]
        
        # === 4. 預測未來狀態 ===
        pred = self.out_fc(last_hidden)     
        pred = pred.reshape(B, N, -1)   
        
        # Baseline 沒有動態注意力矩陣，回傳 None
        return pred, None
