"""
AOI_Net model definitions.

Only the classes actually used by train.py are kept:
    - AOI_Net  (temporal + structural encoders + gating router)
    - plus its dependency closure (GCN/TCN encoders, router, classifier head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except Exception:
    HAS_PYG = False


class GCNEncoder_PyG(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=64, out_dim=128, dropout=0.5, num_layers=2):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        # node -> graph representation
        g = global_mean_pool(x, batch)  # [B, out_dim]
        return g


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        # build symmetrically normalized A_hat = D^-1/2 (A+I) D^-1/2
        N = x.size(0)
        device = x.device
        self_loop = torch.arange(N, device=device)
        ei = edge_index
        ei = torch.cat([ei, torch.stack([self_loop, self_loop], dim=0)], dim=1)
        # degrees
        deg = torch.bincount(ei[0], minlength=N).float()
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        # normalized weights
        norm = deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]
        # sparse adjacency
        A = torch.sparse_coo_tensor(ei, norm, (N, N))
        xw = self.lin(x)
        out = torch.sparse.mm(A, xw) + self.bias
        return out


class GCNEncoder_Torch(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=64, out_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        # mean pooling to graph representation
        g = x.mean(dim=0, keepdim=True)  # [1, out_dim]
        return g


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, g):
        return self.net(g)


class GCNClassifier(nn.Module):
    def __init__(self, in_dim=8, gnn_dim=128, num_classes=2, hidden=64, dropout=0.5, use_pyg=True, gnn_layers=2):
        super().__init__()
        self.use_pyg = HAS_PYG and use_pyg
        if self.use_pyg:
            self.encoder = GCNEncoder_PyG(in_dim=in_dim, hidden_dim=64, out_dim=gnn_dim,
                                          dropout=dropout, num_layers=gnn_layers)
        else:
            self.encoder = GCNEncoder_Torch(in_dim=in_dim, hidden_dim=64, out_dim=gnn_dim,
                                            num_layers=gnn_layers, dropout=dropout)
        self.head = ClassifierHead(gnn_dim, num_classes=num_classes, hidden=hidden, dropout=dropout)

    def forward(self, x, edge_index, batch=None):
        if self.use_pyg:
            g = self.encoder(x, edge_index, batch)         # [B, gnn_dim]
            logits = self.head(g)
            return logits, g
        else:
            # Pure PyTorch: handles one graph per call
            g = self.encoder(x, edge_index)                # [1, gnn_dim]
            logits = self.head(g)                          # [1, C]
            return logits, g


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # preserve sequence length
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out + self.residual(x)  # residual connection


class TCNTimeSeriesClassifier(nn.Module):
    def __init__(self, num_features=2, num_classes=2, dropout=0.5, kernel_size=3, max_seq_length=32):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(num_features, 64, dilation=1, kernel_size=kernel_size),
            TCNBlock(64, 128, dilation=2, kernel_size=kernel_size),
            TCNBlock(128, 128, dilation=4, kernel_size=kernel_size),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)  # global pooling
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, num_features)
        x = x.permute(0, 2, 1)  # -> (batch, num_features, seq_len)
        x = self.tcn(x)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            x = x.masked_fill(mask == 0, -1e9)

        embeddings = self.pool(x).squeeze(-1)  # (batch, 128)
        x = F.relu(self.fc1(embeddings))
        x = self.dropout(x)
        out = self.fc2(x)
        return out, embeddings


class MoERouter(nn.Module):
    """
    Router: gate_in -> pi (per-sample per-expert weights) + router_logits
    top_k: 0 means dense-softmax; >0 means sparse top-k gating.
    """
    def __init__(self, in_dim: int, num_experts: int, hidden: int = 64, top_k: int = 0, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = int(top_k)
        self.temperature = float(temperature)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts)
        )

    def forward(self, gate_in: torch.Tensor):
        # gate_in: [B, D]
        router_logits = self.net(gate_in)  # [B, E]
        router_logits = router_logits / max(self.temperature, 1e-6)

        if self.top_k is None or self.top_k <= 0 or self.top_k >= self.num_experts:
            # dense softmax
            pi = F.softmax(router_logits, dim=-1)  # [B, E]
            return pi, router_logits

        # sparse top-k: keep only the top_k, set the rest to -inf before softmax
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        mask = torch.full_like(router_logits, float("-inf"))
        mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        pi = F.softmax(mask, dim=-1)  # [B, E]
        return pi, router_logits


class AOI_Net(nn.Module):
    """
    AOI_Net: temporal encoder + structural encoder + gating router,
    with representation-level fusion Z = TE*h + SE*g.

    h = temporal representation (128-dim)
    g = structural representation (gnn_dim)
    TE/SE = router weights pi (softmax gating over the two experts)
    Z = TE*h + SE*g  ->  fusion head -> logits

    Outputs: logits, emb, pi, aux_loss
    """
    def __init__(
        self,
        num_features=8,
        num_classes=2,
        gnn_dim=128,
        dropout=0.5,
        use_pyg=True,
        top_k=0,
        router_hidden=64,
        aux_load_balance=0.01,   # 0 means disabled
        temperature=1.0,
        gnn_layers=2,
    ):
        super().__init__()
        self.aux_load_balance = float(aux_load_balance)

        # ===== Experts (dual-expert: temporal + structural) =====
        self.expert_gnn = GCNClassifier(
            in_dim=num_features, gnn_dim=gnn_dim, num_classes=num_classes,
            hidden=64, dropout=dropout, use_pyg=use_pyg, gnn_layers=gnn_layers,
        )
        self.expert_cnn = TCNTimeSeriesClassifier(
            num_features=num_features, num_classes=num_classes, max_seq_length=32
        )

        # ===== Representation-level fusion (Z = TE*h + SE*g) =====
        # h (cnn_emb, 128) and g (gnn_emb, gnn_dim) must share one dim to be
        # summed; project g onto h's dim when they differ.
        self.cnn_emb_dim = 128
        if gnn_dim != self.cnn_emb_dim:
            self.g_proj = nn.Linear(gnn_dim, self.cnn_emb_dim)
        else:
            self.g_proj = nn.Identity()
        self.fusion_head = ClassifierHead(
            self.cnn_emb_dim, num_classes=num_classes, hidden=64, dropout=dropout
        )

        # ===== Router: gate on the concatenated aligned embeddings =====
        router_in_dim = self.cnn_emb_dim * 2  # cat([h, g_proj])
        self.router = MoERouter(
            in_dim=router_in_dim,
            num_experts=2,  # dual-expert: temporal (TE) + structural (SE)
            hidden=router_hidden,
            top_k=top_k,
            temperature=temperature,
        )

    def forward(self, x, edge_index, batch=None, seq=None, mask=None):
        assert seq is not None, "AOI_Net requires seq for the temporal (CNN) expert"

        # ---- Experts forward (embeddings) ----
        _gnn_logits, gnn_emb = self.expert_gnn(x, edge_index, batch=batch)  # [B, gnn_dim]
        _cnn_logits, cnn_emb = self.expert_cnn(seq, mask)                   # [B, 128]

        # ---- Align batch sizes (safety) ----
        min_b = min(gnn_emb.size(0), cnn_emb.size(0))
        gnn_emb, cnn_emb = gnn_emb[:min_b], cnn_emb[:min_b]

        h = cnn_emb                      # temporal representation
        g = self.g_proj(gnn_emb)         # structural representation (aligned to 128)

        # ---- Router: softmax gating over the two experts ----
        gate_in = torch.cat([h, g], dim=1)  # [B, 256]
        pi, router_logits = self.router(gate_in)   # pi: [B, 2] = [TE, SE]

        # ---- Representation-level fusion: Z = TE*h + SE*g ----
        emb = pi[:, 0:1] * h + pi[:, 1:2] * g   # [B, 128]
        logits = self.fusion_head(emb)          # [B, num_classes]

        # ---- Aux load-balance loss (prevents router collapse) ----
        aux_loss = None
        if self.aux_load_balance > 0:
            load = pi.mean(dim=0)  # [E]
            E = load.numel()
            uniform = torch.full_like(load, 1.0 / E)

            aux_loss = self.aux_load_balance * F.kl_div(
                (load + 1e-9).log(),
                uniform,
                reduction="batchmean",
            )

        return logits, emb, pi, aux_loss
