"""
AOI_Net training script.

Command-line driven training of the AOI_Net model (AOI-graph temporal+structural
fusion) with k-fold cross-validation. The dataset is NOT bundled: point --data-dir at a
directory of .xlsx scanpath files (see dataloader.py for the expected columns).

Examples (run from the repository root Code/):
    python -m AOI_Net.train --data-dir /path/to/data --folds 5 --epochs 50
    python -m AOI_Net.train --data-dir /path/to/data --eta 0.5
    python -m AOI_Net.train --data-dir /path/to/data --folds 10 --seed 2026 --output cv.json
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Imports
#   Run from the repository root (Code/):  python -m AOI_Net.train
#   dataloader.py lives in <code_root>/; model.py is in the AOI_Net package.
# ---------------------------------------------------------------------------
from AOI_Net.model import AOI_Net  # noqa: E402
from dataloader import load_paradigms  # noqa: E402

# ---------------------------------------------------------------------------
# PyG detection
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None
    GeoDataLoader = None
    print("[WARN] torch_geometric not installed; PyG graph batching is unavailable.")


def seed_everything(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Graph construction
# ===========================================================================

def build_temporal_edges(num_nodes: int, device=None):
    """Build undirected temporal edges t <-> t+1."""
    if device is None:
        device = "cpu"
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    src = torch.arange(0, num_nodes - 1, dtype=torch.long, device=device)
    dst = torch.arange(1, num_nodes, dtype=torch.long, device=device)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def build_aoi_edges(aoi_names: torch.Tensor):
    """
    Connect nodes that share the same AOI (integer encoded).
    aoi_names: [N], dtype=torch.long, may be on cuda
    Return undirected edge_index: [2, E]
    """
    device = aoi_names.device
    if aoi_names.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    unique_aoi = torch.unique(aoi_names)

    for a in unique_aoi:
        idx = torch.nonzero(aoi_names == a, as_tuple=False).view(-1)  # [M]
        if idx.numel() <= 1:
            continue
        src = idx.unsqueeze(1).expand(-1, idx.numel()).reshape(-1)
        dst = idx.unsqueeze(0).expand(idx.numel(), -1).reshape(-1)
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        ei = torch.stack([src, dst], dim=0)  # same device as aoi_names
        edge_index = torch.cat([edge_index, ei], dim=1)

    # deduplicate
    if edge_index.numel() > 0:
        edge_index = edge_index.t().unique(dim=0).t()
    return edge_index


def seq_mask_to_graph(seq: torch.Tensor, mask: torch.Tensor, aoi_names: torch.Tensor):
    """Convert one padded sequence into a graph (x, edge_index)."""
    device = seq.device

    valid_len = int(mask.sum().item())
    if valid_len <= 0:
        x = torch.zeros((1, seq.size(1)), dtype=seq.dtype, device=device)
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        return x, edge_index

    x = seq[:valid_len]                      # [N, F] on device
    aoi_valid = aoi_names[:valid_len].to(torch.long).to(device)

    ei_temporal = build_temporal_edges(valid_len, device=device)
    ei_aoi = build_aoi_edges(aoi_valid)

    edge_index = torch.cat([ei_temporal, ei_aoi], dim=1)
    edge_index = edge_index.t().unique(dim=0).t()

    return x, edge_index


# ===========================================================================
# Padding & PyG dataset
# ===========================================================================

def pad_to_mask_and_seq(sequences, max_len=32):
    """
    sequences: list of np.ndarray (L_i, D)
    Returns:
        seqs:  [B, max_len, D]
        masks: [B, max_len]
    """
    seqs, masks = [], []
    for seq in sequences:
        L, D = seq.shape
        Lc = min(L, max_len)
        padded = np.zeros((max_len, D), dtype=np.float32)
        padded[:Lc] = seq[:Lc]
        mask = np.zeros((max_len,), dtype=np.float32)
        mask[:Lc] = 1.0
        seqs.append(padded)
        masks.append(mask)
    return np.array(seqs), np.array(masks)


def build_pyg_dataset(seqs, masks, labels, aois):
    """
    seqs: [B, L, F] (padded)
    masks: [B, L]
    labels: [B]
    aois: list of np.array(L_i,) — integer AOI sequence per sample
    """
    if not HAS_PYG:
        raise RuntimeError("torch_geometric not installed; cannot build PyG dataset")

    data_list = []
    for i, (seq, mask, y, aoi_arr) in enumerate(zip(seqs, masks, labels, aois)):
        seq_t = torch.tensor(seq, dtype=torch.float32)   # [L, F]
        mask_t = torch.tensor(mask, dtype=torch.float32)  # [L]

        # aoi_arr is np.array(int), length >= valid length
        aoi_t = torch.tensor(aoi_arr, dtype=torch.long)

        x, edge_index = seq_mask_to_graph(seq_t, mask_t, aoi_t)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(int(y), dtype=torch.long),
        )

        # keep fixed-length sequence and mask for the CNN branch
        data.seq = seq_t            # [L, F]
        data.mask = mask_t          # [L]

        # variable-length info derived from the mask
        Li = int(mask_t.sum().item())
        data.raw_seq = seq_t[:Li]   # [Li, F]
        data.raw_len = torch.tensor([Li], dtype=torch.long)
        data.idx = torch.tensor([i], dtype=torch.long)

        data_list.append(data)

    return data_list


# ===========================================================================
# Losses & helpers
# ===========================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y):
        p = logits.softmax(dim=-1)
        pt = p[torch.arange(y.size(0)), y]
        alpha_t = torch.where(y == 1, self.alpha, 1 - self.alpha).to(logits.device)
        loss = -alpha_t * (1 - pt).pow(self.gamma) * pt.clamp_min(1e-8).log()
        return loss.mean()


def model_forward(model, x, edge_index, batch, seq, mask):
    """AOI_Net forward pass. Returns (logits, emb, aux_loss)."""
    logits, emb, _pi, aux = model(x, edge_index, batch=batch, seq=seq, mask=mask)
    return logits, emb, aux


def compute_loss(criterion, metric_loss_fn, metric_miner, logits, emb, aux, y, eta):
    """Combine cross-entropy, metric loss and MoE aux loss with weight eta."""
    if metric_miner is not None:
        try:
            mined = metric_miner(emb, y)
        except Exception:
            mined = None
    else:
        mined = None
    metric_loss = metric_loss_fn(emb, y, mined)
    aux_loss = aux if aux is not None else 0.0
    return eta * criterion(logits, y) + (1.0 - eta) * metric_loss + aux_loss


def eval_metrics(logits, y_true_tensor, num_classes):
    """Return dict of metrics from CPU logits/labels."""
    probs = logits.softmax(dim=-1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()
    y = y_true_tensor.numpy()
    acc = float(accuracy_score(y, preds))
    sen = float(recall_score(y, preds, pos_label=1))
    spe = float(recall_score(y, preds, pos_label=0))
    f1 = float(f1_score(y, preds))
    auc = float(roc_auc_score(y, probs)) if num_classes == 2 else float("nan")
    return {"acc": acc, "sen": sen, "spe": spe, "f1": f1, "auc": auc}


def _safe_float(x):
    x = float(x)
    return x if x == x else None  # convert NaN to None for JSON


# ===========================================================================
# Model construction
# ===========================================================================

def build_model(args, device):
    return AOI_Net(
        num_features=args.num_features,
        num_classes=args.num_classes,
        gnn_dim=128,
        dropout=0.5,
        use_pyg=args.use_pyg,
        top_k=args.top_k,
        router_hidden=64,
        aux_load_balance=args.aux_load_balance,
        temperature=1.0,
    ).to(device)


# ===========================================================================
# Fold training
# ===========================================================================

def _evaluate_pyg_loader(model, loader, device, num_classes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            B = getattr(batch, "num_graphs", None) or batch.y.size(0)
            F = batch.seq.size(-1)
            L = int(batch.seq.size(0) // B)
            seq_b = batch.seq.view(B, L, F)
            mask_b = batch.mask.view(B, L)
            logits, _emb, _aux = model_forward(
                model, batch.x, batch.edge_index, batch.batch, seq_b, mask_b
            )
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
    return eval_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), num_classes)


def _train_fold_pyg(model, criterion, optimizer, metric_loss_fn, metric_miner,
                    train_seqs, train_masks, y_train, aoi_train,
                    val_seqs, val_masks, y_val, aoi_val, args, device, fold):
    """Train one fold with PyG graph batching and early stopping on val acc."""
    train_data = build_pyg_dataset(train_seqs, train_masks, y_train, aoi_train)
    val_data = build_pyg_dataset(val_seqs, val_masks, y_val, aoi_val)
    train_loader = GeoDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    best = None
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            B = getattr(batch, "num_graphs", None) or batch.y.size(0)
            F = batch.seq.size(-1)
            L = int(batch.seq.size(0) // B)
            seq_b = batch.seq.view(B, L, F)
            mask_b = batch.mask.view(B, L)

            logits, emb, aux = model_forward(
                model, batch.x, batch.edge_index, batch.batch, seq_b, mask_b
            )
            loss = compute_loss(criterion, metric_loss_fn, metric_miner,
                                logits, emb, aux, batch.y, args.eta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        metrics = _evaluate_pyg_loader(model, val_loader, device, args.num_classes)
        print(
            f"[Fold {fold} | Epoch {epoch:02d}] loss={loss.item():.4f} "
            f"acc={metrics['acc']:.4f} sen={metrics['sen']:.4f} spe={metrics['spe']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
        )

        if best is None or metrics["acc"] > best["acc"]:
            best = dict(metrics)
            best["epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"[Fold {fold} | Epoch {epoch:02d}] early stop: "
                    f"val acc did not improve for {args.patience} epochs"
                )
                break

    return best


def _train_fold_single(model, criterion, optimizer, metric_loss_fn, metric_miner,
                       train_seqs, train_masks, y_train, aoi_train,
                       val_seqs, val_masks, y_val, aoi_val, args, device, fold):
    """Train one fold without PyG, processing one graph at a time."""
    train_tensors = [
        (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            int(y),
        )
        for s, m, a, y in zip(train_seqs, train_masks, aoi_train, y_train)
    ]
    val_tensors = [
        (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            int(y),
        )
        for s, m, a, y in zip(val_seqs, val_masks, aoi_val, y_val)
    ]

    best = None
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ce_losses, emb_list, y_list, aux_list = [], [], [], []
        for seq, mask, aoi, y in train_tensors:
            seq = seq.to(device)
            mask = mask.to(device)
            aoi = aoi.to(device)

            x_nodes, edge_index = seq_mask_to_graph(seq, mask, aoi)
            logits, emb, aux = model_forward(
                model, x_nodes.to(device), edge_index.to(device),
                batch=None, seq=seq.unsqueeze(0), mask=mask.unsqueeze(0),
            )
            ce_losses.append(criterion(logits, torch.tensor([y], dtype=torch.long, device=device)))
            emb_list.append(emb)
            y_list.append(y)
            if isinstance(aux, torch.Tensor):
                aux_list.append(aux)

        emb_all = torch.cat(emb_list, dim=0)
        y_all = torch.tensor(y_list, device=device)
        try:
            mined = metric_miner(emb_all, y_all)
        except Exception:
            mined = None
        metric_loss = metric_loss_fn(emb_all, y_all, mined)
        ce_loss = torch.stack(ce_losses).mean()
        aux_loss = torch.stack(aux_list).mean() if aux_list else torch.tensor(0.0, device=device)
        loss = args.eta * ce_loss + (1.0 - args.eta) * metric_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for seq, mask, aoi, y in val_tensors:
                seq = seq.to(device)
                mask = mask.to(device)
                aoi = aoi.to(device)
                x_nodes, edge_index = seq_mask_to_graph(seq, mask, aoi)
                logits, _emb, _aux = model_forward(
                    model, x_nodes.to(device), edge_index.to(device),
                    batch=None, seq=seq.unsqueeze(0), mask=mask.unsqueeze(0),
                )
                all_logits.append(logits.cpu())
                all_labels.append(torch.tensor([y]))
        metrics = eval_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), args.num_classes)
        print(
            f"[Fold {fold} | Epoch {epoch:02d}] loss={loss.item():.4f} "
            f"acc={metrics['acc']:.4f} sen={metrics['sen']:.4f} spe={metrics['spe']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
        )

        if best is None or metrics["acc"] > best["acc"]:
            best = dict(metrics)
            best["epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"[Fold {fold} | Epoch {epoch:02d}] early stop: "
                    f"val acc did not improve for {args.patience} epochs"
                )
                break

    return best


def train_fold(sequences, labels, aois, train_idx, val_idx, args, device, fold):
    """Slice the dataset for one CV fold and train a model on it."""
    X_train = [sequences[i] for i in train_idx]
    y_train = labels[train_idx]
    aoi_train = [aois[i] for i in train_idx]
    X_val = [sequences[i] for i in val_idx]
    y_val = labels[val_idx]
    aoi_val = [aois[i] for i in val_idx]

    train_seqs, train_masks = pad_to_mask_and_seq(X_train, max_len=args.max_seq_length)
    val_seqs, val_masks = pad_to_mask_and_seq(X_val, max_len=args.max_seq_length)

    # class weights computed from the training fold only
    classes, counts = np.unique(y_train, return_counts=True)
    w = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
    w = w / w.sum() * len(classes)
    print(f"[INFO] train fold class counts: {dict(zip(classes, counts))}, class_weight={w.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    metric_loss_fn = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
    metric_miner = miners.MultiSimilarityMiner(epsilon=0.05)

    if HAS_PYG and args.use_pyg:
        return _train_fold_pyg(
            model, criterion, optimizer, metric_loss_fn, metric_miner,
            train_seqs, train_masks, y_train, aoi_train,
            val_seqs, val_masks, y_val, aoi_val, args, device, fold,
        )
    return _train_fold_single(
        model, criterion, optimizer, metric_loss_fn, metric_miner,
        train_seqs, train_masks, y_train, aoi_train,
        val_seqs, val_masks, y_val, aoi_val, args, device, fold,
    )


# ===========================================================================
# k-fold cross-validation
# ===========================================================================

def summarize(metrics_list):
    """Mean / std over a list of metric dicts (NaN-safe)."""
    keys = ["acc", "sen", "spe", "f1", "auc"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not (m[k] != m[k])]
        summary[f"{k}_mean"] = _safe_float(np.mean(vals)) if vals else None
        summary[f"{k}_std"] = _safe_float(np.std(vals)) if vals else None
    return summary


def run_kfold_cv(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    sequences, labels, aois, scaler, classes = load_paradigms(
        args.data_dir, max_seq_length=args.max_seq_length
    )
    labels = np.asarray(labels)
    print(f"[INFO] samples: {len(sequences)}, classes: {args.num_classes}")

    # StratifiedKFold requires each class to have at least n_splits members.
    counts = np.bincount(labels, minlength=args.num_classes)
    n_folds = min(args.folds, int(counts.min()))
    if n_folds < args.folds:
        print(f"[WARN] reducing folds from {args.folds} to {n_folds} (class size limit)")
    if n_folds < 2:
        raise RuntimeError("Not enough samples per class to run cross-validation.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    all_fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\n=== Fold {fold + 1}/{n_folds} (seed={args.seed}) ===")
        result = train_fold(sequences, labels, aois, train_idx, val_idx, args, device, fold)
        result["fold"] = fold
        # convert float NaN to None so the results JSON stays valid
        result = {k: (_safe_float(v) if isinstance(v, float) else v) for k, v in result.items()}
        all_fold_metrics.append(result)
        print(
            f"[FOLD RESULT] acc={result['acc']:.4f} f1={result['f1']:.4f} "
            f"sen={result['sen']:.4f} spe={result['spe']:.4f} auc={result['auc']:.4f}"
        )

    summary = summarize(all_fold_metrics)
    return {
        "args": vars(args),
        "folds": all_fold_metrics,
        "summary": summary,
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AOI_Net: k-fold cross-validated training of the AOI-graph temporal+structural fusion model."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing .xlsx scanpath files (dataset is not bundled).")
    parser.add_argument("--folds", type=int, default=5, help="Number of k-fold CV folds.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")

    # model / training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-features", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=32)

    # loss
    parser.add_argument("--eta", type=float, default=0.25,
                        help="Weight of cross-entropy vs metric loss (0..1).")
    parser.add_argument("--top-k", type=int, default=0,
                        help="MoE routing top-k (0 = dense softmax).")
    parser.add_argument("--aux-load-balance", type=float, default=0.01,
                        help="MoE load-balance regularization strength.")

    # runtime
    parser.add_argument("--no-pyg", action="store_true",
                        help="Disable PyG graph batching (per-sample training).")
    parser.add_argument("--device", default=None, help="Device override (e.g. 'cuda:0', 'cpu').")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping: stop when val acc does not improve for this many epochs (0 = off).")
    parser.add_argument("--output", default="results.json",
                        help="Path to write the results JSON.")

    args = parser.parse_args()
    args.use_pyg = not args.no_pyg

    seed_everything(args.seed)

    result = run_kfold_cv(args)

    s = result["summary"]
    print("\n========== CV RESULT ==========")
    print(f"ACC: {s['acc_mean']:.4f} +/- {s['acc_std']:.4f}")
    print(f"F1 : {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}")
    print(f"SEN: {s['sen_mean']:.4f} +/- {s['sen_std']:.4f}")
    print(f"SPE: {s['spe_mean']:.4f} +/- {s['spe_std']:.4f}")
    print("===============================")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[INFO] results written to {args.output}")


if __name__ == "__main__":
    main()
