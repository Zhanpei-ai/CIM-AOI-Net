"""
AOI_Net training script: fixed 80% development / 20% test split.

The development participants undergo five-fold stratified validation. Each fold
fits its StandardScaler on training fixations only, saves the weights with the
highest validation ACC, and applies that scaler to validation and test data.
The five selected models' softmax probabilities are averaged on the fixed test
participants. ASD (result=2) is positive; the decision threshold is 0.5.

The split seed is fixed at 2026 for every dataset. Independent fold models use
2026 + the zero-based fold index. Architecture, losses and training defaults
follow the original repository (50 epochs, patience 8, batch size 32).

Run from the repository root (datasets and generated results are not bundled):
    python -m AOI_Net.train --data-dir /path/to/video1 --output /path/to/results/video1/results.json
    python -m AOI_Net.train --data-root /path/to/pro1 --output-dir /path/to/results
    python -m AOI_Net.train --data-dir /path/to/video1 --epochs 2 --device cpu --no-pyg

--data-root evaluates the eight datasets sequentially and reports internal
validation mean/std and ensemble test ACC/AUC/F1 after each dataset. The default
output directory is outside the Git checkout, keeping results separate.
dataloader.py handles raw loading, splits and fold-only standardization. Its
legacy global-scaling loaders remain available but are not used by this entry.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

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

# ---------------------------------------------------------------------------
# Imports
#   Run from the repository root (Code/):  python -m AOI_Net.train
#   dataloader.py lives in <code_root>/; model.py is in the AOI_Net package.
# ---------------------------------------------------------------------------
from AOI_Net.model import AOI_Net  # noqa: E402
from dataloader import (  # noqa: E402
    FEATURE_COLS, FIXED_SEED, LABEL_MAPPING, fit_preprocessing,
    load_raw_scanpaths, make_splits, prepare_scanpaths,
)


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


def seed_everything(seed=FIXED_SEED):
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
# PyG dataset
# ===========================================================================

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


# All datasets share FIXED_SEED from dataloader.py.
PARADIGMS = ("video1", "video2a", "video2b", "video4", "video5a", "video7", "video8a", "video9b")
_OUTPUT_PARENT = Path(__file__).resolve().parents[2]
if (_OUTPUT_PARENT / ".git").exists():
    _OUTPUT_PARENT = _OUTPUT_PARENT.parent
DEFAULT_OUTPUT_DIR = _OUTPUT_PARENT / "results_aoinet_holdout_cv5_2026"


# ===========================================================================
# Holdout test + internal cross-validation
# ===========================================================================

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def metrics_from_probabilities(y, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    y = np.asarray(y, dtype=int)
    if probabilities.shape != (len(y), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite Nx2 probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to one")
    predictions = (probabilities[:, 1] >= 0.5).astype(int)
    return {"acc": float(accuracy_score(y, predictions)),
            "auc": float(roc_auc_score(y, probabilities[:, 1])) if len(np.unique(y)) == 2 else None,
            "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0))}


def summarize(metrics_list):
    summary = {}
    for key in ("acc", "auc", "f1"):
        values = [item[key] for item in metrics_list if item[key] is not None]
        summary[key + "_mean"] = float(np.mean(values)) if values else None
        summary[key + "_std"] = float(np.std(values)) if values else None
    return summary


def prepare_data(scanpaths, indices, preprocessing, args):
    seqs, masks, labels, aois = prepare_scanpaths(
        scanpaths, indices, preprocessing, args.max_seq_length)
    if args.use_pyg:
        return build_pyg_dataset(seqs, masks, labels, aois)
    return [(torch.from_numpy(s), torch.from_numpy(m), torch.as_tensor(a, dtype=torch.long), int(y))
            for s, m, a, y in zip(seqs, masks, aois, labels)]


def pyg_forward(model, batch, device):
    batch = batch.to(device)
    count = batch.num_graphs
    seq = batch.seq.view(count, -1, batch.seq.size(-1))
    mask = batch.mask.view(count, -1)
    logits, emb, aux = model_forward(model, batch.x, batch.edge_index, batch.batch, seq, mask)
    return batch.y, logits, emb, aux


def single_forward(model, item, device):
    seq, mask, aoi, y = item
    seq, mask, aoi = seq.to(device), mask.to(device), aoi.to(device)
    nodes, edges = seq_mask_to_graph(seq, mask, aoi)
    logits, emb, aux = model_forward(model, nodes, edges, None, seq.unsqueeze(0), mask.unsqueeze(0))
    return torch.tensor([y], device=device), logits, emb, aux


def predict(model, data, args, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        if args.use_pyg:
            loader = GeoDataLoader(data, batch_size=args.batch_size, shuffle=False)
            iterator = (pyg_forward(model, batch, device) for batch in loader)
        else:
            iterator = (single_forward(model, item, device) for item in data)
        for labels, logits, _emb, _aux in iterator:
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    probabilities = torch.cat(logits_list).softmax(dim=-1).numpy()
    labels = torch.cat(labels_list).numpy()
    return probabilities, labels


def train_epoch(model, data, criterion, optimizer, metric_loss, miner, args, device):
    model.train()
    if args.use_pyg:
        loader = GeoDataLoader(data, batch_size=args.batch_size, shuffle=True)
        total = 0.0
        for batch in loader:
            labels, logits, embedding, aux = pyg_forward(model, batch, device)
            loss = compute_loss(criterion, metric_loss, miner, logits, embedding, aux, labels, args.eta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        return total / len(loader)
    # Keep per-graph forwards, but compute losses and update once per minibatch.
    # Weighted CE must see the entire batch: its mean reduction cancels the
    # weight if called separately on each single-labelled participant.
    order = np.random.permutation(len(data))
    total, batches = 0.0, 0
    for start in range(0, len(order), args.batch_size):
        logits_list, embeddings, labels_list, gates = [], [], [], []
        for index in order[start:start + args.batch_size]:
            seq, mask, aoi, label = data[index]
            seq, mask, aoi = seq.to(device), mask.to(device), aoi.to(device)
            nodes, edges = seq_mask_to_graph(seq, mask, aoi)
            logits, embedding, gate, _aux = model(
                nodes, edges, batch=None, seq=seq.unsqueeze(0), mask=mask.unsqueeze(0))
            logits_list.append(logits)
            embeddings.append(embedding)
            labels_list.append(label)
            gates.append(gate)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)
        logits, embedding = torch.cat(logits_list), torch.cat(embeddings)
        aux = None
        if args.aux_load_balance > 0:
            load = torch.cat(gates).mean(dim=0)
            uniform = torch.full_like(load, 1.0 / load.numel())
            aux = args.aux_load_balance * torch.nn.functional.kl_div(
                (load + 1e-9).log(), uniform, reduction="batchmean")
        loss = compute_loss(criterion, metric_loss, miner, logits, embedding, aux, labels, args.eta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / batches


def train_fold(scanpaths, train_indices, val_indices, args, device, fold, directory):
    seed_everything(args.seed + fold)
    preprocessing = fit_preprocessing(scanpaths.sequences, train_indices)
    train_data = prepare_data(scanpaths, train_indices, preprocessing, args)
    val_data = prepare_data(scanpaths, val_indices, preprocessing, args)
    counts = np.bincount(scanpaths.labels[train_indices], minlength=2)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    metric_loss = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
    miner = miners.MultiSimilarityMiner(epsilon=0.05)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best, best_state, history, stale = None, None, [], 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_data, criterion, optimizer, metric_loss, miner, args, device)
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at fold {fold}, epoch {epoch}")
        probabilities, y = predict(model, val_data, args, device)
        metrics = metrics_from_probabilities(y, probabilities)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        print(f"[fold {fold + 1}/{args.folds} epoch {epoch:02d}] loss={loss:.4f} "
              f"val ACC={metrics['acc']:.4f} AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f}", flush=True)
        if best is None or metrics["acc"] > best["acc"]:
            best = {"epoch": epoch, **metrics}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                break
    model.load_state_dict(best_state)
    probabilities, y = predict(model, val_data, args, device)
    replay = metrics_from_probabilities(y, probabilities)
    if any(abs(replay[k] - best[k]) > 1e-10 for k in ("acc", "auc", "f1")):
        raise RuntimeError("Restored best model does not reproduce validation metrics")
    directory.mkdir(parents=True, exist_ok=True)
    ids = scanpaths.participant_ids
    checkpoint = {"model_state_dict": best_state, "preprocessing": preprocessing,
                  "best_validation": best, "fold": fold, "seed": args.seed + fold,
                  "train_ids": [ids[i] for i in train_indices],
                  "validation_ids": [ids[i] for i in val_indices],
                  "label_mapping": LABEL_MAPPING, "model_args": vars(args)}
    torch.save(checkpoint, directory / "best.pt")
    np.savez_compressed(directory / "validation_predictions.npz",
                        participant_ids=np.asarray([ids[i] for i in val_indices]),
                        labels=y, probabilities=probabilities)
    write_json(directory / "training.json", {"best_validation": best, "history": history,
                                              "epochs_run": len(history), "restored_best_verified": True})
    return {"fold": fold, **best, "epochs_run": len(history)}


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "dataloader.py", Path(__file__),
             root / "AOI_Net" / "model.py"]
    if (root / "config.json").is_file():
        paths.append(root / "config.json")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_kfold_cv(args):
    """Run internal CV, then evaluate the untouched holdout by probability ensemble."""
    validate_args(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.use_pyg = not args.no_pyg and HAS_PYG
    torch.set_num_threads(args.threads)
    output = Path(args.output).resolve()
    directory = output.parent
    directory.mkdir(parents=True, exist_ok=True)
    scanpaths = load_raw_scanpaths(args.data_dir, args.max_seq_length)
    development, test, inner = make_splits(scanpaths.labels, args.folds, args.seed, args.test_size)
    ids = scanpaths.participant_ids
    protocol = {"args": vars(args), "device": device, "label_mapping": LABEL_MAPPING,
                "input_hashes": scanpaths.input_hashes, "source_hashes": code_hashes(),
                "participant_ids": ids, "labels": scanpaths.labels.tolist(),
                "development_ids": [ids[i] for i in development],
                "test_ids": [ids[i] for i in test], "threshold": 0.5,
                "ensemble": "equal mean of softmax probabilities from internal best models",
                "folds": [{"fold": f, "train_ids": [ids[i] for i in tr],
                           "validation_ids": [ids[i] for i in va]}
                          for f, (tr, va) in enumerate(inner)]}
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        if not args.resume:
            raise FileExistsError(f"Existing experiment at {directory}; use --resume for identical configuration")
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        for config in (protocol, previous):
            config["args"] = {key: value for key, value in config["args"].items() if key != "resume"}
        if previous != protocol:
            raise ValueError("Resume configuration, data, code, or split mismatch")
        if output.exists():
            result = json.loads(output.read_text(encoding="utf-8"))
            report_result(result)
            return result
    else:
        write_json(protocol_path, protocol)
    print(f"[INFO] {Path(args.data_dir).name}: subjects={len(ids)} "
          f"development={len(development)} test={len(test)} device={device} PyG={args.use_pyg}", flush=True)
    started = time.monotonic()
    fold_metrics = []
    for fold, (train, validation) in enumerate(inner):
        fold_dir = directory / f"fold{fold}"
        write_json(directory / "status.json", {"state": "training", "fold": fold,
                                               "completed_folds": len(fold_metrics)})
        if args.resume and (fold_dir / "training.json").exists() and (fold_dir / "best.pt").exists():
            record = json.loads((fold_dir / "training.json").read_text(encoding="utf-8"))
            result = {"fold": fold, **record["best_validation"], "epochs_run": record["epochs_run"]}
        else:
            result = train_fold(scanpaths, train, validation, args, device, fold, fold_dir)
        fold_metrics.append(result)
        write_json(directory / "internal_validation.json", {"folds": fold_metrics,
                                                            "summary": summarize(fold_metrics)})
    # Test data is prepared/predicted only after all internal models are selected.
    write_json(directory / "status.json", {"state": "testing", "completed_folds": args.folds})
    member_probabilities = []
    for fold in range(args.folds):
        checkpoint = torch.load(directory / f"fold{fold}" / "best.pt", map_location="cpu", weights_only=True)
        model = build_model(args, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_data = prepare_data(scanpaths, test, checkpoint["preprocessing"], args)
        probabilities, y = predict(model, test_data, args, device)
        if not np.array_equal(y, scanpaths.labels[test]):
            raise RuntimeError("Test prediction ordering mismatch")
        member_probabilities.append(probabilities)
        del model, test_data
    members = np.stack(member_probabilities)
    ensemble = members.mean(axis=0)
    test_metrics = metrics_from_probabilities(scanpaths.labels[test], ensemble)
    np.savez_compressed(directory / "test_predictions.npz", participant_ids=np.asarray([ids[i] for i in test]),
                        labels=scanpaths.labels[test], member_probabilities=members,
                        probabilities=ensemble, predictions=(ensemble[:, 1] >= 0.5).astype(int))
    result = {"dataset": Path(args.data_dir).name, "n_subjects": len(ids),
              "n_development": len(development), "n_test": len(test),
              "test": test_metrics, "internal_validation": {"folds": fold_metrics,
                                                           "summary": summarize(fold_metrics)},
              "threshold": 0.5, "positive_class": "ASD", "elapsed_seconds": time.monotonic() - started}
    write_json(output, result)
    write_json(directory / "status.json", {"state": "complete", "completed_folds": args.folds,
                                           "test": test_metrics})
    report_result(result)
    return result


def validate_args(args):
    if args.seed != FIXED_SEED:
        raise ValueError("The repository evaluation uses the fixed seed 2026")
    if args.num_classes != 2 or args.num_features != len(FEATURE_COLS):
        raise ValueError("Expected binary classification with the configured eight features")
    if min(args.epochs, args.batch_size, args.max_seq_length, args.threads) < 1 or args.patience < 0:
        raise ValueError("Invalid training budget, batch size, sequence length, threads, or patience")
    if args.lr <= 0 or not 0 <= args.eta <= 1:
        raise ValueError("lr must be positive and eta must be between 0 and 1")


def report_result(result):
    """Report internal validation separately from the ensemble holdout test."""
    summary = result["internal_validation"]["summary"]
    averages = " ".join(
        f"{key.upper()}={summary[key + '_mean']:.4f} +/- {summary[key + '_std']:.4f}"
        for key in ("acc", "auc", "f1"))
    print(f"[VALIDATION MEAN +/- STD] {result['dataset']} {averages}", flush=True)
    metrics = result["test"]
    print(f"[TEST ENSEMBLE] {result['dataset']} ACC={metrics['acc']:.4f} "
          f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f}", flush=True)


def save_batch_summary(directory, results):
    """Save four-column test results and a separate validation mean/std table."""
    write_json(directory / "summary.json", {"seed": FIXED_SEED, "results": results})
    with (directory / "test_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "ACC", "AUC", "F1"])
        for result in results:
            writer.writerow([result["dataset"], *[result["test"][key] for key in ("acc", "auc", "f1")]])
    with (directory / "internal_validation_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "ACC_mean", "ACC_std", "AUC_mean", "AUC_std", "F1_mean", "F1_std"])
        for result in results:
            summary = result["internal_validation"]["summary"]
            writer.writerow([result["dataset"], *[summary[key + "_" + stat]
                             for key in ("acc", "auc", "f1") for stat in ("mean", "std")]])


def run_eight_datasets(args):
    root = Path(args.data_root).resolve()
    directory = Path(args.output_dir).resolve()
    for name in PARADIGMS:
        if not list((root / name).glob("*.xlsx")):
            raise FileNotFoundError(f"Missing dataset: {root / name}")
    results = []
    for name in PARADIGMS:
        dataset_args = copy.copy(args)
        del dataset_args.data_root, dataset_args.output_dir
        dataset_args.data_dir = str(root / name)
        dataset_args.output = str(directory / name / "results.json")
        print(f"\n=== Dataset {name} ({len(results) + 1}/{len(PARADIGMS)}), seed={FIXED_SEED} ===", flush=True)
        results.append(run_kfold_cv(dataset_args))
        save_batch_summary(directory, results)
    print("\nDataset       ACC       AUC        F1", flush=True)
    for result in results:
        metrics = result["test"]
        print(f"{result['dataset']:<10} {metrics['acc']:.4f}    {metrics['auc']:.4f}    {metrics['f1']:.4f}", flush=True)
    return results


def add_training_arguments(parser):
    parser.add_argument("--folds", type=int, default=5, help="Internal development-data CV folds.")
    parser.add_argument("--seed", type=int, default=FIXED_SEED, choices=[FIXED_SEED], help="Fixed split seed: 2026.")
    parser.add_argument("--test-size", type=float, default=0.2)

    # model / training (original defaults)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-features", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=32)

    # loss
    parser.add_argument("--eta", type=float, default=0.25, help="Weight of cross-entropy vs metric loss (0..1).")
    parser.add_argument("--top-k", type=int, default=0, help="MoE routing top-k (0 = dense softmax).")
    parser.add_argument("--aux-load-balance", type=float, default=0.01)

    # runtime
    parser.add_argument("--no-pyg", action="store_true", help="Use per-graph forwards with minibatch loss/updates.")
    parser.add_argument("--device", default=None, help="Device override (e.g. 'cuda:0', 'cpu').")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (0 = off).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume only matching data/code/configuration.")


def main():
    parser = argparse.ArgumentParser(description="AOI_Net: 80/20 holdout with internal five-fold validation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", help="One dataset directory containing .xlsx scanpaths.")
    source.add_argument("--data-root", help="Parent directory containing the eight video datasets.")
    parser.add_argument("--output", help="Single-dataset results JSON path.")
    parser.add_argument("--output-dir", help="Eight-dataset results directory.")
    add_training_arguments(parser)
    args = parser.parse_args()
    if args.data_root:
        if args.output:
            parser.error("Use --output-dir with --data-root")
        args.output_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)
        run_eight_datasets(args)
    else:
        if args.output_dir:
            parser.error("Use --output with --data-dir")
        args.output = args.output or str(DEFAULT_OUTPUT_DIR / Path(args.data_dir).name / "results.json")
        del args.data_root, args.output_dir
        run_kfold_cv(args)


if __name__ == "__main__":
    main()
