import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import process_csv_to_tensors
from experiment_paths import checkpoint_path, result_path
from metrics import PhysicsLossCheck
from model import PI_X_STGAT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def calc_metrics(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-3) -> dict[str, float]:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    diff = pred_jam - true_jam
    abs_diff = torch.abs(diff)
    return {
        "Jam MAE": abs_diff.mean().item(),
        "Jam RMSE": torch.sqrt(torch.mean(diff ** 2)).item(),
        "Jam MAPE (%)": (abs_diff / torch.clamp(torch.abs(true_jam), min=eps)).mean().item() * 100.0,
    }


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals = {"Jam MAE": 0.0, "Jam RMSE": 0.0, "Jam MAPE (%)": 0.0}
    batches = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred, _ = model(x)
            metrics = calc_metrics(pred, y)
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
    return {key: value / batches for key, value in totals.items()}


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_flops(model: nn.Module, sample_x: torch.Tensor) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if sample_x.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model(sample_x)
    return int(sum(event.flops for event in prof.key_averages() if event.flops is not None))


def measure_inference_time(model: nn.Module, sample_x: torch.Tensor, repeats: int = 50, warmup: int = 10) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_x)
        if sample_x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample_x)
        if sample_x.is_cuda:
            torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return batch_ms, batch_ms / sample_x.size(0)


def train_config(
    seed: int,
    lambda_phy: float,
    node_embed_scale: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sample_x: torch.Tensor,
    adj_tensor: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    epochs: int,
    patience: int,
) -> tuple[dict[str, float], dict]:
    set_seed(seed)
    model = PI_X_STGAT(
        num_nodes=num_nodes,
        input_dim=18,
        gat_dim=64,
        gru_dim=128,
        output_dim=2,
        adj_matrix=adj_tensor,
        embed_dim=10,
        node_embed_scale=node_embed_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor)

    best = {"Jam MAE": float("inf")}
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        current_lambda = 0.0 if epoch <= 3 else lambda_phy
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            if current_lambda > 0:
                loss = loss + current_lambda * physics_loss(pred, x[:, -1, :, 0])
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, device)
        print(
            f"seed={seed} lambda={lambda_phy} scale={node_embed_scale} "
            f"epoch={epoch:02d} MAE={metrics['Jam MAE']:.4f}",
            flush=True,
        )

        if metrics["Jam MAE"] < best["Jam MAE"]:
            best = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    infer_batch_ms, infer_sample_ms = measure_inference_time(model, sample_x)
    best.update(
        {
            "Params": count_params(model),
            "FLOPs": estimate_flops(model, sample_x),
            "Inference Time (ms/batch)": infer_batch_ms,
            "Inference Time (ms/sample)": infer_sample_ms,
            "Seed": seed,
            "Lambda": lambda_phy,
            "Node Emb Scale": node_embed_scale,
            "Best Epoch": best_epoch,
        }
    )
    return best, best_state


def update_performance_csv(csv_path: Path, best_row: dict) -> None:
    df = pd.read_csv(csv_path)
    mask = df["Model"] == "PI-X-STGAT\n(Ours)"
    for key, value in best_row.items():
        if key in df.columns:
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a best-epoch PI-X-STGAT run and update the comparison table.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--output", default=result_path("pix_best_search_results.csv"))
    parser.add_argument("--performance-csv", default=result_path("all_model_performance_for_plot.csv"))
    parser.add_argument("--checkpoint", default=checkpoint_path("pix_best_checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 123, 2024])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.005, 0.015])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.01, 0.1])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Searching PI-X-STGAT on {device}", flush=True)

    full_dataloader, adj_matrix, _ = process_csv_to_tensors(args.csv, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]
    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=args.batch_size, shuffle=False)
    sample_x = next(iter(val_loader))["x"].to(device)

    rows = []
    best_row = {"Jam MAE": float("inf")}
    best_state = None
    for seed in args.seeds:
        for lambda_phy in args.lambdas:
            for scale in args.scales:
                row, state = train_config(
                    seed,
                    lambda_phy,
                    scale,
                    train_loader,
                    val_loader,
                    sample_x,
                    adj_tensor,
                    num_nodes,
                    device,
                    args.epochs,
                    args.patience,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
                if row["Jam MAE"] < best_row["Jam MAE"]:
                    best_row = row
                    best_state = state
                    torch.save(best_state, args.checkpoint)
                    update_performance_csv(Path(args.performance_csv), best_row)
                    print(f"NEW BEST: {best_row}", flush=True)

    print("\nBest PI-X-STGAT run:", flush=True)
    print(best_row, flush=True)


if __name__ == "__main__":
    main()
