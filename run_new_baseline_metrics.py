import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import process_csv_to_tensors
from experiment_paths import result_path
from model import GraphWaveNet_Baseline, MTGNN_Baseline


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_prediction(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def calc_jam_mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    return torch.abs(pred_jam - true_jam).mean()


def calc_jam_mape(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    denom = torch.clamp(torch.abs(true_jam), min=eps)
    return (torch.abs(pred_jam - true_jam) / denom).mean() * 100.0


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_flops(model: nn.Module, sample_x: torch.Tensor) -> int:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if sample_x.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        model.eval()
        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(sample_x)
        return int(sum(event.flops for event in prof.key_averages() if event.flops is not None))
    except Exception:
        return -1


def measure_inference_time(
    model: nn.Module,
    sample_x: torch.Tensor,
    repeats: int = 50,
    warmup: int = 10,
) -> tuple[float, float]:
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

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    batch_ms = elapsed_ms / repeats
    sample_ms = batch_ms / sample_x.size(0)
    return batch_ms, sample_ms


def train_one_model(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    model.train()

    for _ in range(epochs):
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            pred = unpack_prediction(model(x))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    total_mae = 0.0
    total_mape = 0.0
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = unpack_prediction(model(x))

            total_mae += calc_jam_mae(pred, y).item()
            total_mape += calc_jam_mape(pred, y).item()
            total_batches += 1

    return {
        "Jam MAE": total_mae / total_batches,
        "Jam MAPE (%)": total_mape / total_batches,
    }


def build_model(model_name: str, num_nodes: int, adj_tensor: torch.Tensor) -> nn.Module:
    if model_name == "Graph WaveNet":
        return GraphWaveNet_Baseline(
            num_nodes=num_nodes,
            input_dim=18,
            hidden_dim=64,
            output_dim=2,
            adj_matrix=adj_tensor,
            embed_dim=10,
        )

    if model_name == "MTGNN":
        return MTGNN_Baseline(
            num_nodes=num_nodes,
            input_dim=18,
            hidden_dim=64,
            output_dim=2,
            embed_dim=10,
            top_k=6,
        )

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graph WaveNet and MTGNN baseline metrics.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--output", default=result_path("new_baseline_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=["Graph WaveNet", "MTGNN"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon_label_map = {
        1: "15 min",
        2: "30 min",
        3: "45 min",
        4: "60 min",
    }
    model_names = args.models
    output_path = Path(args.output)
    output_columns = [
        "Horizon",
        "Model",
        "Jam MAE",
        "Jam MAPE (%)",
        "Params",
        "FLOPs",
        "Inference Time (ms/batch)",
        "Inference Time (ms/sample)",
    ]

    existing_pairs = set()
    if args.skip_existing and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_pairs = set(zip(existing_df["Horizon"], existing_df["Model"]))

    print(f"Running new baselines on {device}", flush=True)

    for horizon in args.horizons:
        horizon_label = horizon_label_map.get(horizon, f"{horizon} step")
        print(f"\nHorizon: {horizon_label}", flush=True)
        full_dataloader, adj_matrix, _ = process_csv_to_tensors(
            args.csv,
            seq_len=args.seq_len,
            horizon=horizon,
        )
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        num_nodes = adj_matrix.shape[0]

        full_dataset = full_dataloader.dataset
        train_size = int(len(full_dataset) * 0.8)
        train_end = train_size
        val_end = len(full_dataset)
        if args.max_train_samples > 0:
            train_end = min(train_end, args.max_train_samples)
        if args.max_val_samples > 0:
            val_end = min(val_end, train_size + args.max_val_samples)
        train_loader = DataLoader(
            Subset(full_dataset, range(0, train_end)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(full_dataset, range(train_size, val_end)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        sample_x = next(iter(val_loader))["x"].to(device)

        for model_name in model_names:
            if (horizon_label, model_name) in existing_pairs:
                print(f"Skipping existing row: {horizon_label} / {model_name}", flush=True)
                continue

            print(f"Training {model_name}...", flush=True)
            model = build_model(model_name, num_nodes, adj_tensor).to(device)
            params = count_params(model)

            train_one_model(model, train_loader, device, args.epochs)
            metrics = evaluate_model(model, val_loader, device)
            flops = estimate_flops(model, sample_x)
            infer_batch_ms, infer_sample_ms = measure_inference_time(model, sample_x)

            row = {
                "Horizon": horizon_label,
                "Model": model_name,
                **metrics,
                "Params": params,
                "FLOPs": flops,
                "Inference Time (ms/batch)": infer_batch_ms,
                "Inference Time (ms/sample)": infer_sample_ms,
            }
            row_df = pd.DataFrame([row], columns=output_columns)
            row_df.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"{model_name}: MAE={row['Jam MAE']:.4f}, "
                f"MAPE={row['Jam MAPE (%)']:.2f}%, Params={params:,}, "
                f"FLOPs={flops:,}, Time={infer_sample_ms:.4f} ms/sample"
                f" -> appended to {output_path}",
                flush=True,
            )

    print(f"\nDone. Metrics are in {output_path}", flush=True)


if __name__ == "__main__":
    main()
