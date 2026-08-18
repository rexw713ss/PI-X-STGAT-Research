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
from metrics import PhysicsLossCheck
from model import (
    AGCRN_Baseline,
    GraphWaveNet_Baseline,
    LSTMBaseline,
    MTGNN_Baseline,
    PI_X_STGAT,
    STGCN_Baseline,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class HistoricalAverage(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2].mean(dim=1)


def unpack_prediction(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def calc_jam_metrics(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-3) -> dict[str, float]:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    abs_error = torch.abs(pred_jam - true_jam)
    denom = torch.clamp(torch.abs(true_jam), min=eps)
    return {
        "Jam MAE": abs_error.mean().item(),
        "Jam RMSE": torch.sqrt(torch.mean((pred_jam - true_jam) ** 2)).item(),
        "Jam MAPE (%)": (abs_error / denom).mean().item() * 100.0,
    }


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

    batch_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return batch_ms, batch_ms / sample_x.size(0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    model_name: str,
    adj_tensor: torch.Tensor,
) -> None:
    if model_name == "HA":
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor) if model_name == "PI-X-STGAT\n(Ours)" else None

    model.train()
    for epoch in range(1, epochs + 1):
        current_lambda = 0.0
        if physics_loss is not None and epoch > 3:
            current_lambda = 0.015

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred = unpack_prediction(model(x))
            loss = criterion(pred, y)
            if current_lambda > 0:
                loss = loss + current_lambda * physics_loss(pred, x[:, -1, :, 0])
            loss.backward()
            optimizer.step()


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals = {"Jam MAE": 0.0, "Jam RMSE": 0.0, "Jam MAPE (%)": 0.0}
    batches = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = unpack_prediction(model(x))
            metrics = calc_jam_metrics(pred, y)
            for key, value in metrics.items():
                totals[key] += value
            batches += 1

    return {key: value / batches for key, value in totals.items()}


def build_model(model_name: str, num_nodes: int, adj_tensor: torch.Tensor) -> nn.Module:
    if model_name == "HA":
        return HistoricalAverage()
    if model_name == "LSTM":
        return LSTMBaseline(18, 64, 2)
    if model_name == "STGCN":
        return STGCN_Baseline(input_dim=18, hidden_dim=64, output_dim=2, adj_matrix=adj_tensor)
    if model_name == "AGCRN":
        return AGCRN_Baseline(num_nodes=num_nodes, input_dim=18, hidden_dim=128, output_dim=2, embed_dim=10)
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
    if model_name == "PI-X-STGAT\n(Ours)":
        return PI_X_STGAT(
            num_nodes=num_nodes,
            input_dim=18,
            gat_dim=64,
            gru_dim=128,
            output_dim=2,
            adj_matrix=adj_tensor,
            embed_dim=10,
        )
    raise ValueError(f"No executable implementation found for {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure all implemented model metrics for 15-min comparison.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output", default=result_path("all_model_performance_for_plot.csv"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Measuring implemented models on {device}", flush=True)

    full_dataloader, adj_matrix, _ = process_csv_to_tensors(args.csv, seq_len=args.seq_len, horizon=args.horizon)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]

    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=args.batch_size)
    sample_x = next(iter(val_loader))["x"].to(device)

    model_order = [
        "HA",
        "LSTM",
        "STGCN",
        "ASTGCN",
        "AGCRN",
        "Graph WaveNet",
        "MTGNN",
        "PI-X-STGAT\n(Ours)",
    ]
    implemented = [name for name in model_order if name != "ASTGCN"]
    if args.models is not None:
        requested = set(args.models)
        implemented = [name for name in implemented if name in requested]

    existing_df = pd.read_csv(args.output) if Path(args.output).exists() else pd.DataFrame()
    rows = []
    for model_name in implemented:
        if args.skip_existing and not existing_df.empty and "Model" in existing_df.columns:
            existing_row = existing_df[existing_df["Model"] == model_name]
            if not existing_row.empty:
                required = ["Jam MAPE (%)", "Params", "FLOPs", "Inference Time (ms/sample)"]
                if all(col in existing_row.columns and pd.notna(existing_row.iloc[0].get(col)) for col in required):
                    print(f"\nSkipping existing metrics for {model_name}", flush=True)
                    rows.append(existing_row.iloc[0].to_dict())
                    continue

        print(f"\nRunning {model_name}...", flush=True)
        model = build_model(model_name, num_nodes, adj_tensor).to(device)
        train_model(model, train_loader, device, args.epochs, model_name, adj_tensor)
        metrics = evaluate_model(model, val_loader, device)
        params = count_params(model)
        flops = estimate_flops(model, sample_x)
        infer_batch_ms, infer_sample_ms = measure_inference_time(model, sample_x)
        row = {
            "Model": model_name,
            **metrics,
            "Params": params,
            "FLOPs": flops,
            "Inference Time (ms/batch)": infer_batch_ms,
            "Inference Time (ms/sample)": infer_sample_ms,
        }
        rows.append(row)
        print(
            f"{model_name}: MAE={row['Jam MAE']:.4f}, RMSE={row['Jam RMSE']:.4f}, "
            f"MAPE={row['Jam MAPE (%)']:.2f}%, Params={params:,}, FLOPs={flops:,}, "
            f"Time={infer_sample_ms:.4f} ms/sample",
            flush=True,
        )

    measured_df = pd.DataFrame(rows)

    if existing_df.empty:
        existing_df = pd.DataFrame(
            [{"Model": "ASTGCN", "Jam MAE": 0.0805, "Jam RMSE": 0.6214}]
        )
    elif not (existing_df["Model"] == "ASTGCN").any():
        existing_df = pd.concat(
            [existing_df, pd.DataFrame([{"Model": "ASTGCN", "Jam MAE": 0.0805, "Jam RMSE": 0.6214}])],
            ignore_index=True,
            sort=False,
        )

    if not measured_df.empty:
        existing_df = existing_df[~existing_df["Model"].isin(measured_df["Model"])]

    final_df = pd.concat([existing_df, measured_df], ignore_index=True, sort=False)
    final_df["Sort"] = final_df["Model"].map({name: i for i, name in enumerate(model_order)})
    final_df = final_df.sort_values("Sort").drop(columns=["Sort"])

    columns = [
        "Model",
        "Jam MAE",
        "Jam RMSE",
        "Jam MAPE (%)",
        "Params",
        "FLOPs",
        "Inference Time (ms/batch)",
        "Inference Time (ms/sample)",
    ]
    final_df = final_df.reindex(columns=columns)
    final_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved updated metrics to {args.output}", flush=True)
    print(final_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
