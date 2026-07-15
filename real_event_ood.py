import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import process_csv_to_tensors
from metrics import PhysicsLossCheck
from model import (
    AGCRN_Baseline,
    GraphWaveNet_Baseline,
    LSTMBaseline,
    MTGNN_Baseline,
    PI_X_STGAT,
    STGCN_Baseline,
)


class HistoricalAverage(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2].mean(dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def unpack_prediction(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def calc_jam_mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    return torch.abs(pred_jam - true_jam).mean().item()


def calc_jam_rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    return torch.sqrt(torch.mean((pred_jam - true_jam) ** 2)).item()


def build_time_metadata(csv_path: str, seq_len: int, horizon: int) -> tuple[pd.DataFrame, int]:
    weather_cols = [
        "Description_Rainy",
        "Description_Drizzle",
        "Description_Foggy",
        "Description_Windy",
    ]
    usecols = ["Date_Time", "Holiday", "Jam_Factor", *weather_cols]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"])
    time_meta = (
        df.groupby("Date_Time")
        .agg(
            Holiday=("Holiday", "max"),
            Rainy=("Description_Rainy", "max"),
            Drizzle=("Description_Drizzle", "max"),
            Foggy=("Description_Foggy", "max"),
            Windy=("Description_Windy", "max"),
            Mean_Jam=("Jam_Factor", "mean"),
        )
        .reset_index()
        .sort_values("Date_Time")
        .reset_index(drop=True)
    )
    num_samples = len(time_meta) - seq_len - horizon + 1
    train_size = int(num_samples * 0.8)
    time_meta["Sample_Index"] = np.arange(len(time_meta)) - seq_len - horizon + 1
    time_meta["Is_Test_Target"] = time_meta["Sample_Index"] >= train_size
    return time_meta, train_size


def summarize_real_events(time_meta: pd.DataFrame, train_size: int) -> pd.DataFrame:
    target_meta = time_meta[time_meta["Sample_Index"] >= train_size].copy()
    target_meta["Rain_or_Drizzle"] = (target_meta["Rainy"] > 0) | (target_meta["Drizzle"] > 0)
    target_meta["Date"] = target_meta["Date_Time"].dt.date
    rows = []
    for name, mask in {
        "Holiday": target_meta["Holiday"] > 0,
        "Rain/Drizzle": target_meta["Rain_or_Drizzle"],
    }.items():
        subset = target_meta[mask]
        if subset.empty:
            continue
        daily = (
            subset.groupby("Date")
            .agg(
                Start=("Date_Time", "min"),
                End=("Date_Time", "max"),
                Target_Samples=("Date_Time", "size"),
                Mean_Jam=("Mean_Jam", "mean"),
            )
            .reset_index()
        )
        daily["Scenario"] = name
        rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["Scenario", "Date", "Start", "End", "Target_Samples", "Mean_Jam"])
    return pd.concat(rows, ignore_index=True).sort_values(["Scenario", "Date"])


def select_scenario_indices(time_meta: pd.DataFrame, train_size: int, scenario: str) -> tuple[list[int], str]:
    target_meta = time_meta[time_meta["Sample_Index"] >= train_size].copy()
    target_meta["Rain_or_Drizzle"] = (target_meta["Rainy"] > 0) | (target_meta["Drizzle"] > 0)
    target_meta["Date"] = target_meta["Date_Time"].dt.date

    if scenario == "holiday":
        candidates = target_meta[target_meta["Holiday"] > 0]
        label = "Real Holiday OoD"
    elif scenario == "rain":
        candidates = target_meta[target_meta["Rain_or_Drizzle"]]
        label = "Real Rain/Drizzle OoD"
    else:
        raise ValueError("scenario must be holiday or rain")

    if candidates.empty:
        raise ValueError(f"No {scenario} target samples are present in the test split.")

    best_date = (
        candidates.groupby("Date")
        .agg(Target_Samples=("Date_Time", "size"), Mean_Jam=("Mean_Jam", "mean"))
        .sort_values(["Target_Samples", "Mean_Jam"], ascending=False)
        .index[0]
    )
    selected = candidates[candidates["Date"] == best_date].copy()
    sample_indices = selected["Sample_Index"].astype(int).tolist()
    label = f"{label}: {best_date}"
    return sample_indices, label


def build_model(model_name: str, num_nodes: int, adj_tensor: torch.Tensor, checkpoint: str | None = None) -> nn.Module:
    if model_name == "HA":
        return HistoricalAverage()
    if model_name == "LSTM":
        return LSTMBaseline(18, 64, 2)
    if model_name == "STGCN":
        return STGCN_Baseline(input_dim=18, hidden_dim=64, output_dim=2, adj_matrix=adj_tensor)
    if model_name == "AGCRN":
        return AGCRN_Baseline(num_nodes=num_nodes, input_dim=18, hidden_dim=128, output_dim=2, embed_dim=10)
    if model_name == "Graph WaveNet":
        return GraphWaveNet_Baseline(num_nodes, 18, 64, 2, adj_tensor, embed_dim=10)
    if model_name == "MTGNN":
        return MTGNN_Baseline(num_nodes, 18, 64, 2, embed_dim=10, top_k=6)
    if model_name == "PI-X-STGAT (Ours)":
        model = PI_X_STGAT(
            num_nodes=num_nodes,
            input_dim=18,
            gat_dim=64,
            gru_dim=128,
            output_dim=2,
            adj_matrix=adj_tensor,
            embed_dim=10,
            node_embed_scale=0.01,
        )
        if checkpoint and Path(checkpoint).exists():
            model.load_state_dict(torch.load(checkpoint, map_location=adj_tensor.device))
        return model
    raise ValueError(f"Unknown model: {model_name}")


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    device: torch.device,
    adj_tensor: torch.Tensor,
    epochs: int,
) -> None:
    if model_name in {"HA", "PI-X-STGAT (Ours)"}:
        return
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    total_mae = 0.0
    total_rmse = 0.0
    batches = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = unpack_prediction(model(x))
            total_mae += calc_jam_mae(pred, y)
            total_rmse += calc_jam_rmse(pred, y)
            batches += 1
    return {"Jam MAE": total_mae / batches, "Jam RMSE": total_rmse / batches}


def plot_results(results: pd.DataFrame, scenario_label: str) -> None:
    model_order = [
        "HA",
        "LSTM",
        "STGCN",
        "AGCRN",
        "Graph WaveNet",
        "MTGNN",
        "PI-X-STGAT (Ours)",
    ]
    colors = ["#B0BEC5", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad", "#0097a7", "#1565C0"]
    results["Model"] = pd.Categorical(results["Model"], categories=model_order, ordered=True)
    results = results.sort_values("Model")

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle(scenario_label, fontsize=17, fontweight="bold", y=1.02)

    for ax, metric, ylabel in [
        (axes[0], "Jam MAE", "Jam Factor MAE"),
        (axes[1], "Jam RMSE", "Jam Factor RMSE"),
    ]:
        bars = ax.bar(results["Model"], results[metric], color=colors, edgecolor="black", linewidth=1.1)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(axis="x", visible=False)
        for i, bar in enumerate(bars):
            value = bar.get_height()
            is_ours = "Ours" in str(results.iloc[i]["Model"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(results[metric].max() * 0.015, 0.005),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if is_ours else "normal",
                color="#B71C1C" if is_ours else "black",
            )

    plt.tight_layout()
    plt.savefig("Fig_Real_Event_OoD.png", dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real dataset OoD slices such as holidays or rain.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--scenario", choices=["holiday", "rain"], default="holiday")
    parser.add_argument("--checkpoint", default="pix_best_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running real-event OoD on {device}", flush=True)

    full_dataloader, adj_matrix, _ = process_csv_to_tensors(args.csv, seq_len=12, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]
    dataset = full_dataloader.dataset
    train_size = int(len(dataset) * 0.8)

    time_meta, meta_train_size = build_time_metadata(args.csv, seq_len=12, horizon=1)
    event_summary = summarize_real_events(time_meta, meta_train_size)
    event_summary.to_csv("real_ood_event_candidates.csv", index=False, encoding="utf-8-sig")
    print("Real OoD candidates in test split:")
    print(event_summary.to_string(index=False), flush=True)

    scenario_indices, scenario_label = select_scenario_indices(time_meta, meta_train_size, args.scenario)
    train_loader = DataLoader(Subset(dataset, range(0, train_size)), batch_size=args.batch_size, shuffle=True)
    scenario_loader = DataLoader(Subset(dataset, scenario_indices), batch_size=args.batch_size, shuffle=False)

    rows = []
    for model_name in ["HA", "LSTM", "STGCN", "AGCRN", "Graph WaveNet", "MTGNN", "PI-X-STGAT (Ours)"]:
        print(f"Evaluating {model_name}...", flush=True)
        model = build_model(model_name, num_nodes, adj_tensor, args.checkpoint).to(device)
        train_model(model, model_name, train_loader, device, adj_tensor, args.epochs)
        metrics = evaluate(model, scenario_loader, device)
        rows.append({"Scenario": scenario_label, "Model": model_name, "Target Samples": len(scenario_indices), **metrics})
        print(f"{model_name}: MAE={metrics['Jam MAE']:.4f}, RMSE={metrics['Jam RMSE']:.4f}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv("real_event_ood_results.csv", index=False, encoding="utf-8-sig")
    plot_results(results, scenario_label)
    print("Saved: real_event_ood_results.csv, real_ood_event_candidates.csv, Fig_Real_Event_OoD.png", flush=True)


if __name__ == "__main__":
    main()
