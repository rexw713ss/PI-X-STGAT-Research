import argparse
import random
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset

from dataset import process_csv_to_tensors
from metrics import PhysicsLossCheck
from model import AGCRN_Baseline, LSTMBaseline, PI_X_STGAT, STGCN_Baseline


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


def train_and_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_name: str,
    adj_tensor: torch.Tensor,
    epochs: int,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor) if model_name == "PI-X-STGAT (Ours)" else None

    model.train()
    for epoch in range(1, epochs + 1):
        current_lambda = 0.015 if physics_loss is not None and epoch > 3 else 0.0
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

    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = unpack_prediction(model(x))
            total_mae += calc_jam_mae(pred, y)

    return total_mae / len(val_loader)


def build_model(model_name: str, num_nodes: int, adj_tensor: torch.Tensor) -> nn.Module:
    if model_name == "LSTM":
        return LSTMBaseline(18, 64, 2)
    if model_name == "STGCN":
        return STGCN_Baseline(input_dim=18, hidden_dim=64, output_dim=2, adj_matrix=adj_tensor)
    if model_name == "AGCRN":
        return AGCRN_Baseline(num_nodes=num_nodes, input_dim=18, hidden_dim=128, output_dim=2, embed_dim=10)
    if model_name == "PI-X-STGAT (Ours)":
        return PI_X_STGAT(
            num_nodes=num_nodes,
            input_dim=18,
            gat_dim=64,
            gru_dim=128,
            output_dim=2,
            adj_matrix=adj_tensor,
            embed_dim=10,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def append_row(output_path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def plot_multistep(results_path: Path) -> None:
    df = pd.read_csv(results_path)
    horizon_order = ["15 min", "30 min", "45 min", "60 min"]
    model_order = [
        "HA",
        "LSTM",
        "STGCN",
        "AGCRN",
        "Graph WaveNet",
        "MTGNN",
        "PI-X-STGAT (Ours)",
    ]
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values(["Model", "Horizon"])

    palette = {
        "HA": "#B0BEC5",
        "LSTM": "#e74c3c",
        "STGCN": "#27ae60",
        "AGCRN": "#f39c12",
        "Graph WaveNet": "#8e44ad",
        "MTGNN": "#0097a7",
        "PI-X-STGAT (Ours)": "#1565C0",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.4)
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=df,
        x="Horizon",
        y="Jam MAE",
        hue="Model",
        marker="o",
        linewidth=3,
        markersize=10,
        palette=palette,
    )
    plt.title("Multi-step Forecasting Performance Degradation", fontsize=18, fontweight="bold", pad=15)
    plt.ylabel("Jam Factor MAE", fontsize=14, fontweight="bold")
    plt.xlabel("Forecasting Horizon", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Model", title_fontsize="13", fontsize="12", loc="upper left")
    plt.tight_layout()
    plt.savefig("Fig_Multistep_Forecasting_Final.png", dpi=300, bbox_inches="tight")
    plt.savefig("Fig_Multistep_Forecasting.png", dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot multi-step forecasting comparison.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--output", default="multistep_forecasting_all_models.csv")
    parser.add_argument("--new-baseline-metrics", default="new_baseline_metrics.csv")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    existing_pairs = set()
    if args.skip_existing and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_pairs = set(zip(existing_df["Horizon"], existing_df["Model"]))

    horizons = [(1, "15 min"), (2, "30 min"), (3, "45 min"), (4, "60 min")]
    trained_models = ["LSTM", "STGCN", "AGCRN", "PI-X-STGAT (Ours)"]
    cached_new = pd.read_csv(args.new_baseline_metrics)

    print(f"Running multi-step comparison on {device}", flush=True)
    for horizon, horizon_label in horizons:
        print(f"\nHorizon: {horizon_label}", flush=True)
        full_dataloader, adj_matrix, _ = process_csv_to_tensors(args.csv, seq_len=args.seq_len, horizon=horizon)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        num_nodes = adj_matrix.shape[0]

        full_dataset = full_dataloader.dataset
        train_size = int(len(full_dataset) * 0.8)
        train_loader = DataLoader(
            Subset(full_dataset, range(0, train_size)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(full_dataset, range(train_size, len(full_dataset))),
            batch_size=args.batch_size,
            shuffle=False,
        )

        if (horizon_label, "HA") not in existing_pairs:
            ha_mae = 0.0
            for batch in val_loader:
                ha_pred = batch["x"][..., :2].mean(dim=1)
                ha_mae += calc_jam_mae(ha_pred, batch["y"])
            ha_mae /= len(val_loader)
            append_row(output_path, {"Horizon": horizon_label, "Model": "HA", "Jam MAE": ha_mae})
            print(f"HA: {ha_mae:.4f}", flush=True)

        for cached_model in ["Graph WaveNet", "MTGNN"]:
            if (horizon_label, cached_model) in existing_pairs:
                continue
            row = cached_new[(cached_new["Horizon"] == horizon_label) & (cached_new["Model"] == cached_model)].iloc[0]
            append_row(
                output_path,
                {"Horizon": horizon_label, "Model": cached_model, "Jam MAE": row["Jam MAE"]},
            )
            print(f"{cached_model}: {row['Jam MAE']:.4f} (cached)", flush=True)

        for model_name in trained_models:
            if (horizon_label, model_name) in existing_pairs:
                continue
            model = build_model(model_name, num_nodes, adj_tensor).to(device)
            mae = train_and_eval(model, train_loader, val_loader, device, model_name, adj_tensor, args.epochs)
            append_row(output_path, {"Horizon": horizon_label, "Model": model_name, "Jam MAE": mae})
            print(f"{model_name}: {mae:.4f}", flush=True)

    plot_multistep(output_path)
    print(f"\nSaved data: {output_path}", flush=True)
    print("Saved figures: Fig_Multistep_Forecasting.png, Fig_Multistep_Forecasting_Final.png", flush=True)


if __name__ == "__main__":
    main()
