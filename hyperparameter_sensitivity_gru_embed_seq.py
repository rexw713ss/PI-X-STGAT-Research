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
from model import PI_X_STGAT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def calc_jam_mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    return torch.abs(pred_jam - true_jam).mean().item()


def calc_jam_rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    return torch.sqrt(torch.mean((pred_jam - true_jam) ** 2)).item()


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    total_mae = 0.0
    total_rmse = 0.0
    batches = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred, _ = model(x)
            total_mae += calc_jam_mae(pred, y)
            total_rmse += calc_jam_rmse(pred, y)
            batches += 1
    return {
        "Jam MAE": total_mae / batches,
        "Jam RMSE": total_rmse / batches,
    }


def run_config(
    csv_path: str,
    device: torch.device,
    seed: int,
    seq_len: int,
    embed_dim: int,
    gru_dim: int,
    gat_dim: int,
    lambda_phy: float,
    node_embed_scale: float,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict[str, float]:
    set_seed(seed)
    full_dataloader, adj_matrix, _ = process_csv_to_tensors(csv_path, seq_len=seq_len, horizon=1)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]

    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(
        Subset(full_dataset, range(0, train_size)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, range(train_size, len(full_dataset))),
        batch_size=batch_size,
        shuffle=False,
    )

    model = PI_X_STGAT(
        num_nodes=num_nodes,
        input_dim=18,
        gat_dim=gat_dim,
        gru_dim=gru_dim,
        output_dim=2,
        adj_matrix=adj_tensor,
        embed_dim=embed_dim,
        node_embed_scale=node_embed_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor)

    best = {"Jam MAE": float("inf"), "Jam RMSE": float("inf"), "Best Epoch": 0}
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
        if metrics["Jam MAE"] < best["Jam MAE"]:
            best = {**metrics, "Best Epoch": epoch}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

        print(
            f"seq={seq_len} emb={embed_dim} gru={gru_dim} epoch={epoch:02d} "
            f"MAE={metrics['Jam MAE']:.4f} best={best['Jam MAE']:.4f}",
            flush=True,
        )

    return best


def append_row(output_path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def plot_results(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    panel_specs = [
        ("GRU Hidden Size", "GRU Hidden Size"),
        ("Embedding Dim", "Embedding Dimension d"),
        ("Sequence Length", "Sequence Length L"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.15)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=16, fontweight="bold")
    y_min = max(0.0, df["Jam MAE"].min() - 0.004)
    y_max = df["Jam MAE"].max() + 0.004

    for ax, (group_name, xlabel) in zip(axes, panel_specs):
        sub = df[df["Variable"] == group_name].sort_values("Value")
        values = sub["Value"].to_numpy()
        maes = sub["Jam MAE"].to_numpy()
        ax.plot(values, maes, marker="o", linewidth=2.8, markersize=7.5, color="#1565C0")
        best_idx = int(np.argmin(maes))
        ax.scatter(values[best_idx], maes[best_idx], s=95, color="#C62828", zorder=3)
        ax.set_title(group_name, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("Jam Factor MAE" if ax is axes[0] else "", fontsize=11, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(values)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.7)
        for idx, (_, row) in enumerate(sub.iterrows()):
            offset = 0.00075 if idx != best_idx else -0.00115
            va = "bottom" if idx != best_idx else "top"
            ax.text(
                row["Value"],
                row["Jam MAE"] + offset,
                f"{row['Jam MAE']:.4f}",
                ha="center",
                va=va,
                fontsize=8.7,
                fontweight="bold" if idx == best_idx else "normal",
                color="#C62828" if idx == best_idx else "#333333",
                clip_on=True,
            )

    plt.savefig("Fig_Hyperparameter_Sensitivity_GRU_Embed_Seq.png", dpi=300, bbox_inches="tight")
    plt.savefig("Fig_Hyperparameter_Sensitivity_GRU_Embed_Seq.pdf", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="PI-X-STGAT sensitivity for GRU size, embedding dim, and sequence length.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--output", default="hyperparameter_sensitivity_gru_embed_seq.csv")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-phy", type=float, default=0.005)
    parser.add_argument("--node-embed-scale", type=float, default=0.01)
    parser.add_argument("--gat-dim", type=int, default=64)
    parser.add_argument("--base-gru", type=int, default=128)
    parser.add_argument("--base-embed", type=int, default=10)
    parser.add_argument("--base-seq", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    existing = pd.read_csv(output_path) if args.skip_existing and output_path.exists() else pd.DataFrame()
    existing_keys = set()
    if not existing.empty:
        existing_keys = set(zip(existing["Variable"], existing["Value"]))

    experiments = []
    for value in [32, 64, 128, 256]:
        experiments.append(("GRU Hidden Size", value, args.base_seq, args.base_embed, value))
    for value in [4, 8, 10, 16]:
        experiments.append(("Embedding Dim", value, args.base_seq, value, args.base_gru))
    for value in [6, 12, 18, 24]:
        experiments.append(("Sequence Length", value, value, args.base_embed, args.base_gru))

    print(f"Running sensitivity on {device}", flush=True)
    for variable, value, seq_len, embed_dim, gru_dim in experiments:
        if (variable, value) in existing_keys:
            print(f"Skipping existing {variable}={value}", flush=True)
            continue

        print(f"\nExperiment: {variable}={value}", flush=True)
        metrics = run_config(
            csv_path=args.csv,
            device=device,
            seed=args.seed,
            seq_len=seq_len,
            embed_dim=embed_dim,
            gru_dim=gru_dim,
            gat_dim=args.gat_dim,
            lambda_phy=args.lambda_phy,
            node_embed_scale=args.node_embed_scale,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        row = {
            "Variable": variable,
            "Value": value,
            "Seq Len": seq_len,
            "Embedding Dim": embed_dim,
            "GRU Hidden Size": gru_dim,
            "Jam MAE": metrics["Jam MAE"],
            "Jam RMSE": metrics["Jam RMSE"],
            "Best Epoch": metrics["Best Epoch"],
        }
        append_row(output_path, row)
        print(f"Result: {row}", flush=True)

    plot_results(output_path)
    print(f"\nSaved data: {output_path}", flush=True)
    print("Saved figure: Fig_Hyperparameter_Sensitivity_GRU_Embed_Seq.png", flush=True)


if __name__ == "__main__":
    main()
