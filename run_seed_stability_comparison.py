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
from model import MTGNN_Baseline, PI_X_STGAT


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


def calc_metrics(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-3) -> dict[str, float]:
    pred_jam = pred[..., 0] * 10.0
    true_jam = true[..., 0] * 10.0
    error = pred_jam - true_jam
    abs_error = torch.abs(error)
    denom = torch.clamp(torch.abs(true_jam), min=eps)
    return {
        "Jam MAE": abs_error.mean().item(),
        "Jam RMSE": torch.sqrt(torch.mean(error ** 2)).item(),
        "Jam MAPE (%)": (abs_error / denom).mean().item() * 100.0,
    }


def build_model(model_name: str, num_nodes: int, adj_tensor: torch.Tensor) -> nn.Module:
    if model_name == "PI-X-STGAT":
        return PI_X_STGAT(
            num_nodes=num_nodes,
            input_dim=18,
            gat_dim=64,
            gru_dim=128,
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
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals = {"Jam MAE": 0.0, "Jam RMSE": 0.0, "Jam MAPE (%)": 0.0}
    batches = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = unpack_prediction(model(x))
            metrics = calc_metrics(pred, y)
            for key, value in metrics.items():
                totals[key] += value
            batches += 1

    return {key: value / batches for key, value in totals.items()}


def train_best_epoch(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    adj_tensor: torch.Tensor,
    epochs: int,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = PhysicsLossCheck(adj_tensor) if model_name == "PI-X-STGAT" else None

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
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

        metrics = evaluate(model, val_loader, device)
        if best is None or metrics["Jam MAE"] < best["Jam MAE"]:
            best = {"Best Epoch": epoch, **metrics}

        print(
            f"    epoch {epoch:02d}: MAE={metrics['Jam MAE']:.4f}, "
            f"RMSE={metrics['Jam RMSE']:.4f}, MAPE={metrics['Jam MAPE (%)']:.2f}% "
            f"(best={best['Jam MAE']:.4f})",
            flush=True,
        )

    return best


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def plot_results(result_path: Path, summary_path: Path) -> None:
    df = pd.read_csv(result_path)
    summary = (
        df.groupby("Model", as_index=False)
        .agg(
            **{
                "Jam MAE Mean": ("Jam MAE", "mean"),
                "Jam MAE Std": ("Jam MAE", "std"),
                "Jam RMSE Mean": ("Jam RMSE", "mean"),
                "Jam RMSE Std": ("Jam RMSE", "std"),
                "Jam MAPE Mean": ("Jam MAPE (%)", "mean"),
                "Jam MAPE Std": ("Jam MAPE (%)", "std"),
            }
        )
        .sort_values("Jam MAE Mean")
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.25)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Seed Stability Comparison", fontsize=17, fontweight="bold", y=1.03)

    metrics = [
        ("Jam MAE", "Jam MAE Mean", "Jam MAE Std", "Jam MAE"),
        ("Jam RMSE", "Jam RMSE Mean", "Jam RMSE Std", "Jam RMSE"),
        ("Jam MAPE (%)", "Jam MAPE Mean", "Jam MAPE Std", "Jam MAPE (%)"),
    ]
    colors = {"PI-X-STGAT": "#1565C0", "MTGNN": "#0097a7"}

    for ax, (_, mean_col, std_col, title) in zip(axes, metrics):
        x = np.arange(len(summary))
        y = summary[mean_col].to_numpy()
        yerr = summary[std_col].fillna(0.0).to_numpy()
        bar_colors = [colors.get(model, "#607D8B") for model in summary["Model"]]
        bars = ax.bar(x, y, yerr=yerr, capsize=5, color=bar_colors, edgecolor="#263238", linewidth=1.2)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["Model"], fontsize=11, fontweight="bold", rotation=12, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(axis="x", visible=False)

        for bar, mean, std in zip(bars, y, yerr):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(y) * 0.025,
                f"{mean:.4f}\n±{std:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output = "Fig_Seed_Stability_Comparison.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed stability comparison for paper figures.")
    parser.add_argument("--csv", default="all_segments_preprocessed.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024, 3407])
    parser.add_argument("--models", nargs="+", default=["PI-X-STGAT", "MTGNN"])
    parser.add_argument("--output", default="seed_stability_results.csv")
    parser.add_argument("--summary-output", default="seed_stability_summary.csv")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    result_path = Path(args.output)
    summary_path = Path(args.summary_output)
    existing_pairs = set()
    if args.skip_existing and result_path.exists():
        existing_df = pd.read_csv(result_path)
        existing_pairs = set(zip(existing_df["Model"], existing_df["Seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data once, running on {device}", flush=True)
    full_dataloader, adj_matrix, _ = process_csv_to_tensors(args.csv, seq_len=args.seq_len, horizon=args.horizon)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    num_nodes = adj_matrix.shape[0]

    full_dataset = full_dataloader.dataset
    train_size = int(len(full_dataset) * 0.8)
    train_loader = DataLoader(Subset(full_dataset, range(0, train_size)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, range(train_size, len(full_dataset))), batch_size=args.batch_size)

    for seed in args.seeds:
        for model_name in args.models:
            if (model_name, seed) in existing_pairs:
                print(f"Skipping existing {model_name} seed={seed}", flush=True)
                continue

            print(f"\nTraining {model_name} seed={seed}", flush=True)
            set_seed(seed)
            model = build_model(model_name, num_nodes, adj_tensor).to(device)
            best = train_best_epoch(model, model_name, train_loader, val_loader, device, adj_tensor, args.epochs)
            row = {"Model": model_name, "Seed": seed, **best}
            append_row(result_path, row)
            print(
                f"  best {model_name} seed={seed}: epoch={best['Best Epoch']}, "
                f"MAE={best['Jam MAE']:.4f}",
                flush=True,
            )

    plot_results(result_path, summary_path)


if __name__ == "__main__":
    main()
