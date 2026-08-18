import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from experiment_paths import figure_path, result_path


def main():
    metric_output = result_path("all_model_performance_for_plot.csv")
    df = pd.read_csv(metric_output)

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
    df["Sort"] = df["Model"].map({name: i for i, name in enumerate(model_order)})
    df = df.sort_values("Sort").drop(columns=["Sort"]).reset_index(drop=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    fig.suptitle("Overall Performance Comparison", fontsize=18, fontweight="bold", y=1.02)

    colors = [
        "#B0BEC5",
        "#90CAF9",
        "#64B5F6",
        "#42A5F5",
        "#1E88E5",
        "#8e44ad",
        "#0097a7",
        "#1565C0",
    ]
    edge_colors = [
        "#78909C",
        "#64B5F6",
        "#42A5F5",
        "#1E88E5",
        "#1565C0",
        "#6c3483",
        "#00796b",
        "#0D47A1",
    ]

    x_pos = np.arange(len(df))
    bar_width = 0.62

    bars_mae = axes[0].bar(
        x_pos,
        df["Jam MAE"],
        width=bar_width,
        color=colors[: len(df)],
        edgecolor=edge_colors[: len(df)],
        linewidth=1.5,
    )
    axes[0].set_title("Jam Factor Prediction: MAE", fontsize=13, fontweight="bold", pad=14)
    axes[0].set_ylabel("Mean Absolute Error", fontsize=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(df["Model"], fontsize=10, fontweight="bold", rotation=18, ha="right")
    axes[0].set_ylim(0, max(0.7, df["Jam MAE"].max() * 1.15))

    for i, bar in enumerate(bars_mae):
        yval = bar.get_height()
        is_ours = "Ours" in df.loc[i, "Model"]
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.012,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if is_ours else "normal",
            color="#B71C1C" if is_ours else "black",
        )

    rmse_df = df.dropna(subset=["Jam RMSE"]).reset_index(drop=True)
    rmse_x = np.arange(len(rmse_df))
    bars_rmse = axes[1].bar(
        rmse_x,
        rmse_df["Jam RMSE"],
        width=bar_width,
        color=colors[: len(rmse_df)],
        edgecolor=edge_colors[: len(rmse_df)],
        linewidth=1.5,
    )
    axes[1].set_title("Jam Factor Prediction: RMSE", fontsize=13, fontweight="bold", pad=14)
    axes[1].set_ylabel("Root Mean Square Error", fontsize=12)
    axes[1].set_xticks(rmse_x)
    axes[1].set_xticklabels(rmse_df["Model"], fontsize=10, fontweight="bold", rotation=18, ha="right")
    axes[1].set_ylim(0, max(2.2, rmse_df["Jam RMSE"].max() * 1.15))

    for i, bar in enumerate(bars_rmse):
        yval = bar.get_height()
        is_ours = "Ours" in rmse_df.loc[i, "Model"]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.025,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if is_ours else "normal",
            color="#B71C1C" if is_ours else "black",
        )

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(axis="x", visible=False)

    plt.tight_layout()

    output = figure_path("Fig_Performance_Comparison_All_Models.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output}")
    print(f"Saved plot data: {metric_output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
