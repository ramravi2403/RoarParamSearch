import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

CSV_PATH = "combined_evaluation_runs/run_20260203_162511/combined_evaluation_results.csv"
OUT_DIR = '/'.join(CSV_PATH.split('/')[:-1] + ['plots'])


METRICS = ["extracted_accuracy", "extracted_auc", "cf_mean_distance"]
METHODS = ["roar", "optimal"]

def norm_sort_key(v: str):
    # numeric norms first (e.g., "1.0"), then "inf" last
    try:
        return (0, float(v))
    except Exception:
        return (1, v)


def read_results_from_csv(file_name):
    return pd.read_csv(file_name, header=0)


def clean_strings(df: pd.DataFrame):
    for c in ["model_type", "recourse_method", "norm"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def coerce_numerical_columns(df: pd.DataFrame):
    for c in ["delta_max", "lamb", "alpha", "extracted_accuracy", "extracted_auc", "cf_mean_distance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plot_metric_grid(df: pd.DataFrame, metric: str, fix_01_ylim: bool = True):
    model_types = sorted(df["model_type"].dropna().unique())
    norms = sorted(df["norm"].dropna().unique(), key=norm_sort_key)

    if not model_types or not norms:
        print(f"[skip] {metric}: no data.")
        return

    nrows, ncols = len(model_types), len(norms)
    fig_w = max(5 * ncols, 8)
    fig_h = max(3.5 * nrows, 4.5)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True)

    # normalize axes indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # If there are duplicate runs for same config, average them.
    group_cols = ["model_type", "norm", "recourse_method", "delta_max"]
    plot_df = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()

    for i, mt in enumerate(model_types):
        for j, nm in enumerate(norms):
            ax = axes[i, j]
            sub = plot_df[(plot_df["model_type"] == mt) & (plot_df["norm"] == nm)].copy()

            if sub.empty:
                ax.set_title(f"{mt} | norm={nm}\n(no data)")
                ax.grid(True, alpha=0.3)
                continue

            for method, mdf in sub.groupby("recourse_method"):
                mdf = mdf.sort_values("delta_max")
                ax.plot(mdf["delta_max"], mdf[metric], marker="o", label=method)

            ax.set_title(f"{mt} | norm={nm}")
            ax.grid(True, alpha=0.3)

            if metric in ("extracted_accuracy", "extracted_auc") and fix_01_ylim:
                ax.set_ylim(0.0, 1.0)

            if j == 0:
                ax.set_ylabel(metric)
            if i == nrows - 1:
                ax.set_xlabel("delta_max")

    # one legend for the whole figure
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='C0', marker='o', label='ROAR'),
        Line2D([0], [0], color='C1', marker='o', label='Optimal'),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12
    )

    fig.suptitle(f"{metric}: rows=model_type, cols=norm, x=delta_max", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_results_from_csv(CSV_PATH)
    df = clean_strings(df)
    df = coerce_numerical_columns(df)

    print("Rows:", len(df))
    print("model_type:", sorted(df["model_type"].unique()))
    print("norm:", sorted(df["norm"].unique(), key=norm_sort_key))
    print("methods:", sorted(df["recourse_method"].unique()))

    for metric in METRICS:
        if metric not in df.columns:
            print(f"[skip] missing metric column: {metric}")
            continue

        fig = plot_metric_grid(df, metric)
        if fig is None:
            continue

        out_path = os.path.join(OUT_DIR, f"{metric}_grid.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print("[saved]", out_path)

if __name__ == "__main__":
    main()

