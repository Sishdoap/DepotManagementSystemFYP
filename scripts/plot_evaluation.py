"""Generate report-ready plots and tables from results/results.csv.

Outputs to results/plots/:
    mean_wait_vs_rate.png        — main result, all 5 algorithms
    mean_wait_log_scale.png      — same data, log y-axis (round-robin fits)
    utilization_redistribution.png — strict vs unstrict per-gate utilization
    summary_table.md             — formatted markdown table for the report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULTS_CSV = Path("results/results.csv")
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Consistent ordering and colors across all plots.
ROUTER_ORDER = ["StrictFIFO", "ShortestQueue", "Random", "UnstrictFIFO", "RoundRobin"]
ROUTER_COLORS = {
    "StrictFIFO":    "#1f77b4",
    "ShortestQueue": "#2ca02c",
    "Random":        "#9467bd",
    "UnstrictFIFO":  "#d62728",
    "RoundRobin":    "#7f7f7f",
}


def load() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Run scripts/run_evaluation.py first to produce {RESULTS_CSV}"
        )
    return pd.read_csv(RESULTS_CSV)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (router, rate) mean wait + 95% CI."""
    grouped = df.groupby(["router_name", "arrival_rate"])["mean_wait"]
    rows = []
    for (router, rate), values in grouped:
        arr = values.to_numpy()
        # Bootstrap 95% CI.
        rng = np.random.default_rng(0)
        boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "router": router,
            "rate": rate,
            "mean": arr.mean(),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(arr),
        })
    return pd.DataFrame(rows)


def plot_mean_wait(df: pd.DataFrame, log: bool = False) -> None:
    """Line plot: mean wait vs arrival rate, error bars = 95% CI."""
    summary = summary_table(df)
    fig, ax = plt.subplots(figsize=(8, 5))

    for router in ROUTER_ORDER:
        sub = summary[summary["router"] == router].sort_values("rate")
        if sub.empty:
            continue
        ax.errorbar(
            sub["rate"],
            sub["mean"],
            yerr=[sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]],
            label=router,
            color=ROUTER_COLORS[router],
            marker="o",
            capsize=3,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Arrival rate (trucks per minute)")
    ax.set_ylabel("Mean wait time (seconds)")
    if log:
        ax.set_yscale("log")
        ax.set_title("Mean wait time by routing algorithm (log scale)")
        suffix = "log_scale"
    else:
        ax.set_title("Mean wait time by routing algorithm")
        suffix = "vs_rate"

    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"mean_wait_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / f'mean_wait_{suffix}.png'}")


def plot_utilization(df: pd.DataFrame) -> None:
    """Grouped bar chart: per-gate utilization for strict vs unstrict FIFO,
    averaged across all rates and seeds."""
    pair = df[df["router_name"].isin(["StrictFIFO", "UnstrictFIFO"])]
    grouped = pair.groupby("router_name")[["util_A", "util_B", "util_C"]].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    width = 0.35
    gates = ["Gate A (slow)", "Gate B (medium)", "Gate C (fast)"]

    ax.bar(
        x - width / 2,
        grouped.loc["StrictFIFO"].values,
        width,
        label="Strict FIFO",
        color=ROUTER_COLORS["StrictFIFO"],
    )
    ax.bar(
        x + width / 2,
        grouped.loc["UnstrictFIFO"].values,
        width,
        label="Unstrict FIFO",
        color=ROUTER_COLORS["UnstrictFIFO"],
    )

    ax.set_ylabel("Utilization (%)")
    ax.set_title("Per-gate utilization redistribution\n(averaged across rates 1.5–2.4/min, 30 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(gates)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate the deltas.
    for i, gate in enumerate(["util_A", "util_B", "util_C"]):
        strict = grouped.loc["StrictFIFO", gate]
        unstrict = grouped.loc["UnstrictFIFO", gate]
        delta = unstrict - strict
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta:.1f}pp",
            xy=(i, max(strict, unstrict) + 2),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "utilization_redistribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOTS_DIR / 'utilization_redistribution.png'}")


def write_summary_table(df: pd.DataFrame) -> None:
    """Markdown table for the report."""
    summary = summary_table(df)
    rates = sorted(summary["rate"].unique())

    lines = [
        "# Mean wait time results",
        "",
        "Mean wait time across 30 seeds per cell, with bootstrap 95% confidence intervals.",
        "All values in seconds.",
        "",
    ]

    # Header row.
    header = "| Arrival rate (per min) | " + " | ".join(ROUTER_ORDER) + " |"
    sep = "|---|" + "---|" * len(ROUTER_ORDER)
    lines.append(header)
    lines.append(sep)

    for rate in rates:
        row = [f"{rate:.1f}"]
        for router in ROUTER_ORDER:
            sub = summary[(summary["router"] == router) & (summary["rate"] == rate)]
            if sub.empty:
                row.append("—")
            else:
                m = sub.iloc[0]["mean"]
                lo = sub.iloc[0]["ci_lo"]
                hi = sub.iloc[0]["ci_hi"]
                row.append(f"{m:.1f} [{lo:.1f}, {hi:.1f}]")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("**Reading the table.** Each cell is `mean [95% CI low, high]` over 30 seeds.")
    lines.append("Non-overlapping CIs indicate the difference is unlikely to be due to chance.")

    out = PLOTS_DIR / "summary_table.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main():
    df = load()
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    plot_mean_wait(df, log=False)
    plot_mean_wait(df, log=True)
    plot_utilization(df)
    write_summary_table(df)

    print("\nAll outputs in:", PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()