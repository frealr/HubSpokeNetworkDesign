#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


BASE_DIR = Path(__file__).resolve().parent
MAT_DIR = BASE_DIR / "hs_prueba_v0_blo"
OUT_DIR = BASE_DIR / "plots_blo"

MAT_RE = re.compile(
    r"^bud=(?P<budget>[^_]+)"
    r"_lam=(?P<lam>[^_]+)"
    r"_alfa=(?P<alfa>[^_]+)"
    r"_mu_al=(?P<mu_alfa>[^_]+)"
    r"_mu_bet=(?P<mu_beta>[^_]+)"
    r"_python(?P<longrun>_longrun)?\.mat$"
)


def parse_name(path: Path) -> dict[str, float | str | bool] | None:
    match = MAT_RE.match(path.name)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "file": path.name,
        "budget": float(groups["budget"]),
        "lam": float(groups["lam"]),
        "alfa": float(groups["alfa"]),
        "mu_alfa": float(groups["mu_alfa"]),
        "mu_beta": float(groups["mu_beta"]),
        "is_longrun": bool(groups["longrun"]),
    }


def load_scalar(data: dict, key: str) -> float | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        return None
    return float(value.squeeze())


def combo_label(row: pd.Series) -> str:
    if row["mu_alfa"] == 0.0 and row["mu_beta"] == 0.0:
        return "mu_al=0e+00, mu_b=0e+00"
    base = f"mu_al={row['mu_alfa']:.0e}, mu_b={row['mu_beta']:.0e}"
    if row["is_longrun"]:
        return f"{base} bliters=10"
    return f"{base} bliters=3"


def series_label(row: pd.Series) -> str:
    if row["mu_alfa"] == 0.0 and row["mu_beta"] == 0.0:
        return "mu_al=0e+00 | mu_b=0e+00"
    base = f"mu_al={row['mu_alfa']:.0e} | mu_b={row['mu_beta']:.0e}"
    if row["is_longrun"]:
        return f"{base} | bliters=10"
    return f"{base} | bliters=3"


def load_results() -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for path in sorted(MAT_DIR.glob("*.mat")):
        parsed = parse_name(path)
        if parsed is None:
            continue
        data = sio.loadmat(path)
        parsed["obj_val"] = load_scalar(data, "obj_val")
        parsed["pax_obj"] = load_scalar(data, "pax_obj")
        parsed["op_obj"] = load_scalar(data, "op_obj")
        parsed["used_budget"] = load_scalar(data, "used_budget")
        parsed["comp_time"] = load_scalar(data, "comp_time")
        rows.append(parsed)

    if not rows:
        raise FileNotFoundError(f"No MAT files found in {MAT_DIR}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["obj_val"]).copy()
    df["combo"] = df.apply(combo_label, axis=1)
    df["series"] = df.apply(series_label, axis=1)
    df["objective_gain"] = -df["obj_val"]

    df["rank_in_budget_lam"] = (
        df.groupby(["lam", "budget"])["obj_val"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    best_by_group = df.groupby(["lam", "budget"])["obj_val"].transform("min")
    df["gap_vs_best_pct"] = 100.0 * (df["obj_val"] - best_by_group) / best_by_group.abs().clip(lower=1e-9)

    regular = df[~df["is_longrun"]].copy()
    regular_best = (
        regular.sort_values(["lam", "budget", "obj_val"])
        .groupby(["lam", "budget"], as_index=False)
        .first()[["lam", "budget", "obj_val", "series", "combo"]]
        .rename(
            columns={
                "obj_val": "best_regular_obj_val",
                "series": "best_regular_series",
                "combo": "best_regular_combo",
            }
        )
    )
    df = df.merge(regular_best, on=["lam", "budget"], how="left")
    df["delta_vs_best_regular"] = df["best_regular_obj_val"] - df["obj_val"]
    df["improvement_vs_best_regular_pct"] = 100.0 * df["delta_vs_best_regular"] / df["best_regular_obj_val"].abs().clip(lower=1e-9)

    baseline_mu0 = (
        regular[
            (regular["mu_alfa"] == 0.0)
            & (regular["mu_beta"] == 0.0)
        ][["lam", "budget", "obj_val", "series"]]
        .rename(
            columns={
                "obj_val": "baseline_mu0_obj_val",
                "series": "baseline_mu0_series",
            }
        )
    )
    df = df.merge(baseline_mu0, on=["lam", "budget"], how="left")
    df["delta_vs_mu0"] = df["baseline_mu0_obj_val"] - df["obj_val"]
    df["improvement_vs_mu0_pct"] = 100.0 * df["delta_vs_mu0"] / df["baseline_mu0_obj_val"].abs().clip(lower=1e-9)
    return df.sort_values(["lam", "budget", "is_longrun", "mu_alfa", "mu_beta"]).reset_index(drop=True)


def save_tables(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "objective_comparison_by_budget_with_longrun.csv", index=False)

    pivot = (
        df.pivot_table(
            index=["lam", "budget"],
            columns="series",
            values="obj_val",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.to_csv(OUT_DIR / "objective_comparison_by_budget_with_longrun_pivot.csv", index=False)

    longrun = df[df["is_longrun"]].copy()
    if not longrun.empty:
        longrun = longrun[
            [
                "file",
                "lam",
                "budget",
                "obj_val",
                "objective_gain",
                "used_budget",
                "comp_time",
                "best_regular_obj_val",
                "best_regular_series",
                "delta_vs_best_regular",
                "improvement_vs_best_regular_pct",
                "baseline_mu0_obj_val",
                "baseline_mu0_series",
                "delta_vs_mu0",
                "improvement_vs_mu0_pct",
                "rank_in_budget_lam",
            ]
        ]
        longrun.to_csv(OUT_DIR / "longrun_vs_others_by_budget.csv", index=False)
        longrun.to_csv(OUT_DIR / "longrun_vs_mu0_by_budget.csv", index=False)


def plot_objective_by_budget(df: pd.DataFrame) -> None:
    for lam, df_lam in df.groupby("lam"):
        # Filter for budgets <= 5e5 (500k)
        df_lam = df_lam[df_lam["budget"] <= 5e5].copy()
        if df_lam.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort groups so jittering is consistently ordered by parameters/is_longrun
        series_groups = sorted(df_lam.groupby("series"), key=lambda x: (
            bool(x[1]["is_longrun"].iloc[0]),
            float(x[1]["mu_alfa"].iloc[0]),
            float(x[1]["mu_beta"].iloc[0])
        ))
        
        num_series = len(series_groups)
        for i, (series, df_series) in enumerate(series_groups):
            df_series = df_series.sort_values("budget")
            first_row = df_series.iloc[0]
            is_longrun = bool(first_row["is_longrun"])
            mu_al = float(first_row["mu_alfa"])
            mu_bet = float(first_row["mu_beta"])
            
            # Premium & distinct line/marker styles matching the color scheme of computational times
            if mu_al == 0.0 and mu_bet == 0.0:
                color = "#7f7f7f"  # Grey for Baseline
                linestyle = ":"
                marker = "o"
                linewidth = 1.8
                markersize = 6
                zorder = 2
            elif np.isclose(mu_al, 1e-8) and np.isclose(mu_bet, 1e-3):
                # Light Orange for bliters=10, Light Blue for bliters=3
                color = "#f5b041" if is_longrun else "#5dade2"
                if is_longrun:
                    linestyle = "-"
                    marker = "^"
                    linewidth = 2.4
                    markersize = 7
                    zorder = 8
                else:
                    linestyle = "-."
                    marker = "v"
                    linewidth = 1.8
                    markersize = 6
                    zorder = 4
            elif np.isclose(mu_al, 1e-7) and np.isclose(mu_bet, 5e-3):
                # Dark Orange/Rust for bliters=10, Dark Blue for bliters=3
                color = "#d35400" if is_longrun else "#1f77b4"
                if is_longrun:
                    linestyle = "-"
                    marker = "D"
                    linewidth = 2.4
                    markersize = 7
                    zorder = 10
                else:
                    linestyle = "--"
                    marker = "s"
                    linewidth = 1.8
                    markersize = 6
                    zorder = 5
            else:
                color = "#2ca02c"  # Fallback green
                linestyle = "-" if is_longrun else "--"
                marker = "x"
                linewidth = 1.8
                markersize = 6
                zorder = 3

            # Tiny jittering (1% step) to avoid overlapping lines and points on log-scale x-axis 
            # without distorting visual budget-efficiency
            jitter_factor = 1.0 + (i - (num_series - 1) / 2) * 0.01
            jittered_budgets = df_series["budget"] * jitter_factor

            ax.plot(
                jittered_budgets,
                df_series["obj_val"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                label=series,
                zorder=zorder,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Objective value (lower is better)")
        ax.set_title(f"red_iberia_simple: objective by budget (lam={lam:g})")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"objective_by_budget_lam_{lam:g}_with_longrun_longrun.png", dpi=180)
        plt.close(fig)


def plot_longrun_delta(df: pd.DataFrame) -> None:
    longrun = df[df["is_longrun"]].sort_values(["lam", "budget"])
    if longrun.empty:
        return

    for lam, df_lam in longrun.groupby("lam"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df_lam["budget"],
            df_lam["delta_vs_best_regular"],
            color="#c0392b",
            marker="D",
            linewidth=2.4,
            markersize=6,
        )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Budget")
        ax.set_ylabel("best regular obj - longrun obj")
        ax.set_title(f"Longrun improvement vs best regular run (lam={lam:g})")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"longrun_improvement_vs_best_regular_lam_{lam:g}_longrun.png", dpi=180)
        plt.close(fig)


def plot_longrun_improvement_vs_mu0(df: pd.DataFrame) -> None:
    longrun = df[df["is_longrun"]].sort_values(["lam", "budget"])
    if longrun.empty:
        return

    for lam, df_lam in longrun.groupby("lam"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df_lam["budget"],
            df_lam["improvement_vs_mu0_pct"],
            color="#1f77b4",
            marker="D",
            linewidth=2.6,
            markersize=6,
        )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Improvement vs mu_al=0, mu_b=0 [%]")
        ax.set_title(f"Longrun improvement vs mu=0 baseline (lam={lam:g})")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"longrun_improvement_pct_vs_mu0_lam_{lam:g}_longrun.png", dpi=180)
        plt.close(fig)


def plot_comp_time_by_budget(df: pd.DataFrame) -> None:
    # Filter for lam=10 and budget <= 5e5
    df_lam = df[(df["lam"] == 10.0) & (df["budget"] <= 5e5)].copy()
    if df_lam.empty:
        return

    # Unique budgets and unique series
    budgets = sorted(df_lam["budget"].unique())
    series_list = sorted(df_lam["series"].unique())
    
    # Sort groups so they are consistently ordered in the legend and bars
    series_props = {}
    for s in series_list:
        sub = df_lam[df_lam["series"] == s]
        if not sub.empty:
            row = sub.iloc[0]
            series_props[s] = (
                bool(row["is_longrun"]),
                float(row["mu_alfa"]),
                float(row["mu_beta"])
            )
        else:
            series_props[s] = (False, 0.0, 0.0)
            
    series_list = sorted(series_list, key=lambda s: series_props[s])

    # Position of bars
    n_budgets = len(budgets)
    n_series = len(series_list)
    
    # Width of a single bar (0.8 of the total category width is distributed among series)
    width = 0.8 / n_series
    
    # x positions for groups
    x = np.arange(n_budgets)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, series in enumerate(series_list):
        df_series = df_lam[df_lam["series"] == series]
        # Map budgets to the values they have, filling with 0 if missing
        times = []
        for b in budgets:
            match = df_series[df_series["budget"] == b]
            if not match.empty:
                times.append(float(match["comp_time"].iloc[0]))
            else:
                times.append(0.0)

        # Offset for the current series bar inside the budget group
        offset = (i - (n_series - 1) / 2) * width
        
        # Style/Color matching our professional design system
        first_row = df_series.iloc[0] if not df_series.empty else None
        is_longrun = bool(first_row["is_longrun"]) if first_row is not None else False
        mu_al = float(first_row["mu_alfa"]) if first_row is not None else 0.0
        mu_bet = float(first_row["mu_beta"]) if first_row is not None else 0.0

        if mu_al == 0.0 and mu_bet == 0.0:
            color = "#7f7f7f"  # Grey for Baseline
            hatch = ""
        else:
            # Color represents bliters (is_longrun)
            color = "#e67e22" if is_longrun else "#1f77b4"  # Orange for bliters=10, Blue for bliters=3
            
            # Hatch represents mu_alfa and mu_beta values
            if np.isclose(mu_al, 1e-8) and np.isclose(mu_bet, 1e-3):
                hatch = "//"
            elif np.isclose(mu_al, 1e-7) and np.isclose(mu_bet, 5e-3):
                hatch = "xx"
            else:
                hatch = ""

        ax.bar(
            x + offset,
            times,
            width,
            label=series,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85
        )

    # Helper function to format budget labels beautifully (e.g. 100000 -> 100k)
    def format_budget(b: float) -> str:
        if b >= 1e6:
            return f"{b/1e6:g}M"
        elif b >= 1e3:
            return f"{b/1e3:g}k"
        return f"{b:g}"

    ax.set_xticks(x)
    ax.set_xticklabels([format_budget(b) for b in budgets], fontsize=10, fontweight="bold")
    ax.set_xlabel("Budget", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Computational Time (seconds)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("red_iberia_simple: Computational Time Comparison by Budget (lam=10)", fontsize=13, fontweight="bold", pad=15)
    
    # Grid lines behind the bars
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    
    # Save the plot
    fig.savefig(OUT_DIR / "comp_time_by_budget_lam_10.png", dpi=180)
    plt.close(fig)


def main() -> None:
    df = load_results()
    save_tables(df)
    plot_objective_by_budget(df)
    plot_longrun_delta(df)
    plot_longrun_improvement_vs_mu0(df)
    plot_comp_time_by_budget(df)

    longrun = df[df["is_longrun"]].sort_values(["lam", "budget"])
    if longrun.empty:
        print("No longrun files found.")
        return

    print("Longrun vs mu=0 baseline by budget")
    print(
        longrun[
            [
                "lam",
                "budget",
                "obj_val",
                "baseline_mu0_obj_val",
                "delta_vs_mu0",
                "improvement_vs_mu0_pct",
                "baseline_mu0_series",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
