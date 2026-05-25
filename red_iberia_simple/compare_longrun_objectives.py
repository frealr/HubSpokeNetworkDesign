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
    base = f"mu_al={row['mu_alfa']:.0e}, mu_b={row['mu_beta']:.0e}"
    if row["is_longrun"]:
        return f"{base} longrun"
    if np.isclose(row["mu_alfa"], 1e-8) and np.isclose(row["mu_beta"], 1e-3):
        return f"{base} bliters=3"
    return base


def series_label(row: pd.Series) -> str:
    base = f"mu_al={row['mu_alfa']:.0e} | mu_b={row['mu_beta']:.0e}"
    if row["is_longrun"]:
        return f"{base} | longrun"
    if np.isclose(row["mu_alfa"], 1e-8) and np.isclose(row["mu_beta"], 1e-3):
        return f"{base} | bliters=3"
    return base


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
            
            # Premium & distinct line/marker styles
            if is_longrun:
                linestyle, marker, linewidth, markersize = "-", "D", 2.6, 7
            elif mu_al == 0.0 and mu_bet == 0.0:
                linestyle, marker, linewidth, markersize = ":", "o", 1.8, 5
            elif mu_al == 1e-7 and mu_bet == 5e-3:
                linestyle, marker, linewidth, markersize = "--", "s", 1.8, 5
            elif np.isclose(mu_al, 1e-8) and np.isclose(mu_bet, 1e-3):
                linestyle, marker, linewidth, markersize = "-.", "v", 1.8, 5
            else:
                linestyle, marker, linewidth, markersize = "-.", "^", 1.8, 5

            ax.plot(
                df_series["budget"],
                df_series["obj_val"],
                linestyle=linestyle,
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                label=series,
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


def main() -> None:
    df = load_results()
    save_tables(df)
    plot_objective_by_budget(df)
    plot_longrun_delta(df)
    plot_longrun_improvement_vs_mu0(df)

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
