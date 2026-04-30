import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from mips_6node import parameters_6node_network


SCRIPT_DIR = Path(__file__).resolve().parent
MIP_DIR = SCRIPT_DIR / "6node_hs_prueba_v0"
BLO_DIR = SCRIPT_DIR / "6node_hs_prueba_v0_blo"
RELATIVE_GAP_RE = re.compile(r"Relative gap:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
PERCENT_GAP_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
MIP_STATUS_RE = re.compile(r"--- MIP status \(\d+\):\s*(.+)")
ELAPSED_RE = re.compile(r"elapsed\s+(\d+):(\d+):(\d+(?:\.\d+)?)")
CPLEX_ELAPSED_RE = re.compile(r"Elapsed time =\s*([0-9]+(?:\.[0-9]+)?) sec\.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 6-node BLO python-euler results against MIP results."
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[3e4, 4e4, 5e4, 6e4, 7e4, 8e4],
        help="Budgets to compare.",
    )
    parser.add_argument("--lam", type=int, default=4, help="Lambda value.")
    parser.add_argument("--alfa", type=float, default=0.1, help="Alfa value.")
    parser.add_argument(
        "--mu-alfa",
        type=float,
        default=1e-4,
        dest="mu_alfa",
        help="mu_alfa value for BLO files.",
    )
    parser.add_argument(
        "--mu-beta",
        type=float,
        default=1e-2,
        dest="mu_beta",
        help="mu_beta value for BLO files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "6node_gap_comparison_python.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--time-output",
        type=Path,
        default=SCRIPT_DIR / "6node_time_comparison_python.png",
        help="Output path for the time comparison figure.",
    )
    return parser.parse_args()


def candidate_blo_paths(budget, lam, alfa, mu_alfa, mu_beta):
    budget_int = int(round(budget))
    pattern = (
        f"bud={budget_int}*lam={lam}*alfa={alfa:.6e}*mu_al={mu_alfa:.6e}"
        f"*mu_bet={mu_beta:.6e}*_replica8node_py-euler.mat"
    )
    return sorted(glob.glob(str(BLO_DIR / pattern)))


def find_blo_file(budget, lam, alfa, mu_alfa, mu_beta):
    matches = candidate_blo_paths(budget, lam, alfa, mu_alfa, mu_beta)
    if matches:
        return Path(matches[0])
    raise FileNotFoundError(
        f"No BLO python-euler file found for budget={budget}, lam={lam}, "
        f"alfa={alfa}, mu_alfa={mu_alfa}, mu_beta={mu_beta}."
    )


def load_result(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return sio.loadmat(path)


def parse_mip_gap_from_log(log_path):
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    status_match = MIP_STATUS_RE.findall(text)
    mip_status = status_match[-1].strip().lower() if status_match else ""
    if "integer optimal" in mip_status or mip_status == "optimal":
        return 0.0

    matches = RELATIVE_GAP_RE.findall(text)
    if matches:
        return 100.0 * float(matches[-1])

    mip_status_pos = text.rfind("--- MIP status")
    search_area = text if mip_status_pos == -1 else text[:mip_status_pos]
    percent_matches = PERCENT_GAP_RE.findall(search_area)
    if not percent_matches:
        raise ValueError(f"No MIP gap found in {log_path}")
    return float(percent_matches[-1])


def parse_mip_time_from_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    stop_match = re.search(r"Job .* Stop .* elapsed\s+(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if stop_match:
        hours, minutes, seconds = stop_match.groups()
        return 3600.0 * int(hours) + 60.0 * int(minutes) + float(seconds)

    after_solve_match = re.search(r"Executing after solve: elapsed\s+(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if after_solve_match:
        hours, minutes, seconds = after_solve_match.groups()
        return 3600.0 * int(hours) + 60.0 * int(minutes) + float(seconds)

    cplex_elapsed_matches = CPLEX_ELAPSED_RE.findall(text)
    if cplex_elapsed_matches:
        return float(cplex_elapsed_matches[-1])

    elapsed_matches = ELAPSED_RE.findall(text)
    if elapsed_matches:
        hours, minutes, seconds = elapsed_matches[-1]
        return 3600.0 * int(hours) + 60.0 * int(minutes) + float(seconds)

    raise ValueError(f"No elapsed time found in {log_path}")


def parse_log_flags(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    status_match = MIP_STATUS_RE.findall(text)
    status = status_match[-1].strip() if status_match else "missing MIP status"
    oom_like = (
        status_match == []
        or "No data for symbol >a_level<. Skipping." in text
        or "out of memory" in text.lower()
    )
    return {"status": status, "oom_like": oom_like}


def compare_budget(budget, lam, alfa, mu_alfa, mu_beta, demand, prices, op_link_cost):
    budget_int = int(round(budget))
    mip_path = MIP_DIR / f"2h_bud={budget_int}_lam={int(lam)}.mat"
    blo_path = find_blo_file(budget, lam, alfa, mu_alfa, mu_beta)
    log_path = MIP_DIR / f"log_budget={budget_int}.txt"

    mip_data = load_result(mip_path)
    blo_data = load_result(blo_path)

    f_mip = np.asarray(mip_data["f"], dtype=float).copy()
    a_mip = np.asarray(mip_data["a"], dtype=float).copy()
    f_blo = np.asarray(blo_data["f"], dtype=float)
    a_blo = np.asarray(blo_data["a"], dtype=float)

    mask_f = np.abs(f_blo - f_mip) < 2e-2
    f_mip[mask_f] = f_blo[mask_f]

    mask_a = np.abs(a_blo - a_mip) < 5e-2
    a_mip[mask_a] = a_blo[mask_a]

    obj_mip = float(np.sum(f_mip * demand * prices) - np.sum(op_link_cost * a_mip))
    obj_blo = float(np.sum(f_blo * demand * prices) - np.sum(op_link_cost * a_blo))
    blo_vs_mip_gap = 100.0 * (obj_mip - obj_blo) / obj_mip
    mip_log_gap = parse_mip_gap_from_log(log_path)
    mip_time = parse_mip_time_from_log(log_path)
    log_flags = parse_log_flags(log_path)
    blo_time = float(np.asarray(blo_data.get("comp_time", [[np.nan]]), dtype=float).reshape(-1)[-1])

    return {
        "budget": float(budget),
        "mip_path": mip_path,
        "blo_path": blo_path,
        "log_path": log_path,
        "obj_mip": obj_mip,
        "obj_blo": obj_blo,
        "blo_vs_mip_gap": blo_vs_mip_gap,
        "mip_log_gap": mip_log_gap,
        "mip_time": mip_time,
        "blo_time": blo_time,
        "mip_status": log_flags["status"],
        "mip_oom_like": log_flags["oom_like"],
    }


def plot_results(results, output_path):
    budgets = [int(round(item["budget"])) for item in results]
    blo_vs_mip_gap = [item["blo_vs_mip_gap"] for item in results]
    mip_log_gap = [item["mip_log_gap"] for item in results]

    fig, ax_gap = plt.subplots(figsize=(9.5, 5.8))
    ax_mip = ax_gap.twinx()

    if len(budgets) > 1:
        bar_width = 0.55 * min(np.diff(budgets))
    else:
        bar_width = 5000

    bars = ax_mip.bar(
        budgets,
        mip_log_gap,
        width=bar_width,
        color="#1f77b4",
        alpha=0.18,
        edgecolor="#1f77b4",
        linewidth=1.5,
        label="Optimality gap MIP (log)",
        zorder=1,
    )

    line, = ax_gap.plot(
        budgets,
        blo_vs_mip_gap,
        "x-",
        linewidth=2,
        markersize=8,
        color="#2e7d32",
        label="Gap BLO vs MIP",
        zorder=3,
    )

    gap_min = min(blo_vs_mip_gap)
    gap_max = max(blo_vs_mip_gap)
    gap_span = max(gap_max - gap_min, 0.25)
    gap_pad = 0.2 * gap_span

    ax_gap.set_xlabel("Budget")
    ax_gap.set_ylabel("Gap BLO vs MIP [%]", color="#2e7d32")
    ax_gap.set_ylim(gap_min - gap_pad, gap_max + gap_pad)
    ax_gap.tick_params(axis="y", colors="#2e7d32")
    ax_gap.set_xticks(budgets)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.set_axisbelow(True)

    ax_mip.set_ylabel("Optimality gap MIP [%]", color="#1f77b4")
    ax_mip.tick_params(axis="y", colors="#1f77b4")
    ax_mip.set_ylim(0, max(mip_log_gap) * 1.08)

    ax_gap.set_title("6-node Spain: BLO vs MIP and MIP optimality gap")
    ax_gap.legend([line, bars], [line.get_label(), bars.get_label()], loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_results(results, output_path):
    budgets = np.array([int(round(item["budget"])) for item in results], dtype=float)
    mip_times = np.array([item["mip_time"] for item in results], dtype=float)
    blo_times = np.array([item["blo_time"] for item in results], dtype=float)
    oom_mask = np.array([item["mip_oom_like"] for item in results], dtype=bool)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    if len(budgets) > 1:
        width = 0.35 * min(np.diff(budgets))
    else:
        width = 5000

    mip_bars = ax.bar(
        budgets - width / 2,
        mip_times,
        width=width,
        color="#1f77b4",
        alpha=0.8,
        label="MIP",
    )
    blo_bars = ax.bar(
        budgets + width / 2,
        blo_times,
        width=width,
        color="#2e7d32",
        alpha=0.8,
        label="BLO python-euler",
    )

    for idx, is_oom in enumerate(oom_mask):
        if not is_oom:
            continue
        mip_bars[idx].set_facecolor("#f4a261")
        mip_bars[idx].set_edgecolor("#c1121f")
        mip_bars[idx].set_hatch("///")
        ax.text(
            budgets[idx] - width / 2,
            mip_times[idx] * 1.08,
            "OOM",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            color="#c1121f",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Computation time [s]")
    ax.set_title("6-node Spain: computation time BLO vs MIP")
    ax.set_xticks(budgets)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(results):
    print("budget | blo_vs_mip_gap[%] | mip_log_gap[%] | mip_time[s] | blo_time[s] | oom | obj_mip | obj_blo")
    for item in results:
        print(
            f"{int(round(item['budget'])):6d} | "
            f"{item['blo_vs_mip_gap']:17.6f} | "
            f"{item['mip_log_gap']:14.6f} | "
            f"{item['mip_time']:11.3f} | "
            f"{item['blo_time']:11.3f} | "
            f"{str(item['mip_oom_like']):3s} | "
            f"{item['obj_mip']:.6f} | "
            f"{item['obj_blo']:.6f}"
        )


def main():
    args = parse_args()
    params = parameters_6node_network(SCRIPT_DIR)
    demand = params.demand / 365.0

    results = []
    for budget in args.budgets:
        results.append(
            compare_budget(
                budget=budget,
                lam=args.lam,
                alfa=args.alfa,
                mu_alfa=args.mu_alfa,
                mu_beta=args.mu_beta,
                demand=demand,
                prices=params.prices,
                op_link_cost=params.op_link_cost,
            )
        )

    print_summary(results)
    plot_results(results, args.output)
    plot_time_results(results, args.time_output)
    print(f"\nSaved figure to: {args.output}")
    print(f"Saved time figure to: {args.time_output}")


if __name__ == "__main__":
    main()
