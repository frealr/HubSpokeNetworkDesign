import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from generate_data import parameters_8node_network


SCRIPT_DIR = Path(__file__).resolve().parent
MIP_DIR = SCRIPT_DIR / "8node_hs_prueba_v0"
BLO_DIR = SCRIPT_DIR / "8node_hs_prueba_v0_blo"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare 8-node BLO python-euler results against MIP results."
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5],
        help="Budgets to compare.",
    )
    parser.add_argument("--lam", type=int, default=4, help="Lambda value.")
    parser.add_argument("--alfa", type=float, default=0.1, help="Alfa value.")
    parser.add_argument(
        "--mu-alfa",
        type=float,
        default=1e-7,
        dest="mu_alfa",
        help="mu_alfa value for BLO files.",
    )
    parser.add_argument(
        "--mu-beta",
        type=float,
        default=0.2,
        dest="mu_beta",
        help="mu_beta value for BLO files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "8node_gap_comparison_python.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--time-output",
        type=Path,
        default=SCRIPT_DIR / "8node_time_comparison_python.png",
        help="Output path for the time comparison figure.",
    )
    return parser.parse_args()


def to_scalar(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[-1])


def candidate_blo_patterns(budget, lam, alfa, mu_alfa, mu_beta):
    budget_int = int(round(budget))
    budget_float = float(budget)
    return [
        BLO_DIR / (
            f"bud={budget_float}_lam={lam}_alfa={alfa}_mu_al={mu_alfa}"
            f"_mu_bet={mu_beta}_python-euler.mat"
        ),
        BLO_DIR / (
            f"bud={budget_int}.0_lam={lam}_alfa={alfa}_mu_al={mu_alfa}"
            f"_mu_bet={mu_beta}_python-euler.mat"
        ),
        BLO_DIR / (
            f"bud={budget_int}_lam={lam}_alfa={alfa}_mu_al={mu_alfa}"
            f"_mu_bet={mu_beta}_python-euler.mat"
        ),
    ]


def find_blo_file(budget, lam, alfa, mu_alfa, mu_beta):
    for candidate in candidate_blo_patterns(budget, lam, alfa, mu_alfa, mu_beta):
        if candidate.is_file():
            return candidate

    pattern = (
        f"bud={int(round(budget))}*lam={lam}*alfa={alfa}*mu_al={mu_alfa}"
        f"*mu_bet={mu_beta}*_python-euler.mat"
    )
    matches = sorted(glob.glob(str(BLO_DIR / pattern)))
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


def compare_budget(budget, lam, alfa, mu_alfa, mu_beta, demand, prices, op_link_cost):
    mip_path = MIP_DIR / f"2h_budget={int(round(budget))}_lam={int(lam)}.mat"
    blo_path = find_blo_file(budget, lam, alfa, mu_alfa, mu_beta)

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
    mip_reported_gap = 100.0 * to_scalar(mip_data.get("mipgap", [0.0]))
    mip_time = to_scalar(mip_data.get("comp_time", [np.nan]))
    blo_time = to_scalar(blo_data.get("comp_time", [np.nan]))

    return {
        "budget": float(budget),
        "mip_path": mip_path,
        "blo_path": blo_path,
        "obj_mip": obj_mip,
        "obj_blo": obj_blo,
        "blo_vs_mip_gap": blo_vs_mip_gap,
        "mip_reported_gap": mip_reported_gap,
        "mip_time": mip_time,
        "blo_time": blo_time,
    }


def plot_results(results, output_path):
    budgets = np.array([int(round(item["budget"])) for item in results], dtype=float)
    blo_vs_mip_gap = np.array([item["blo_vs_mip_gap"] for item in results], dtype=float)
    mip_reported_gap = np.array([item["mip_reported_gap"] for item in results], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    
    if len(budgets) > 1:
        width = 0.35 * min(np.diff(budgets))
    else:
        width = 5000

    # Grouped bars side-by-side
    bars_mip = ax.bar(
        budgets - width / 2,
        mip_reported_gap,
        width=width,
        color="#1f77b4",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
        label="Optimality gap MIP (.mat)",
    )
    bars_blo = ax.bar(
        budgets + width / 2,
        blo_vs_mip_gap,
        width=width,
        color="#2e7d32",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.7,
        label="Gap BLO vs MIP",
    )

    ax.set_xlabel("Budget", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gap [%]", fontsize=11, fontweight="bold")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("8-node Spain: Optimality Gap Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(budgets)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_results(results, output_path):
    budgets = np.array([int(round(item["budget"])) for item in results], dtype=float)
    mip_times = np.array([item["mip_time"] for item in results], dtype=float)
    blo_times = np.array([item["blo_time"] for item in results], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    if len(budgets) > 1:
        width = 0.35 * min(np.diff(budgets))
    else:
        width = 5000

    ax.bar(
        budgets - width / 2,
        mip_times,
        width=width,
        color="#1f77b4",
        alpha=0.8,
        label="MIP",
    )
    ax.bar(
        budgets + width / 2,
        blo_times,
        width=width,
        color="#2e7d32",
        alpha=0.8,
        label="BLO python-euler",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Computation time [s]")
    ax.set_title("8-node Spain: computation time BLO vs MIP")
    ax.set_xticks(budgets)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(results):
    print("budget | blo_vs_mip_gap[%] | mip_reported_gap[%] | mip_time[s] | blo_time[s] | obj_mip | obj_blo")
    for item in results:
        print(
            f"{int(round(item['budget'])):6d} | "
            f"{item['blo_vs_mip_gap']:17.6f} | "
            f"{item['mip_reported_gap']:19.6f} | "
            f"{item['mip_time']:11.3f} | "
            f"{item['blo_time']:11.3f} | "
            f"{item['obj_mip']:.6f} | "
            f"{item['obj_blo']:.6f}"
        )


def main():
    args = parse_args()
    (
        _n,
        _link_cost,
        _station_cost,
        _hub_cost,
        _link_capacity_slope,
        _station_capacity_slope,
        demand,
        prices,
        _load_factor,
        op_link_cost,
        _congestion_coef_stations,
        _congestion_coef_links,
        _travel_time,
        _alt_utility,
        _a_nom,
        _tau,
        _eta,
        _a_max,
        _candidates,
        _omega_t,
        _omega_p,
    ) = parameters_8node_network()

    results = []
    for budget in args.budgets:
        result = compare_budget(
            budget=budget,
            lam=args.lam,
            alfa=args.alfa,
            mu_alfa=args.mu_alfa,
            mu_beta=args.mu_beta,
            demand=demand,
            prices=prices,
            op_link_cost=op_link_cost,
        )
        results.append(result)

    print_summary(results)
    plot_results(results, args.output)
    plot_time_results(results, args.time_output)
    print(f"\nSaved figure to: {args.output}")
    print(f"Saved time figure to: {args.time_output}")


if __name__ == "__main__":
    main()
