from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


DEFAULT_GAMS = Path("/opt/gams/gams49.6_linux_x64_64_sfx/gams")
MATLAB_NREG = 20
N_AIRLINES = 5


@dataclass
class NetworkParams:
    n: int
    link_cost: np.ndarray
    station_cost: np.ndarray
    hub_cost: np.ndarray
    link_capacity_slope: np.ndarray
    station_capacity_slope: np.ndarray
    demand: np.ndarray
    prices: np.ndarray
    load_factor: np.ndarray
    op_link_cost: np.ndarray
    congestion_coef_stations: np.ndarray
    congestion_coef_links: np.ndarray
    travel_time: np.ndarray
    alt_utility: np.ndarray
    a_nom: float
    tau: float
    eta: float
    a_max: float
    candidates_raw: np.ndarray
    omega_t: float
    omega_p: float


def read_matrix_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def parameters_6node_network(script_dir: Path) -> NetworkParams:
    n = 6
    candidates = np.ones((n, n)) - np.eye(n)

    distance = read_matrix_csv(script_dir / "distance.csv")
    prices = read_matrix_csv(script_dir / "prices.csv")
    demand = read_matrix_csv(script_dir / "demand.csv")

    omega_t = -0.02
    omega_p = -0.02

    link_cost = 10.0 * distance
    eye_mask = np.eye(n, dtype=bool)
    link_cost[eye_mask] = 1e4
    link_cost[link_cost == 0] = 1e4

    station_cost = 3e3 * np.ones(n)
    hub_cost = 5e3 * np.ones(n)
    link_capacity_slope = 0.2 * link_cost
    station_capacity_slope = (5 * 5e2 + 4 * 50 * 8) * np.ones(n)
    load_factor = 0.25 * np.ones(n)
    congestion_coef_stations = 0.1 * np.ones(n)
    congestion_coef_links = 0.1 * np.ones((n, n))

    takeoff_time = 20
    landing_time = 20
    taxi_time = 10
    cruise_time = 60 * distance / 800
    travel_time = cruise_time + takeoff_time + landing_time + taxi_time
    travel_time[eye_mask] = 0

    # Match MATLAB's rng(123); rand(...) usage more closely than default_rng.
    rng = np.random.RandomState(123)
    p_escala = 0.4
    alt_utility = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            escala = rng.rand(N_AIRLINES) < p_escala
            alt_time_vec = travel_time[i, j] * (1 + 0.5 * escala.astype(float)) + 60 * escala.astype(float)
            alt_price_vec = prices[i, j] + 0.3 * prices[i, j] * (rng.rand(N_AIRLINES) - 0.5)
            alt_u = np.log(np.sum(np.exp(omega_p * alt_price_vec + omega_t * alt_time_vec))) - np.log(N_AIRLINES)
            alt_utility[i, j] = alt_u
            alt_utility[j, i] = alt_u

    a_nom = 171
    tau = 0.85
    eta = 0.3
    a_max = 1e9

    return NetworkParams(
        n=n,
        link_cost=link_cost,
        station_cost=station_cost,
        hub_cost=hub_cost,
        link_capacity_slope=link_capacity_slope,
        station_capacity_slope=station_capacity_slope,
        demand=demand,
        prices=prices,
        load_factor=load_factor,
        op_link_cost=7600 * travel_time / 60,
        congestion_coef_stations=congestion_coef_stations,
        congestion_coef_links=congestion_coef_links,
        travel_time=travel_time,
        alt_utility=alt_utility,
        a_nom=a_nom,
        tau=tau,
        eta=eta,
        a_max=a_max,
        candidates_raw=candidates,
        omega_t=omega_t,
        omega_p=omega_p,
    )


def get_linearization(
    n: int, nreg: int, alt_utility: np.ndarray, vals_regs: np.ndarray, n_airlines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dmax = np.zeros((nreg, n, n))
    dmin = np.zeros((nreg, n, n))
    lin_coef = np.zeros((nreg, n, n))
    bord = np.zeros((nreg, n, n))

    for o in range(n):
        for d in range(n):
            u = alt_utility[o, d]
            for r in range(nreg - 1):
                dmax[r, o, d] = min(0.0, u + math.log(n_airlines * vals_regs[r] / (1 - vals_regs[r])))
            dmax[nreg - 1, o, d] = 0.0
            dmin[0, o, d] = -3e1

            for r in range(1, nreg):
                dmin[r, o, d] = dmax[r - 1, o, d]

            for r in range(1, nreg - 1):
                if dmax[r, o, d] == dmin[r, o, d]:
                    lin_coef[r, o, d] = 0.0
                    bord[r, o, d] = vals_regs[r]
                else:
                    lin_coef[r, o, d] = (vals_regs[r] - vals_regs[r - 1]) / (dmax[r, o, d] - dmin[r, o, d])
                    bord[r, o, d] = vals_regs[r - 1]

            lin_coef[0, o, d] = vals_regs[0] / (dmax[0, o, d] - dmin[0, o, d])
            bord[0, o, d] = 0.0

            if dmin[nreg - 1, o, d] == 0:
                lin_coef[nreg - 1, o, d] = 0.0
            else:
                lin_coef[nreg - 1, o, d] = (1 - vals_regs[nreg - 2]) / (0 - dmin[nreg - 1, o, d])
            bord[nreg - 1, o, d] = vals_regs[nreg - 2]

    return lin_coef, bord, dmin


def matlab_sprintf_d(val: float) -> str:
    """Replicate MATLAB's sprintf('%d', val) behavior."""
    if val == int(val) and abs(val) < 1e15:
        return str(int(val))
    else:
        return f"{val:e}"


def write_scalar(path: Path, value: float) -> None:
    path.write_text(matlab_sprintf_d(value), encoding="ascii")


def write_gams_param_ii(path: Path, matrix: np.ndarray) -> None:
    n1, n2 = matrix.shape
    if n1 != n2:
        raise ValueError("M debe ser cuadrada para dominio (i,i).")

    with path.open("w", encoding="ascii") as fh:
        for r in range(n1):
            for c in range(n2):
                fh.write(f"i{r + 1}.i{c + 1} {matrix[r, c]:.12g}\n")


def write_gams_param_iii(path: Path, tensor: np.ndarray) -> None:
    n1, n2, n3 = tensor.shape
    with path.open("w", encoding="ascii") as fh:
        for s in range(n1):
            for r in range(n2):
                for c in range(n3):
                    fh.write(f"seg{s + 1}.i{r + 1}.i{c + 1} {tensor[s, r, c]:.12g}\n")


def write_gams_param1d_full(path: Path, vector: np.ndarray) -> None:
    vector = np.asarray(vector).reshape(-1)
    with path.open("w", encoding="ascii") as fh:
        for idx, value in enumerate(vector, start=1):
            fh.write(f"i{idx} {value:.12g}\n")


def get_budget(
    s: np.ndarray,
    sh: np.ndarray,
    a: np.ndarray,
    n: int,
    station_cost: np.ndarray,
    station_capacity_slope: np.ndarray,
    hub_cost: np.ndarray,
    link_cost: np.ndarray,
    lam: float,
) -> float:
    del a, link_cost
    budget = 0.0
    for i in range(n):
        if s[i] > 5e-2 and sh[i] < 5e-2:
            budget += station_cost[i] + station_capacity_slope[i] * s[i]
        if sh[i] > 5e-2:
            budget += station_cost[i] + lam * hub_cost[i] + station_capacity_slope[i] * (sh[i] + s[i])
    return budget


def get_obj_val(
    op_link_cost: np.ndarray, prices: np.ndarray, a: np.ndarray, f: np.ndarray, demand: np.ndarray
) -> tuple[float, float, float]:
    op_obj = float(np.sum(op_link_cost * a))
    pax_obj = float(-np.sum(prices * demand * f))
    return pax_obj + op_obj, pax_obj, op_obj


def write_initial_data(export_dir: Path, params: NetworkParams) -> None:
    demand = params.demand / 365.0
    nreg = MATLAB_NREG
    vals_regs = np.linspace(0.005, 0.995, nreg - 1)
    lin_coef, bord, b = get_linearization(params.n, nreg, params.alt_utility, vals_regs, N_AIRLINES)

    candidates = np.zeros((params.n, params.n))
    for i in range(params.n):
        candidates[i, params.candidates_raw[i, :] > 0] = 1

    alfa_od = np.ones((params.n, params.n))
    beta_od = np.ones((params.n, params.n))
    gamma = 20

    write_gams_param_iii(export_dir / "lin_coef.txt", lin_coef)
    write_gams_param_iii(export_dir / "b.txt", b)
    write_gams_param_iii(export_dir / "bord.txt", bord)

    write_gams_param_ii(export_dir / "demand.txt", demand)
    write_gams_param_ii(export_dir / "travel_time.txt", params.travel_time)
    write_gams_param_ii(export_dir / "alt_utility.txt", params.alt_utility)
    write_gams_param_ii(export_dir / "link_cost.txt", params.link_cost)
    write_gams_param_ii(export_dir / "link_capacity_slope.txt", params.link_capacity_slope)
    write_gams_param_ii(export_dir / "prices.txt", params.prices)
    write_gams_param_ii(export_dir / "op_link_cost.txt", params.op_link_cost)
    write_gams_param_ii(export_dir / "candidates.txt", candidates)
    write_gams_param_ii(export_dir / "congestion_coefs_links.txt", params.congestion_coef_links)
    write_gams_param_ii(export_dir / "alfa_od.txt", alfa_od)
    write_gams_param_ii(export_dir / "beta_od.txt", beta_od)
    write_scalar(export_dir / "gamma.txt", gamma)

    write_gams_param1d_full(export_dir / "station_cost.txt", params.station_cost)
    write_gams_param1d_full(export_dir / "hub_cost.txt", params.hub_cost)
    write_gams_param1d_full(export_dir / "station_capacity_slope.txt", params.station_capacity_slope)
    write_gams_param1d_full(export_dir / "congestion_coefs_stations.txt", params.congestion_coef_stations)

    a_prev = 1e4 * np.ones((params.n, params.n))
    s_prev = 1e4 * np.ones(params.n)
    sh_prev = 1e-3 * s_prev

    write_gams_param_ii(export_dir / "a_prev.txt", a_prev)
    write_gams_param1d_full(export_dir / "s_prev.txt", s_prev)
    write_gams_param1d_full(export_dir / "sh_prev.txt", sh_prev)


def read_declared_segment_count(path: Path) -> int:
    for line in path.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if not stripped.startswith("Set seg / seg1*seg"):
            continue
        tail = stripped.removeprefix("Set seg / seg1*seg")
        number = tail.split()[0].rstrip("/;")
        return int(number)
    raise ValueError(f"No pude leer el numero de segmentos en {path}")


def read_numeric_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric_df.empty:
        header_values = pd.to_numeric(pd.Index(df.columns), errors="coerce")
        header_values = header_values[~pd.isna(header_values)]
        if len(header_values) > 0:
            return pd.DataFrame([header_values.to_numpy(dtype=float)])
    if numeric_df.empty:
        raise ValueError(f"La hoja {sheet_name!r} en {path} no contiene datos numéricos.")
    return numeric_df


def read_vector_sheet(path: Path, sheet_name: str) -> np.ndarray:
    return read_numeric_sheet(path, sheet_name).iloc[0].to_numpy(dtype=float)


def read_matrix_sheet(path: Path, sheet_name: str, n: int) -> np.ndarray:
    numeric_df = read_numeric_sheet(path, sheet_name)
    return numeric_df.iloc[:n, :n].to_numpy(dtype=float)


def build_fij_tensor(path: Path, n: int) -> np.ndarray:
    df = pd.read_csv(path)
    for col in ["i", "j", "o", "d"]:
        df[col] = df[col].astype(str).str.strip().str.extract(r"(\d+)").astype(int) - 1
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    fij = np.zeros((n, n, n, n))
    for row in df.itertuples(index=False):
        fij[row.i, row.j, row.o, row.d] += row.value
    return fij


def compute_sim_mip(
    script_dir: Path,
    export_dir: Path,
    output_dir: Path,
    gams_exe: Path,
    lam: float,
    alfa: float,
    budget: float,
    run_gams: bool = True,
) -> dict[str, np.ndarray | float]:
    params = parameters_6node_network(script_dir)
    demand = params.demand / 365.0

    write_scalar(export_dir / "lam.txt", lam)
    write_scalar(export_dir / "alfa.txt", alfa)
    write_scalar(export_dir / "budget.txt", budget)

    a_prev = 1e4 * np.ones((params.n, params.n))
    s_prev = 1e4 * np.ones(params.n)
    sh_prev = s_prev.copy()
    sh_prev[1] *= 0.5
    write_gams_param_ii(export_dir / "a_prev.txt", a_prev)
    write_gams_param1d_full(export_dir / "s_prev.txt", s_prev)
    write_gams_param1d_full(export_dir / "sh_prev.txt", sh_prev)

    output_dir.mkdir(exist_ok=True)
    log_path = output_dir / f"log_budget={int(budget)}.txt"
    
    output_lines: list[str] = []
    if run_gams:
        cmd = [str(gams_exe), str(script_dir / "mip.gms")]
        with log_path.open("w", encoding="utf-8") as log_file:
            with subprocess.Popen(
                cmd,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()
                    output_lines.append(line)
                returncode = process.wait()

        if returncode != 0:
            raise RuntimeError(
                f"GAMS falló con código {returncode}\nSALIDA:\n{''.join(output_lines)}"
            )

    workbook = script_dir / "output_all.xlsx"
    s = read_vector_sheet(workbook, "s_level")
    sh = read_vector_sheet(workbook, "sh_level")
    a = read_matrix_sheet(workbook, "a_level", params.n)
    f = read_matrix_sheet(workbook, "f_level", params.n)
    fext = read_matrix_sheet(workbook, "fext_level", params.n)
    mipgap_df = read_numeric_sheet(workbook, "mip_opt_gap")
    mipgap = mipgap_df.to_numpy().flatten() if not mipgap_df.empty else np.array([0])
    
    comp_time_df = read_numeric_sheet(workbook, "solver_time")
    comp_time = comp_time_df.to_numpy().flatten()[-1] if not comp_time_df.empty else 0
    fij = build_fij_tensor(script_dir / "fij_long.csv", params.n)

    obj_val, pax_obj, op_obj = get_obj_val(params.op_link_cost, params.prices, a, f, demand)
    used_budget = get_budget(
        s=s,
        sh=sh,
        a=a,
        n=params.n,
        station_cost=params.station_cost,
        station_capacity_slope=params.station_capacity_slope,
        hub_cost=params.hub_cost,
        link_cost=params.link_cost,
        lam=lam,
    )

    output_dir.mkdir(exist_ok=True)
    result_path = output_dir / f"python_euler_bud={int(budget)}_lam={int(lam)}.mat"
    payload = {
        "s": s,
        "sprim": s.copy(),
        "deltas": np.zeros(params.n),
        "a": a,
        "f": f,
        "fext": fext,
        "fij": fij,
        "comp_time": comp_time,
        "used_budget": used_budget,
        "pax_obj": pax_obj,
        "op_obj": op_obj,
        "obj_val": obj_val,
        "mipgap": mipgap,
        "sh": sh,
    }
    savemat(result_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traducción a Python de mips_6node.m")
    parser.add_argument("--gams", type=Path, default=DEFAULT_GAMS, help="Ruta al ejecutable de GAMS")
    parser.add_argument("--lam", type=float, default=4.0, help="Valor de lam")
    parser.add_argument("--alfa", type=float, default=0.5, help="Valor de alfa")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="*",
        default=[80000],
        help="Lista de presupuestos",
    )
    parser.add_argument(
        "--skip-gams",
        action="store_true",
        help="No ejecuta GAMS y reutiliza output_all.xlsx/fij_long.csv existentes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    export_dir = script_dir / "export_txt"
    output_dir = script_dir / "6node_hs_prueba_v0"
    export_dir.mkdir(exist_ok=True)

    declared_nreg = read_declared_segment_count(script_dir / "param_definition.gms")
    if declared_nreg != MATLAB_NREG:
        raise ValueError(
            f"param_definition.gms declara {declared_nreg} segmentos, pero mips_6node.m usa {MATLAB_NREG}. "
            "Alinea ambos antes de ejecutar el modelo."
        )

    params = parameters_6node_network(script_dir)
    write_initial_data(export_dir, params)

    for budget in args.budgets:
        result = compute_sim_mip(
            script_dir=script_dir,
            export_dir=export_dir,
            output_dir=output_dir,
            gams_exe=args.gams,
            lam=args.lam,
            alfa=args.alfa,
            budget=budget,
            run_gams=not args.skip_gams,
        )
        print(f"budget={budget:.0f} obj_val={float(result['obj_val']):.6g}")


if __name__ == "__main__":
    main()
