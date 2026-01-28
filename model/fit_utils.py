import os
import re
import numpy as np
import networkx as nx
from scipy.io import mmread
from scipy.stats import poisson
from math import log, exp

from utils import average_shortest_path_length, compute_degree_diffs


def find_data_file(folder_path: str):
    """
    Search for .mtx files in subfolders first
    if not found, search for .edges files
    return None if neither is found
    """
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".mtx"):
            return os.path.join(folder_path, fn)

    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".edges"):
            return os.path.join(folder_path, fn)
    return None


def load_real_graph(path: str):
    """
    Read 
    1. .mtx or .edges files and 
    2. compute actual C/L and 
    3. scale information
    """
    if path.lower().endswith(".mtx"):
        M = mmread(path).tocsr()
        if hasattr(nx, "from_scipy_sparse_array"):
            G = nx.from_scipy_sparse_array(M)
        else:
            G = nx.from_scipy_sparse_matrix(M)
    elif path.lower().endswith(".edges"):
        G = nx.read_edgelist(path, nodetype=int, comments="#")
        G = nx.Graph(G)
    else:
        raise ValueError(f"Unsupported file formats: {path}")

    N = G.number_of_nodes()
    E = G.number_of_edges()
    C_real = nx.average_clustering(G)
    L_real = average_shortest_path_length(G)
    deg_hist = nx.degree_histogram(G)
    return {"G": G, "N": N, "E": E, "C": C_real, "L": L_real,"deg_hist": deg_hist}


def distribution_distance(real_deg_hist, sim_deg_hist):
    real_deg_hist = np.asarray(real_deg_hist, dtype=float)
    sim_deg_hist = np.asarray(sim_deg_hist, dtype=float)

    n_nodes = real_deg_hist.sum()
    if n_nodes <= 0:
        return float("inf")

    max_k_real = len(real_deg_hist) - 1
    max_k_sim = len(sim_deg_hist) - 1
    allowed_range = int(round(max_k_real * 0.2))
    penalty_maxdeg = abs(max_k_sim - max_k_real) * 100 if abs(max_k_sim - max_k_real) > allowed_range else 0

    L = max(len(real_deg_hist), len(sim_deg_hist))
    real_deg_hist = np.pad(real_deg_hist, (0, L - len(real_deg_hist)))
    sim_deg_hist = np.pad(sim_deg_hist, (0, L - len(sim_deg_hist)))

    diffs = np.abs(real_deg_hist - sim_deg_hist)
    tolerance = int(round(n_nodes * 0.05))
    penalty = diffs - tolerance
    penalty[penalty < 0] = 0

    return np.sum(penalty) + penalty_maxdeg


BOUNDS = ((0, 30), (0, 15), (0.00, 0.20), (0.80, 0.98))


def clip_params(p, bounds=BOUNDS):
    return (int(np.clip(p[0], *bounds[0])),
            int(np.clip(p[1], *bounds[1])),
            round(float(np.clip(p[2], *bounds[2])), 3),
            round(float(np.clip(p[3], *bounds[3])), 3))


def propose(p, rng, step_int=1, step_thr=0.02, big_prob=0.1, bounds=BOUNDS):
    add_l, dlt_l, add_t, dlt_t = p
    i = rng.integers(0, 4)
    big = rng.random() < big_prob
    if i == 0:   add_l += (3 if big else step_int) * (1 if rng.random() < 0.5 else -1)
    elif i == 1: dlt_l += (3 if big else step_int) * (1 if rng.random() < 0.5 else -1)
    elif i == 2: add_t += (0.06 if big else step_thr) * (1 if rng.random() < 0.5 else -1)
    else:        dlt_t += (0.06 if big else step_thr) * (1 if rng.random() < 0.5 else -1)
    return clip_params((add_l, dlt_l, add_t, dlt_t), bounds)


def calibrate_T0(p0, energy_fn, rng, tries=20, target_accept=0.8, bounds=BOUNDS):
    E0, _ = energy_fn(p0)
    deltas = []
    for _ in range(tries):
        q = propose(p0, rng, bounds=bounds)
        Eq, _ = energy_fn(q)
        if Eq - E0 > 0:
            deltas.append(Eq - E0)
    dpos = np.mean(deltas) if deltas else 1.0
    return max(-dpos / log(target_accept), 1e-6)


def SA_once(p0, energy_fn, rng_seed=123, alpha=0.95, L=80, Tmin_factor=1e-3, max_layers=60,
            target_acc_low=0.25, target_acc_high=0.60, bounds=BOUNDS):
    rng = np.random.default_rng(rng_seed)
    p = clip_params(p0, bounds=bounds)
    T = calibrate_T0(p, energy_fn, rng, bounds=bounds)
    Tmin = max(T * Tmin_factor, 1e-8)

    E, deg = energy_fn(p)
    best_p, best_E, best_deg = p, E, deg
    step_int, step_thr = 1, 0.02
    noimprove, layer = 0, 0

    while T > Tmin and layer < max_layers:
        accepts = 0
        for _ in range(L):
            q = propose(p, rng, step_int, step_thr, bounds=bounds)
            Eq, qdeg = energy_fn(q)
            dE = Eq - E
            if dE <= 0 or rng.random() < exp(-dE / T):
                p, E, deg = q, Eq, qdeg
                accepts += 1
                if E < best_E:
                    best_p, best_E, best_deg = p, E, deg
                    noimprove = 0
        acc_rate = accepts / L
        step_int = max(1, int(round(step_int * (0.8 if acc_rate < target_acc_low else 1.25))))
        step_thr = max(0.005, step_thr * (0.8 if acc_rate < target_acc_low else 1.25))
        T *= alpha
        layer += 1
        noimprove += 1
        if noimprove >= 8: break
    return best_p, best_E, best_deg


def jsd_poisson_from_hist(hist, eps=1e-12):
    hist = np.asarray(hist, dtype=float)
    n = hist.sum()
    if n <= 0:
        return np.nan
    ks = np.arange(len(hist))
    lam = (ks * hist).sum() / n
    p = hist / n
    q = poisson.pmf(ks, lam)
    p = np.clip(p, eps, 1)
    p /= p.sum()
    q = np.clip(q, eps, 1)
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

# =========================
# dynamic util
# =========================
def find_dynamic_gexf_files(folder_path: str):
    """
    Search all 
        .gexf files in the subfolder, sort them by filename, and
        return None if none are found else the list
    """
    files = []
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith(".gexf"):
            files.append(os.path.join(folder_path, fn))
    return files if files else None


def load_dyanamic_real_graphs(paths: list):
    snapshots = []
    for path in paths:
        G = nx.read_gexf(path)
        snapshots.append(G)

    if not snapshots:
        raise ValueError("No snapshots loaded")

    N_initial = snapshots[0].number_of_nodes()
    E_initial = snapshots[0].number_of_edges()
    deg_diffs = compute_degree_diffs(snapshots)
    return {"Gs": snapshots, "N_initial": N_initial, "E_initial": E_initial, "deg_diffs": deg_diffs}


def load_time_series_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    def parse_series(label):
        pattern = rf"{label}\s*=\s*\[(.*?)\];"
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"{label} not found in {txt_path}")
        seq = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        return seq

    time1 = parse_series("Time1")
    time2 = parse_series("Time2")
    return {"Time1": time1, "Time2": time2}
    
def list_snapshot_gexf_files(folder_path: str):
    """
    Return sorted snapshot .gexf files; prefer snapshot_*.gexf naming if present.
    """
    all_gexf = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".gexf")]
    if not all_gexf:
        return []
    snapshot_like = [p for p in all_gexf if os.path.basename(p).lower().startswith("snapshot_")]
    files = snapshot_like if snapshot_like else all_gexf
    return sorted(files)


def load_real_series_from_metrics(metrics_csv: str, target_len: int):
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(metrics_csv)

    series = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if "S_LCC" not in header or "E_glob" not in header:
            raise ValueError("metrics.csv must contain S_LCC and E_glob columns")
        idx_lcc = header.index("S_LCC")
        idx_eff = header.index("E_glob")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            series.append({
                "S_LCC": float(parts[idx_lcc]),
                "E_glob": float(parts[idx_eff]),
            })
            if len(series) >= target_len:
                break

    if len(series) < target_len:
        raise RuntimeError("metrics length < target_len")

    return series
def distribution_distance_dynamic_luq(real_diffs, sim_diffs):

    num_trans = min(len(real_diffs), len(sim_diffs))
    penalty_trans_num = abs(len(real_diffs) - len(sim_diffs)) * 1000
    total_penalty = 0.0
    n_nodes_total = 0.0

    for i in range(num_trans):
        real_hist, real_offset = real_diffs[i]
        sim_hist, sim_offset = sim_diffs[i]

        real_hist = np.asarray(real_hist, dtype=float)
        sim_hist = np.asarray(sim_hist, dtype=float)

        n_nodes = real_hist.sum()
        if n_nodes <= 0:
            return float("inf")
        n_nodes_total += n_nodes

        min_real, max_real = real_offset, real_offset + len(real_hist) - 1
        min_sim, max_sim = sim_offset, sim_offset + len(sim_hist) - 1
        allowed_range = int(round((max_real - min_real) * 0.2))
        penalty_minmax = (abs(min_sim - min_real) + abs(max_sim - max_real)) * 100 if abs(min_sim - min_real) > allowed_range or abs(max_sim - max_real) > allowed_range else 0

        global_min = min(real_offset, sim_offset)
        global_max = max(max_real, max_sim)
        L = global_max - global_min + 1
        r = np.zeros(L)
        s = np.zeros(L)
        r_start = real_offset - global_min
        s_start = sim_offset - global_min
        r[r_start:r_start + len(real_hist)] = real_hist
        s[s_start:s_start + len(sim_hist)] = sim_hist

        diffs = np.abs(r - s)
        tolerance = round(n_nodes * 0.05)
        penalty = diffs - tolerance
        penalty[penalty < 0] = 0

        total_penalty += np.sum(penalty) + penalty_minmax

    return total_penalty + penalty_trans_num


def distribution_distance_dynamic_degree(real_deg_hist, sim_deg_hist):
    real_deg_hist = np.asarray(real_deg_hist, dtype=float)
    sim_deg_hist = np.asarray(sim_deg_hist, dtype=float)
    n_nodes = real_deg_hist.sum()
    if n_nodes <= 0:
        return float("inf")

    max_k_real = len(real_deg_hist) - 1
    max_k_sim = len(sim_deg_hist) - 1
    allowed_range = int(round(max_k_real * 0.2))
    penalty_maxdeg = abs(max_k_sim - max_k_real) * 100 if abs(max_k_sim - max_k_real) > allowed_range else 0

    L = max(len(real_deg_hist), len(sim_deg_hist))
    real_deg_hist = np.pad(real_deg_hist, (0, L - len(real_deg_hist)))
    sim_deg_hist = np.pad(sim_deg_hist, (0, L - len(sim_deg_hist)))

    diffs = np.abs(real_deg_hist - sim_deg_hist)
    tolerance = int(round(n_nodes * 0.05))
    penalty = diffs - tolerance
    penalty[penalty < 0] = 0

    return np.sum(penalty) + penalty_maxdeg

