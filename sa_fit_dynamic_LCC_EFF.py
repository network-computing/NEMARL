import os
import time
import csv
import numpy as np
from math import log, exp
import networkx as nx
import arguments
from scipy.stats import poisson
from train import run


# ==================================================
# Output directory
# ==================================================
base_dir = "fig/parameters/fit_dyn_lcc_eff"
os.makedirs(base_dir, exist_ok=True)

SAVE_CSV = True
summary_path = os.path.join(base_dir, "summary.csv")
if SAVE_CSV and (not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0):
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "snapshots_dir",
            "num_nodes",
            "target_len",
            "best_params(add_l,dlt_l,add_t,dlt_t)",
            "sim_series",
            "real_series",
        ])


# ==================================================
# Dataset configuration
# ==================================================
DATASETS = [
    dict(
        name="contact-primary-school",
        snapshots_dir=r"fig/verification2/contact-primary-school/processed_snapshots_k3",
        target_len=3,
    ),
    dict(
        name="tij_InVS",
        snapshots_dir=r"fig/verification2/tij_InVS/processed_snapshots_k4",
        target_len=4,
    ),
    dict(
        name="detailed_list_of_contacts_Hospital",
        snapshots_dir=r"fig/verification2/detailed_list_of_contacts_Hospital/processed_snapshots_k4",
        target_len=4,
    ),
    dict(
        name="ht09_contact_list",
        snapshots_dir=r"fig/verification2/ht09_contact_list/processed_snapshots_k4",
        target_len=4,
    ),
]


# ==================================================
# Utilities
# ==================================================
def rel_err(a, b, eps=1e-12):
    return abs(a - b) / max(eps, abs(b))


def load_real_series_from_metrics(metrics_csv: str, target_len: int):
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(metrics_csv)

    series = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series.append({
                "S_LCC": float(row["S_LCC"]),
                "E_glob": float(row["E_glob"]),
            })
            if len(series) >= target_len:
                break

    if len(series) < target_len:
        raise RuntimeError("metrics length < target_len")

    return series

def G0_from(snapshots_dir: str):
    p0 = os.path.join(snapshots_dir, "snapshot_0.gexf")
    if not os.path.exists(p0):
        raise FileNotFoundError(p0)
    G0 = nx.read_gexf(p0)
    return G0

def infer_num_nodes_from_gexf(snapshots_dir: str):
    p0 = os.path.join(snapshots_dir, "snapshot_0.gexf")
    if not os.path.exists(p0):
        raise FileNotFoundError(p0)
    G0 = nx.read_gexf(p0)
    return G0.number_of_nodes()


# ==================================================
# Energy function (objective for SA)
# ==================================================
class Energy:
    def __init__(self, dataset_cfg, real_series, graph_dir,
                 base_seed=2025, verbose=1, print_cache_hit=False):
        self.cache = {}
        self.seed = base_seed
        self.verbose = verbose
        self.print_cache_hit = print_cache_hit
        self.eval_count = 0

        self.dataset = dataset_cfg
        self.real_series = real_series
        self.real_lcc = [x["S_LCC"] for x in real_series]
        self.real_eff = [x["E_glob"] for x in real_series]
        self.target_len = len(real_series)

        self.graph_dir = graph_dir

    def _eval_once(self, params, exp_tag):
        add_l, dlt_l, add_t, dlt_t = params

        args = arguments.get_args()
        args.add_list = int(add_l)
        args.dlt_list = int(dlt_l)
        args.add_thresh = float(add_t)
        args.dlt_thresh = float(dlt_t)

        args.snapshots_dir = self.dataset["snapshots_dir"]
        args.reset_snapshot_index = 0
        args.step = self.target_len - 1
        args.episode = 1

        args.num_of_node = int(self.dataset["num_nodes"])
        args.output_dir = self.graph_dir
        args.train_verbose = False

        if hasattr(args, "seed"):
            args.seed = int(self.seed)

        t0 = time.time()
        result = run(args, exp_tag)
        dt = time.time() - t0

        sim_lcc = result["S_LCC_series"][: self.target_len]
        sim_eff = result["E_glob_series"][: self.target_len]

        E_lcc = [rel_err(sim_lcc[i], self.real_lcc[i]) for i in range(self.target_len)]
        E_eff = [rel_err(sim_eff[i], self.real_eff[i]) for i in range(self.target_len)]

        E_lcc_mean = float(np.mean(E_lcc))
        E_eff_mean = float(np.mean(E_eff))

        energy = 0.5 * E_lcc_mean + 0.5 * E_eff_mean

        sim_series = [
            {"S_LCC": float(sim_lcc[i]), "E_glob": float(sim_eff[i])}
            for i in range(self.target_len)
        ]

        return energy, E_lcc_mean, E_eff_mean, sim_series, dt

    def __call__(self, params):
        key = (int(params[0]), int(params[1]), float(params[2]), float(params[3]))

        if key in self.cache:
            return self.cache[key][:4]

        self.eval_count += 1
        energy, E_lcc, E_eff, sim_series, dt = self._eval_once(
            key, exp_tag=f"{self.dataset['name']}_SA_{key}"
        )
        self.cache[key] = (energy, E_lcc, E_eff, sim_series, dt)
        return energy, E_lcc, E_eff, sim_series


# ==================================================
# Simulated Annealing
# ==================================================
def clip_params(p, bounds):
    return (
        int(np.clip(p[0], *bounds[0])),
        int(np.clip(p[1], *bounds[1])),
        round(float(np.clip(p[2], *bounds[2])), 3),
        round(float(np.clip(p[3], *bounds[3])), 3),
    )


def propose(p, rng, bounds, step_int=1, step_thr=0.02, big_prob=0.1):
    add_l, dlt_l, add_t, dlt_t = p
    i = rng.integers(0, 4)
    big = rng.random() < big_prob

    if i == 0:
        add_l += (3 if big else step_int) * (1 if rng.random() < 0.5 else -1)
    elif i == 1:
        dlt_l += (3 if big else step_int) * (1 if rng.random() < 0.5 else -1)
    elif i == 2:
        add_t += (0.06 if big else step_thr) * (1 if rng.random() < 0.5 else -1)
    else:
        dlt_t += (0.06 if big else step_thr) * (1 if rng.random() < 0.5 else -1)

    return clip_params((add_l, dlt_l, add_t, dlt_t), bounds)


def calibrate_T0(p0, energy_fn, rng, bounds, tries=15, target_accept=0.8):
    E0, _, _, _ = energy_fn(p0)
    deltas = []

    for _ in range(tries):
        q = propose(p0, rng, bounds)
        Eq, _, _, _ = energy_fn(q)
        if Eq - E0 > 0:
            deltas.append(Eq - E0)

    dpos = np.mean(deltas) if deltas else 1.0
    return max(-dpos / log(target_accept), 1e-6)


def SA_once(
    p0, energy_fn, bounds,
    rng_seed=2025, alpha=0.95, L=40,
    Tmin_factor=1e-3, max_layers=40,
):
    rng = np.random.default_rng(rng_seed)
    p = clip_params(p0, bounds)
    T = calibrate_T0(p, energy_fn, rng, bounds)
    Tmin = max(T * Tmin_factor, 1e-8)

    E, E_lcc, E_eff, sim_series = energy_fn(p)
    best_p, best_E = p, E
    best_detail = (E_lcc, E_eff, sim_series)

    step_int, step_thr = 1, 0.02
    noimprove, layer = 0, 0

    while T > Tmin and layer < max_layers:
        for _ in range(L):
            q = propose(p, rng, bounds, step_int, step_thr)
            Eq, q_lcc, q_eff, q_series = energy_fn(q)
            dE = Eq - E

            if dE <= 0 or rng.random() < exp(-dE / T):
                p, E = q, Eq
                if E < best_E:
                    best_p, best_E = p, E
                    best_detail = (q_lcc, q_eff, q_series)
                    noimprove = 0

        T *= alpha
        layer += 1
        noimprove += 1
        if noimprove >= 6:
            break

    return best_p, best_E, best_detail
def jsd_poisson_from_hist(hist, eps=1e-12):
    hist = np.asarray(hist, dtype=float)
    n = hist.sum()
    if n <= 0:
        return np.nan
    ks = np.arange(len(hist))
    lam = (ks * hist).sum() / n
    p = hist / n
    q = poisson.pmf(ks, lam)
    p = np.clip(p, eps, 1); p /= p.sum()
    q = np.clip(q, eps, 1); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
def pick_initial_params(degree_hist):
    J = jsd_poisson_from_hist(degree_hist)
    k_max = len(degree_hist) - 1

    if J <= 0.10:
        add_list = max(1, round(0.17 * k_max))
        dlt_list = max(1, add_list // 2)
        add_thresh = 0.02
        dlt_thresh = 0.95
    elif J <= 0.20:
        add_list = max(1, round(0.5 * k_max))
        dlt_list = max(1, add_list // 2)
        add_thresh = 0.06
        dlt_thresh = 0.91
    else:
        add_list = max(1, round(0.83 * k_max))
        dlt_list = max(1, add_list // 2)
        add_thresh = 0.10
        dlt_thresh = 0.87

    return (add_list, dlt_list, add_thresh, dlt_thresh)
# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    for ds in DATASETS:
        name = ds["name"]
        snapshots_dir = ds["snapshots_dir"]
        target_len = int(ds["target_len"])

        metrics_csv = os.path.join(snapshots_dir, "metrics.csv")
        real_series = load_real_series_from_metrics(metrics_csv, target_len)

        if int(ds["num_nodes"]) <= 0:
            ds["num_nodes"] = infer_num_nodes_from_gexf(snapshots_dir)

        graph_dir = os.path.join(base_dir, name, "graphs")
        os.makedirs(graph_dir, exist_ok=True)

        energy_fn = Energy(ds, real_series, graph_dir)

        t0 = time.time()
        max_deg = max(dict(G0_from(snapshots_dir).degree()).values())
        BOUNDS = (
            (1, max_deg),
            (1, max(1, max_deg // 2)),
            (0.00, 0.20),
            (0.85, 0.98),
        )

        p0 = pick_initial_params(G0_from(snapshots_dir).degree_hist())
        best_p, best_E, (best_lcc_err, best_eff_err, best_series) = SA_once(
            p0, energy_fn, BOUNDS
        )
        elapsed = time.time() - t0

        if SAVE_CSV:
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    snapshots_dir,
                    int(ds["num_nodes"]),
                    target_len,
                    str(best_p),
                    str(best_series),
                    str(real_series),
                ])
