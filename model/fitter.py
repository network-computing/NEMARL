import os
import time
import csv

from model.Energy import SwEnergy, DegreeEnergy, DynamicLuqEnergy, DynamicDegreeEnergy, DynamicLcc_Eff
from model.fit_utils import *
from argument import DegreeFitConfig, SWFitConfig, DynamicLuqConfig, DynamicDegreeConfig, DynamicLccEffConfig



class Fitter:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir


class SwFitter(Fitter):
    def __init__(self, output_file_name):
        super().__init__(SWFitConfig())
        self.output_file_name = output_file_name
        self.BOUNDS = ((0, 30), (0, 15), (0.00, 0.20), (0.80, 0.98))
        self.folders = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        self.folders.sort()
    
    def pick_initial_params(self, degree_hist, Clustering):
        C = Clustering
        k_max = len(degree_hist) - 1

        if C <= 0.10:
            add_list = max(1, round(0.17 * k_max))
            dlt_list = max(1, add_list // 2)
            add_thresh = 0.10
            dlt_thresh = 0.87
        elif C <= 0.20:
            add_list = max(1, round(0.5 * k_max))
            dlt_list = max(1, add_list // 2)
            add_thresh = 0.06
            dlt_thresh = 0.91
        else:
            add_list = max(1, round(0.83 * k_max))
            dlt_list = max(1, add_list // 2)
            add_thresh = 0.02
            dlt_thresh = 0.95

        return (add_list, dlt_list, add_thresh, dlt_thresh)

    def fit(self, eval_args, eval_func):
        for folder in self.folders:
            folder_path = os.path.join(self.data_dir, folder)
            data_file = find_data_file(folder_path)
            if data_file is None:
                print(f"[WARN] Skipped: {folder} (No .mtx or .edges files found)")
                continue

            print(f"\n===== dataset {folder} =====")
            print(f"[LOADING] {data_file}")

            try:
                real_info = load_real_graph(data_file)
            except Exception as e:
                print(f"[Error] Failed to load dataset: {folder} -> {e}")
                continue

            print(f"[REAL] N={real_info['N']}, E={real_info['E']}, C={real_info['C']:.4f}, L={real_info['L']:.4f}")

            p0 = self.pick_initial_params(real_info["deg_hist"], real_info['C'])
            starts = [p0]
            energy_fn = SwEnergy(real_info, eval_args, eval_func, base_seed=2025, repeats=1)

            global_best_p, global_best_E = None, float("inf")

            all_start_results = []
            for si, p0 in enumerate(starts):
                name = f"start{si+1}"
                print(f"[SA] Start {si + 1}/{len(starts)}: {p0}")

                try:
                    bp, bE, _ = SA_once(p0, energy_fn,
                                        rng_seed=2025 + 97 * si,
                                        alpha=0.95, L=80, Tmin_factor=1e-3, max_layers=60,
                                        bounds=self.BOUNDS)
                except Exception as e:
                    print(f"[ERROR] Error occurred during SA process: {folder}, start {si+1} -> {e}")
                    continue

                key = (int(bp[0]), int(bp[1]), float(bp[2]), float(bp[3]))
                ec = energy_fn.cache[key]["E_C"]
                el = energy_fn.cache[key]["E_L"]
                c_sim = energy_fn.cache[key]["C_sim"]
                l_sim = energy_fn.cache[key]["L_sim"]

                all_start_results.append((name, bp, bE, ec, el, c_sim, l_sim))

                if bE < global_best_E:
                    global_best_p, global_best_E = bp, bE

            if global_best_p is None:
                print(f"[WARN] The dataset {folder} did not produce valid results; writing was skipped")
                continue

            best_key = (int(global_best_p[0]), int(global_best_p[1]), float(global_best_p[2]), float(global_best_p[3]))
            ec_best = energy_fn.cache[best_key]["E_C"]
            el_best = energy_fn.cache[best_key]["E_L"]
            c_sim_best = energy_fn.cache[best_key]["C_sim"]
            l_sim_best = energy_fn.cache[best_key]["L_sim"]

            print(f"[Completed] {folder} Optimal parameters={global_best_p}")

            summary_path = os.path.join(self.config.summary_dir, self.output_file_name)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
                with open(summary_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "dataset_folder", "data_file", "N", "E",
                        "C_real",
                        "L_real",
                        "best_params(add_l,dlt_l,add_t,dlt_t)",
                        "best_energy",
                        "E_C",
                        "E_L",
                        "C_sim_best",
                        "L_sim_best"
                    ])

            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    folder,
                    os.path.basename(data_file), real_info["N"], real_info["E"],
                    f"{real_info['C']:.6f}",
                    f"{real_info['L']:.6f}",
                    str(global_best_p),
                    f"{global_best_E:.6f}",
                    f"{ec_best:.6f}",
                    f"{el_best:.6f}",
                    f"{c_sim_best:.6f}",
                    f"{l_sim_best:.6f}"
                ])


class DegreeFitter(Fitter):
    def __init__(self, output_file_name):
        super().__init__(DegreeFitConfig())
        self.output_file_name = output_file_name
        self.folders = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        self.folders.sort()

    def pick_initial_params(self, degree_hist):
        J = jsd_poisson_from_hist(degree_hist)
        k_max = len(degree_hist) - 1

        if J <= 0.10:  # Poisson
            add_list = max(1, round(0.17 * k_max))
            dlt_list = max(1, add_list // 2)
            add_thresh = 0.02
            dlt_thresh = 0.95
        elif J <= 0.20:  # Between Poisson and power-law
            add_list = max(1, round(0.5 * k_max))
            dlt_list = max(1, add_list // 2)
            add_thresh = 0.06
            dlt_thresh = 0.91
        else:  # Power Law
            add_list = max(1, round(0.83 * k_max))
            dlt_list = max(1, add_list // 2)
            add_thresh = 0.10
            dlt_thresh = 0.87

        return (add_list, dlt_list, add_thresh, dlt_thresh)

    def fit(self, eval_args, eval_func):
        for folder in self.folders:
            folder_path = os.path.join(self.data_dir, folder)
            data_file = find_data_file(folder_path)
            if data_file is None:
                print(f"[WARN] Skipped: {folder} (No .mtx or .edges files found)")
                continue

            print(f"\n===== dataset {folder} =====")
            print(f"[LOADING] {data_file}")

            try:
                real_info = load_real_graph(data_file)
            except Exception as e:
                print(f"[ERROR] Failed to read the dataset: {folder} -> {e}")
                continue

            print(f"[REAL] N={real_info['N']}, E={real_info['E']}")
            max_deg = max(dict(real_info["G"].degree()).values())

            BOUNDS = ((0, max_deg), (0, max_deg // 2), (0.00, 0.20), (0.85, 0.98))

            p0 = self.pick_initial_params(real_info["deg_hist"])
            starts = [p0]
            print(f"[Init] Dataset {folder} -> Initial parameters: {p0}")

            energy_fn = DegreeEnergy(real_info, eval_args, eval_func, base_seed=2025, repeats=1)

            global_best_p, global_best_E, global_best_deg = None, float("inf"), None
            all_best = []

            for si, p0 in enumerate(starts):
                print(f"[SA] Start {si + 1}/{len(starts)}: {p0}")
                try:
                    bp, bE, bdeg = SA_once(p0, energy_fn,
                                            rng_seed=2025 + 97 * si,
                                            alpha=0.95, L=80, Tmin_factor=1e-3, max_layers=60,
                                            bounds=BOUNDS)
                except Exception as e:
                    print(f"[ERROR] Error occurred during SA process: {folder}, 起点{si + 1} -> {e}")
                    continue

                all_best.append((f"start{si + 1}", bp, bE, bdeg))
                if bE < global_best_E:
                    global_best_p, global_best_E, global_best_deg = bp, bE, bdeg

            if global_best_p is None:
                print(f"[WARN] The dataset {folder} did not produce valid results; writing was skipped")
                continue

            cache_key = (int(global_best_p[0]), int(global_best_p[1]), float(global_best_p[2]), float(global_best_p[3]))
            e_deg_norm = energy_fn.cache[cache_key]["E_deg_norm"]

            print(f"[Completed] {folder} Optimal parameters={global_best_p}")

            real_deg_str = str(real_info["deg_hist"])
            fitted_deg_str = str(global_best_deg)

            summary_path = os.path.join(self.config.summary_dir, self.output_file_name)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
                with open(summary_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "dataset_folder", 
                        "data_file", 
                        "N", 
                        "E",
                        "best_params(add_l,dlt_l,add_t,dlt_t)",
                        "best_energy",
                        "E_deg_norm",
                        "real_degree_distribution",
                        "fitted_degree_distribution"
                    ])

            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    folder, 
                    os.path.basename(data_file), 
                    real_info["N"], 
                    real_info["E"],
                    str(global_best_p), 
                    f"{global_best_E:.6f}",
                    f"{e_deg_norm:.6f}",
                    real_deg_str,
                    fitted_deg_str
                ])


class DynamicLuqFitter(Fitter):
    def __init__(self, output_file_name):
        super().__init__(DynamicLuqConfig())
        self.output_file_name = output_file_name
        self.BOUNDS = ((0, 30), (0, 15), (0.00, 0.20), (0.80, 0.98))

    def fit(self, eval_args, eval_func):
        folder = os.path.basename(self.data_dir)  # 如 "PR_1955-1962"
        data_files = find_dynamic_gexf_files(self.data_dir)

        if data_files is None:
            print(f"[WARN] Skipped: {folder} (No .gexf file found)")
            exit()
        
        print(f"\n===== dataset {folder} =====")
        print(f"[LOADING] {data_files}")

        try:
            real_info = load_dyanamic_real_graphs(data_files)
        except Exception as e:
            print(f"[ERROR] Failed to read dataset: {folder} -> {e}")
            exit()

        print(f"[REAL] N_initial={real_info['N_initial']}, E_initial={real_info['E_initial']}, Num_snapshots={len(real_info['Gs'])}")

        initial_graph_path = data_files[0]

        energy_fn = DynamicLuqEnergy(real_info, eval_args, eval_func, initial_graph_path,
                                     base_seed=2025, repeats=1)

        global_best_p, global_best_E, global_best_diffs = None, float("inf"), None
        all_best = []
        t0 = time.time()
        starts = [(5, 1, 0, 0.98), (8, 3, 0.01, 0.95)]
        for si, p0 in enumerate(starts):
            print(f"[SA] start {si + 1}/{len(starts)}: {p0}")

            try:
                bp, bE, bdiffs = SA_once(p0, energy_fn,
                                        rng_seed=2025 + 97 * si,
                                        alpha=0.95, L=80, Tmin_factor=1e-3, max_layers=60,
                                        bounds=self.BOUNDS)
            except Exception as e:
                print(f"[ERROR] Error during SA process: {folder}, starting from {si+1} -> {e}")
                continue

            all_best.append((f"start{si+1}", bp, bE, bdiffs))
            if bE < global_best_E:
                global_best_p, global_best_E, global_best_diffs = bp, bE, bdiffs

        if global_best_p is None:
            print(f"[WARN] The dataset {folder} did not produce valid results; writing was skipped")
            exit()

        cache_key = (int(global_best_p[0]), int(global_best_p[1]), float(global_best_p[2]), float(global_best_p[3]))
        e_diff_norm = energy_fn.cache[cache_key]["E_diff_norm"]

        print(f"[Completed] {folder} Optimal parameters={global_best_p}")

        summary_path = os.path.join(self.config.summary_dir, self.output_file_name)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "dataset_folder", 
                    "data_files", 
                    "N_initial", 
                    "E_initial",
                    "best_params(add_l,dlt_l,add_t,dlt_t)",
                    "best_energy",
                    "E_diff_norm",
                ])

        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                folder, 
                str([os.path.basename(f) for f in data_files]), 
                real_info["N_initial"], 
                real_info["E_initial"],
                str(global_best_p), 
                f"{global_best_E:.6f}",
                f"{e_diff_norm:.6f}",
            ])


class DynamicDegreeFitter(Fitter):
    def __init__(self, output_file_name):
        super().__init__(DynamicDegreeConfig())
        self.output_file_name = output_file_name
        self.txt_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        self.txt_files.sort()

    def pick_initial_params(self, degree_hist):
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

    def fit(self, eval_args, eval_func):
        for txt_file in self.txt_files:
            dataset_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(self.data_dir, txt_file)
            time_series = load_time_series_from_txt(txt_path)
            if "Time1" not in time_series or "Time2" not in time_series:
                continue
            
            deg1 = time_series["Time1"]
            deg2 = time_series["Time2"]
            N1, N2 = int(np.sum(deg1)), int(np.sum(deg2))

            max_deg1 = len(deg1) - 1
            BOUNDS1 = (
                (1, max_deg1),          # add_list
                (1, max_deg1 // 2),     # dlt_list
                (0.00, 0.20),           # add_thresh
                (0.85, 0.98),           # dlt_thresh
            )
            
            print(f"\n===== dataset {dataset_name}: Time1 annealed fit =====")
            print(f"[Time1] N={N1}, BOUNDS1={BOUNDS1}")

            energy_t1 = DynamicDegreeEnergy({"deg_hist": deg1, "N": N1}, eval_args, eval_func)
            p0_1 = self.pick_initial_params(deg1)

            best_p1, best_E1, best_deg1 = SA_once(p0_1, energy_t1, bounds=BOUNDS1)
            key1 = (int(best_p1[0]), int(best_p1[1]), float(best_p1[2]), float(best_p1[3]))
            e1_norm = energy_t1.cache[key1]["E_deg_norm"]

            print(f"[Time1] best_params={best_p1}, best_E={best_E1:.6f}, E_norm={e1_norm:.6f}")

            print(f"\n===== dataset {dataset_name}: Time2 neighborhood annealing fitting =====")

            energy_t2 = DynamicDegreeEnergy({"deg_hist": deg2, "N": N2}, eval_args, eval_func, base_seed=2026)
            add1, dlt1, add_t1, dlt_t1 = best_p1

            BOUNDS2 = (
                (max(1, add1 - 2),          add1 + 2),                   # add_list ±2
                (max(1, dlt1 - 1),          dlt1 + 1),                   # dlt_list ±1
                (max(0.0, add_t1 - 0.2),    min(1.0, add_t1 + 0.2)),     # add_thresh ±0.2
                (max(0.0, dlt_t1 - 0.2),    min(1.0, dlt_t1 + 0.2)),     # dlt_thresh ±0.2
            )
            print(f"[Time2] N={N2}, BOUNDS1={BOUNDS2}")

            p0_2 = clip_params(best_p1, BOUNDS2)

            best_p2, best_E2, best_deg2 = SA_once(p0_2, energy_t2, rng_seed=321, bounds=BOUNDS2)
            key2 = (int(best_p2[0]), int(best_p2[1]), float(best_p2[2]), float(best_p2[3]))
            e2_norm = energy_t2.cache[key2]["E_deg_norm"]

            print(f"[Time2] best_params={best_p2}, best_E={best_E2:.6f}, E_norm={e2_norm:.6f}")

            summary_path = os.path.join(self.config.summary_dir, self.output_file_name)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
                with open(summary_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "dataset",
                        "time_label",
                        "N",
                        "best_params(add_l,dlt_l,add_t,dlt_t)",
                        "best_energy",
                        "E_deg_norm",
                        "real_degree_distribution",
                        "fitted_degree_distribution"
                    ])

            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name,
                    "Time1",
                    N1,
                    str(best_p1),
                    f"{best_E1:.6f}",
                    f"{e1_norm:.6f}",
                    str(deg1),
                    str(best_deg1)
                ])
                writer.writerow([
                    dataset_name,
                    "Time2",
                    N2,
                    str(best_p2),
                    f"{best_E2:.6f}",
                    f"{e2_norm:.6f}",
                    str(deg2),
                    str(best_deg2)
                ])

