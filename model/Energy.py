import numpy as np
from abc import ABC, abstractmethod

from model.fit_utils import distribution_distance, distribution_distance_dynamic_luq, distribution_distance_dynamic_degree


class Energy(ABC):
    def __init__(self, eval_args, eval_func, base_seed=2025, repeats=1):
        self.cache = {}
        self.seeds = [base_seed + i * 7919 for i in range(repeats)]

        self.eval_args = eval_args
        self.eval_func = eval_func
        assert callable(self.eval_func)
    
    @abstractmethod
    def evaluate_single_run(self, params, seed):
        """Single-assessment implementation"""
        pass

    @abstractmethod
    def compute_energy(self, params):
        """Calculate the total energy value"""
        pass
    

    def __call__(self, params):
        return self.compute_energy(params)
    

class DegreeEnergy(Energy):
    def __init__(self, real_info, eval_args, eval_func, base_seed=2025, repeats=1):
        super().__init__(eval_args, eval_func, base_seed, repeats)
        self.real_deg = real_info["deg_hist"]
        self.real_N = real_info["N"]

    def evaluate_single_run(self, params, seed):
        add_l, dlt_l, add_t, dlt_t = params

        self.eval_args.add_list = int(add_l)
        self.eval_args.dlt_list = int(dlt_l)
        self.eval_args.add_thresh = float(add_t)
        self.eval_args.dlt_thresh = float(dlt_t)
        self.eval_args.num_of_node = self.real_N
        if hasattr(self.eval_args, "seed"):
            self.eval_args.seed = int(seed)

        result = self.eval_func(self.eval_args, train_mode='degree')
        deg_sim = result.get("degree_distribution")
        if deg_sim is None:
            raise RuntimeError("eval_func() must return result[‘degree_distribution’]")

        E_deg = distribution_distance(self.real_deg, deg_sim)
        E_deg_norm = E_deg / max(1.0, float(np.sum(self.real_deg)))

        return E_deg_norm, deg_sim

    def compute_energy(self, params):
        key = (int(params[0]), int(params[1]), float(params[2]), float(params[3]))
        if key in self.cache:
            v = self.cache[key]
            return v["E_deg_norm"], v["deg"]
        e_deg_norm, deg1 = self.evaluate_single_run(key, self.seeds[0])
        self.cache[key] = {"deg": deg1, "E_deg_norm": e_deg_norm}
        return e_deg_norm, deg1


class SwEnergy(Energy):
    def __init__(self, real_info, eval_args, eval_func, base_seed=2025, repeats=1, wC=1.0, wL=1.0):
        super().__init__(eval_args, eval_func, base_seed, repeats)
        self.real_deg = real_info["deg_hist"]
        self.C_real = real_info["C"]
        self.L_real = real_info["L"]
        self.real_N = real_info["N"]
        self.wC = wC
        self.wL = wL

    def evaluate_single_run(self, params, seed):
        add_l, dlt_l, add_t, dlt_t = params

        self.eval_args.add_list   = int(add_l)
        self.eval_args.dlt_list   = int(dlt_l)
        self.eval_args.add_thresh = float(add_t)
        self.eval_args.dlt_thresh = float(dlt_t)
        self.eval_args.num_of_node = self.real_N
        if hasattr(self.eval_args, "seed"):
            self.eval_args.seed = int(seed)

        result = self.eval_func(self.eval_args, train_mode='sw')

        C_sim = result.get("clustering")
        L_sim = result.get("avg_path_length")
        if C_sim is None or L_sim is None or L_sim <= 0:
            raise RuntimeError("eval_func() must return clustering and avg_path_length.")

        E_C = abs(C_sim - self.C_real) / max(self.C_real, 1e-12)
        E_L = abs(L_sim - self.L_real) / max(self.L_real, 1e-12)

        energy = self.wC * E_C + self.wL * E_L
        return energy, (E_C, E_L, C_sim, L_sim)
    
    def compute_energy(self, params):
        key = (int(params[0]), int(params[1]), float(params[2]), float(params[3]))
        if key in self.cache:
            v = self.cache[key]
            return v["E"], None
        e1, (E_C, E_L, C_sim, L_sim) = self.evaluate_single_run(key, self.seeds[0])
        self.cache[key] = {"E": e1, "E_C": E_C, "E_L": E_L, "C_sim": C_sim, "L_sim": L_sim}
        return e1, None
    

class DynamicLuqEnergy(Energy):
    def __init__(self, real_info, eval_args, eval_func, initial_graph_path, 
                 base_seed=2025, repeats=1):
        super().__init__(eval_args, eval_func, base_seed, repeats)
        self.real_diffs = real_info["deg_diffs"]
        self.initial_graph_path = initial_graph_path

    def evaluate_single_run(self, params, seed):
        add_l, dlt_l, add_t, dlt_t = params

        self.eval_args.add_list   = int(add_l)
        self.eval_args.dlt_list   = int(dlt_l)
        self.eval_args.add_thresh = float(add_t)
        self.eval_args.dlt_thresh = float(dlt_t)
        self.eval_args.graph_path = self.initial_graph_path
        if hasattr(self.eval_args, "seed"):
            self.eval_args.seed = int(seed)

        result = self.eval_func(self.eval_args, train_mode='dynamic_luq')

        diff_sim = result.get("degree_diffs")
        if diff_sim is None:
            raise RuntimeError("eval_func() must return result['degree_diffs']")

        E_diff = distribution_distance_dynamic_luq(self.real_diffs, diff_sim)
        total_n = sum([hist.sum() for hist, _ in self.real_diffs])
        E_diff_norm = E_diff / max(1.0, float(total_n))

        return E_diff_norm, diff_sim
    
    def compute_energy(self, params):
        key = (int(params[0]), int(params[1]), float(params[2]), float(params[3]))
        if key in self.cache:
            v = self.cache[key]
            return v["E_diff_norm"], v["diffs"]
        e_diff_norm, diffs1 = self.evaluate_single_run(key, self.seeds[0])
        self.cache[key] = {"diffs": diffs1, "E_diff_norm": e_diff_norm}
        return e_diff_norm, diffs1


class DynamicDegreeEnergy(Energy):
    def __init__(self, real_info, eval_args, eval_func, 
                 base_seed=2025, repeats=1):
        super().__init__(eval_args, eval_func, base_seed, repeats)
        self.real_deg = real_info["deg_hist"]
        self.real_N = real_info["N"]

    def evaluate_single_run(self, params, seed):
        add_l, dlt_l, add_t, dlt_t = params

        self.eval_args.add_list   = int(add_l)
        self.eval_args.dlt_list   = int(dlt_l)
        self.eval_args.add_thresh = float(add_t)
        self.eval_args.dlt_thresh = float(dlt_t)
        self.eval_args.num_of_node = self.real_N
        if hasattr(self.eval_args, "seed"):
            self.eval_args.seed = int(seed)

        result = self.eval_func(self.eval_args, train_mode='dynamic_degree')

        deg_sim = result.get("degree_distribution")
        if deg_sim is None:
            raise RuntimeError("eval_func() must return result['degree_distribution']")

        E_deg = distribution_distance_dynamic_degree(self.real_deg, deg_sim)
        E_deg_norm = E_deg / max(1.0, float(np.sum(self.real_deg)))

        return E_deg_norm, deg_sim
    
    def compute_energy(self, params):
        key = (int(params[0]), int(params[1]), float(params[2]), float(params[3]))
        if key in self.cache:
            v = self.cache[key]
            return v["E_deg_norm"], v["deg"]
        e_diff_norm, diffs1 = self.evaluate_single_run(key, self.seeds[0])
        self.cache[key] = {"deg": diffs1, "E_deg_norm": e_diff_norm}
        return e_diff_norm, diffs1