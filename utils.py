import os
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
import torch


def is_lower_threshold(val, threshold):
    return val < threshold


def is_upper_threshold(val, threshold):
    return val > threshold


# Comparison function map
compare_func_map = {
    "is_lower_threshold": is_lower_threshold,
    "is_upper_threshold": is_upper_threshold,
    "lower": is_lower_threshold,
    "upper": is_upper_threshold,
}


def get_compare_function(func_name: str):
    if func_name in compare_func_map:
        return compare_func_map[func_name]
    else:
        raise ValueError(f"Unknown comparison function: {func_name}")


def get_bo_sum(pr, graph, node_id, r):
    one_ego_graph = nx.ego_graph(graph, node_id, r)
    nodes = list(nx.nodes(one_ego_graph))
    bo = [pr[int(j)] for j in nodes]
    return sum(bo)


def get_reward_by_pg(**kwargs):
    cur_graph = kwargs["cur_graph"]
    node_num = cur_graph.number_of_nodes()
    cur_pr = ig.Graph.from_networkx(cur_graph).personalized_pagerank()

    next_graph = kwargs["next_graph"]
    next_pr = ig.Graph.from_networkx(next_graph).personalized_pagerank()

    add_rewards = {}
    dlt_rewards = {}
    for i in range(node_num):
        cur_add_bo_sum = get_bo_sum(cur_pr, cur_graph, i, 1)
        cur_dlt_bo_sum = 1 - cur_add_bo_sum

        next_add_bo_sum = get_bo_sum(next_pr, next_graph, i, 1)
        next_dlt_bo_sum = 1 - next_add_bo_sum

        add_rewards[i] = next_add_bo_sum - cur_add_bo_sum
        dlt_rewards[i] = next_dlt_bo_sum - cur_dlt_bo_sum

    return add_rewards, dlt_rewards


def get_reward_by_irl(**kwargs):
    reward_model = kwargs["reward_model"]
    cur_embs = kwargs["cur_embs"]
    next_embs = kwargs["next_embs"]

    add_rewards = {}
    dlt_rewards = {}
    for i in range(len(cur_embs)):
        add_s = torch.as_tensor(cur_embs[i], dtype=torch.float32, device=next(reward_model.parameters()).device)
        add_s_ = torch.as_tensor(next_embs[i], dtype=torch.float32, device=next(reward_model.parameters()).device)
        add_rewards[i] = dlt_rewards[i] = reward_model(add_s_).item() - reward_model(add_s).item()

    return add_rewards, dlt_rewards


def get_reward_degree_diff(**kwargs):
    def average_degree_of_neighbors(G, i):
        neighbors = list(G.neighbors(i))
        neighbor_degrees = [G.degree(neighbor) for neighbor in neighbors]
        avg_degree = sum(neighbor_degrees) / len(neighbor_degrees) if neighbors else 0
        return avg_degree

    def reward_based_on_degree_similarity(G, i, method="squared"):
        node_degree = G.degree(i)
        avg_neighbor_degree = average_degree_of_neighbors(G, i)

        if method == "absolute":
            reward = abs(node_degree - avg_neighbor_degree)
        elif method == "squared":
            reward = (node_degree - avg_neighbor_degree) ** 2
        elif method == "reciprocal":
            reward = 1 / (1 + abs(node_degree - avg_neighbor_degree))
        else:
            raise ValueError("Unknown method specified.")

        return reward
    cur_graph = kwargs["cur_graph"]
    next_graph = kwargs["next_graph"]
    node_num = cur_graph.number_of_nodes()
    add_rewards = {}
    dlt_rewards = {}

    for i in range(node_num):
        add_rewards[i] = -reward_based_on_degree_similarity(next_graph, i) + reward_based_on_degree_similarity(cur_graph, i)
        dlt_rewards[i] = reward_based_on_degree_similarity(next_graph, i) - reward_based_on_degree_similarity(cur_graph, i)

    return add_rewards, dlt_rewards


def get_reward_low_closenesee_centrality(**kwargs):
    def closenesee_centrality(G, i):
        shortest_paths = nx.shortest_path_length(G, source=i)
        avg_distance = 1 / sum(shortest_paths.values())
        return avg_distance
    
    cur_graph = kwargs["cur_graph"]
    next_graph = kwargs["next_graph"]
    node_num = cur_graph.number_of_nodes()
    add_rewards = {}
    dlt_rewards = {}

    for i in range(node_num):
        add_rewards[i] = -closenesee_centrality(next_graph, i) + closenesee_centrality(cur_graph, i)
        dlt_rewards[i] = closenesee_centrality(next_graph, i) - closenesee_centrality(cur_graph, i)

    return add_rewards, dlt_rewards


def get_reward_pg_degree(**kwargs):
    def calculate_normalized_degree(G):
        degrees = dict(G.degree())
        total_degree = sum(degrees.values())
        
        if total_degree == 0:
            return {node: 0 for node in G.nodes()}
        
        normalized = {node: deg / total_degree for node, deg in degrees.items()}
        return normalized
     
    cur_graph = kwargs["cur_graph"]
    next_graph = kwargs["next_graph"]
    node_num = cur_graph.number_of_nodes()

    cur_pr = ig.Graph.from_networkx(cur_graph).personalized_pagerank()
    cur_normalized = calculate_normalized_degree(cur_graph)
    next_pr = ig.Graph.from_networkx(next_graph).personalized_pagerank()
    next_normalized = calculate_normalized_degree(next_graph)

    add_rewards = {}
    dlt_rewards = {}
    for i in range(node_num):
        add_rewards[i] = next_pr[i] / next_normalized[i] - cur_pr[i] / cur_normalized[i]
        dlt_rewards[i] = -next_pr[i] / next_normalized[i] + cur_pr[i] / cur_normalized[i]

    return add_rewards, dlt_rewards


# reward function map
reward_func_map = {
    "pagerank": get_reward_by_pg,
    "pg": get_reward_by_pg,
    "irl": get_reward_by_irl,
    "similar_degree": get_reward_degree_diff,
    "low_cc": get_reward_low_closenesee_centrality,
    "pg_degree": get_reward_pg_degree,
}


def get_reward_function(func_name: str):
    if func_name in reward_func_map:
        return reward_func_map[func_name]
    else:
        raise ValueError(f"Unknown reward function: {func_name}")


def get_pg_compare_vals(**kwargs):
    graph = kwargs["graph"]

    node_num = graph.number_of_nodes()
    pr = ig.Graph.from_networkx(graph).personalized_pagerank()

    add_node_vals = {}
    dlt_node_vals = {}
    for i in range(node_num):
        cur_add_bo_sum = get_bo_sum(pr, graph, i, 1)
        cur_dlt_bo_sum = 1 - cur_add_bo_sum

        add_node_vals[i] = cur_add_bo_sum
        dlt_node_vals[i] = cur_dlt_bo_sum

    return add_node_vals, dlt_node_vals

def get_similar_degree_compare_vals(**kwargs):
    graph = kwargs["graph"]

    node_num = graph.number_of_nodes()

    add_node_vals = {}
    dlt_node_vals = {}
    for i in range(node_num):
        add_node_vals[i] = dlt_node_vals[i] = graph.degree(i)

    return add_node_vals, dlt_node_vals

def get_low_cc_compare_vals(**kwargs):
    graph = kwargs["graph"]

    node_num = graph.number_of_nodes()

    add_node_vals = {}
    dlt_node_vals = {}
    closeness_centrality = nx.closeness_centrality(graph)
    for i in range(node_num):
        add_node_vals[i] = dlt_node_vals[i] = closeness_centrality[i]

    return add_node_vals, dlt_node_vals


compare_vals_func_map = {
    "pagerank": get_pg_compare_vals,
    "pg": get_pg_compare_vals,
    "sd": get_similar_degree_compare_vals,
    "lcc": get_low_cc_compare_vals,
}


def get_compare_vals_function(func_name: str):
    if func_name in compare_vals_func_map:
        return compare_vals_func_map[func_name]
    else:
        raise ValueError(f"Unknown compare vals function: {func_name}")


def check_graph_node_id_from_zero(graph):
    """
    Check and normalize graph node IDs to ensure they start from 0 and are continuous.

    Description:
    - Retrieves and sorts all node IDs in the graph.
    - Checks whether the node IDs are consecutive integers (i.e., each difference equals 1).
      If not continuous, raises an assertion error.
    - If continuous, reindexes nodes so that the smallest node ID maps to 0,
      and other nodes follow sequentially.
    - Returns a new graph with relabeled node IDs.

    Args:
        graph (networkx.Graph): The input NetworkX graph.

    Returns:
        networkx.Graph: A new graph where node IDs start from 0 and are continuous.

    Raises:
        AssertionError: If the node IDs are not continuous.
    """
    nodes = sorted(graph.nodes())

    if len(nodes) > 1:
        diffs = np.diff(nodes)
        is_continuous = all(diff == 1 for diff in diffs)
    else:
        is_continuous = True
    
    assert is_continuous, "[Error] Graph node id is not continuous!"

    first_node_id = nodes[0]
    mapping = {node: int(node)-first_node_id for node in nodes}
    return nx.relabel_nodes(graph, mapping)


def average_shortest_path_length(G):
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
        L = nx.average_shortest_path_length(G_sub)
    
    return L


def compute_degree_diffs(graphs):
    """
    Input: A sequence of dynamic images [G1, G2, ..., Gt]
    Output: Difference histograms in the format [(hist_array, offset), ...]
        offset represents the minimum difference (alignment required when negative)
    """
    results = []
    for i in range(len(graphs)-1):
        g1, g2 = graphs[i], graphs[i+1]
        diff_count = {}
        all_nodes = set(g1.nodes()) | set(g2.nodes())
        for node in all_nodes:
            d1 = g1.degree(node) if node in g1 else 0
            d2 = g2.degree(node) if node in g2 else 0
            diff = d2 - d1
            diff_count[diff] = diff_count.get(diff, 0) + 1

        min_diff, max_diff = min(diff_count.keys()), max(diff_count.keys())
        hist = np.zeros(max_diff - min_diff + 1, dtype=int)
        for k, v in diff_count.items():
            hist[k - min_diff] = v

        results.append((hist, min_diff))
    return results


def load_expert_trajectories(dataset_dir, max_steps=10):
    items = os.listdir(dataset_dir)
    arrays = []
    for item in items:
        sub_dir = os.path.join(dataset_dir, item)
        if not os.path.isdir(sub_dir):
            continue
        time_trajectories = {}
        for time in range(max_steps):
            file_path = os.path.join(sub_dir, f"emb/emb_{time}.emb")

            if not os.path.exists(file_path):
                continue
            
            df = pd.read_csv(file_path, sep=' ',header=None)

            node_ids = df.iloc[:, 0].astype(int)
            features = df.iloc[:, 1:].astype(float)

            features.index = node_ids
            features.columns = [f"dim_{i}" for i in range(features.shape[1])]

            time_trajectories[time] = features
        all_nodes = sorted(set().union(*[df.index for df in time_trajectories.values()]))
        all_times = sorted(time_trajectories.keys())
        feature_dim = next(iter(time_trajectories.values())).shape[1]

        tensor_3d = np.full((len(all_nodes), len(all_times), feature_dim), np.nan)

        for time_idx, time in enumerate(all_times):
            df = time_trajectories[time]
            for node in df.index:
                if node in all_nodes:
                    node_idx = all_nodes.index(node)
                    tensor_3d[node_idx, time_idx, :] = df.loc[node].values
        arrays.append(tensor_3d)
    return np.concatenate(arrays, axis=0, dtype=np.float32)

