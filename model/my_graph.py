import torch
import numpy as np
import networkx as nx
import igraph as ig
from community import community_louvain

from .emb_methods.embedding import EmbeddingFactory
from utils import average_shortest_path_length


class MyGraph:
    """
    MyGraph provides methods for 
    1. obtaining node feature tensors
    2. retrieving node observation tensors, and 
    3. supplying network information required for fitting.
    """
    def __init__(self, init_graph: nx.Graph, args):
        self.nx_graph = init_graph

        self.node_num = int(init_graph.number_of_nodes())
        self.node_emb_dims = int(args.node_emb_dims)
        self.first_neighbors_of_node_dict = {}
        self.second_neighbors_of_node_dict = {}
        self.emb_of_nodes_dict = {}

        self.init(args)

    def get_fit_info(self):
        modularity = community_louvain.modularity(community_louvain.best_partition(self.nx_graph), self.nx_graph)
        clustering = nx.average_clustering(self.nx_graph)
        shortestpath = average_shortest_path_length(self.nx_graph)
        assortativity = nx.degree_assortativity_coefficient(self.nx_graph)
        degree_cur = nx.degree_histogram(self.nx_graph)
        edge_num = nx.number_of_edges(self.nx_graph)
        return {
            "edge_num": edge_num,
            "modularity": modularity,
            "clustering": clustering,
            "avg_path_length":shortestpath,
            "assortativity": assortativity,
            "degree_distribution": degree_cur,  # 可选保存
        }

    def get_nx_graph(self):
        return self.nx_graph

    def get_node_emb_dict(self):
        return self.emb_of_nodes_dict

    def get_node_emb(self, node_id):
        return self.emb_of_nodes_dict[node_id]
    
    def get_second_neighbor(self, node_id):
        return self.first_neighbors_of_node_dict[node_id], self.second_neighbors_of_node_dict[node_id]

    def init(self, args):
        self._get_neighbors()
        self.emb_of_nodes_dict = self.create_node_emb(method=args.emb_method)

    def _get_neighbors(self):
        for node_id in range(self.node_num):
            first_neighbors, second_neighbors = self._get_second_neighbor_by_ig(node_id)
            self.first_neighbors_of_node_dict[node_id] = first_neighbors
            self.second_neighbors_of_node_dict[node_id] = second_neighbors
    
    def _get_second_neighbor_by_ig(self, node_id):
        g = ig.Graph.from_networkx(self.nx_graph)
        sec = g.neighborhood(node_id, order=2)
        sec.remove(node_id)

        first = g.neighborhood(node_id, order=1)
        first.remove(node_id)
        return first, sec


    def get_obs_tensor(self, node_id, neighbor_list, device):
        emb_sum = np.zeros(self.node_emb_dims)
        emb_sum += self.emb_of_nodes_dict[node_id]
        for i in neighbor_list:
            emb_sum += self.emb_of_nodes_dict.get(i)

        sum_tensor = torch.as_tensor(emb_sum, dtype=torch.float32)
        sum_tensor = sum_tensor.expand([self.node_num, self.node_emb_dims]).to(device)

        embs_list = list(self.emb_of_nodes_dict.values())
        obs_tensor = torch.as_tensor(np.array(embs_list), device=device, dtype=torch.float32)
        obs = torch.cat((obs_tensor, sum_tensor), dim=1)

        return obs

    def create_node_emb(self, method="wavelet"):
        assert method in ["s2c", "wavelet"]
        emb_model = EmbeddingFactory.create_emb(method)
        return emb_model.create_node_embedding(self.nx_graph)

    def save_gexf(self, out_path):
        nx.write_gexf(self.nx_graph, out_path)
