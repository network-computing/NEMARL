import networkx as nx
from abc import ABC, abstractmethod

import numpy as np

from .s2v import Struc2Vec
from .wavelet import WaveletMachine


class EmbeddingInterface(ABC):

    @abstractmethod
    def create_node_embedding(self, graph: nx.Graph) -> dict:
        pass


class S2C(EmbeddingInterface):

    def __init__(self):
        self.try_cnt = 10

    def create_node_embedding(self, graph: nx.Graph) -> dict:
        node_num = graph.number_of_nodes()
        embs_list = []
        for _ in range(self.try_cnt):
            s2c_model = Struc2Vec(graph, walk_length=20, num_walks=50, workers=4, verbose=0)  # init model
            s2c_model.train(window_size=8, iter=5, seed=42)  # train model
            embs_dict = s2c_model.get_embeddings()  # get embedding vectors

            embs_list.append(embs_dict)

        avg_embeddings = {}
        for node in range(node_num):
            node_vectors = []
            for embs in embs_list:
                node_vectors.append(embs[node])

            if node_vectors:
                # Calculate the average vector for this node
                avg_vector = np.mean(node_vectors, axis=0)
                avg_embeddings[node] = avg_vector

        return avg_embeddings


class WaveLet(EmbeddingInterface):

    def create_node_embedding(self, graph: nx.Graph) -> dict:
        machine = WaveletMachine(graph)
        machine.create_embedding()
        return machine.transform_and_save_embedding()


class EmbeddingFactory:
    _method_model = None

    @classmethod
    def create_emb(cls, emb_method: str) -> EmbeddingInterface:
        if cls._method_model:
            return cls._method_model

        if emb_method == "s2c":
            cls._method_model = S2C()
        elif emb_method == "wavelet":
            cls._method_model = WaveLet()
        else:
            raise ValueError(f"Unknown embedding method: {emb_method}")
        return cls._method_model


if __name__ == "__main__":
    s1 = EmbeddingFactory.create_emb("s2c")
    s2 = EmbeddingFactory.create_emb("s2c")
    print(s1 is s2)


