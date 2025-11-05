
# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx

filename = r"C:\Users\13509\Desktop\Network 06June24\new\bmwiktionary\edit-bmwiktionary\out.edit-bmwiktionary"

data = pd.read_csv(filename, sep='\t', header=None, names=['from_node', 'to_node', 'weight', 'timestamp'])

G = nx.DiGraph()

for index, row in data.iterrows():
    G.add_edge(row['from_node'], row['to_node'], weight=row['weight'], timestamp=row['timestamp'])

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

degree_count = nx.degree_histogram(G)
print("Degree distribution", degree_count)
