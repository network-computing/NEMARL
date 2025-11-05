import networkx as nx

# 使用原始字符串，以避免路径问题
filename = r"C:\Users\13509\Desktop\Complex Network\karate.txt"
G = nx.read_adjlist(filename, comments='%')
# 打印图的节点和边缘数量
print("节点数量:", G.number_of_nodes())
print("边缘数量:", G.number_of_edges())


# 打印图的度分布
# degree_sequence = [graph.degree(node) for node in graph.nodes()]
degree_count = nx.degree_histogram(G)
print("度分布:", degree_count)