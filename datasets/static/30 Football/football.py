import networkx as nx

# 使用原始字符串，以避免路径问题
filename = r"C:\Users\13509\Desktop\Complex Network\football.gml"

# 读取 GML 文件
G = nx.read_gml(filename)

# 打印图的节点和边缘数量
print("节点数量:", G.number_of_nodes())
print("边缘数量:", G.number_of_edges())

# 打印图的度分布
degree_count = nx.degree_histogram(G)
print("度分布:", degree_count)