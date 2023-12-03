from gradflow.tensor import Tensor

a = Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], name='a0')
b = a[0]
b._name = 'b'
c = b+1
c._name = 'c'
d = b+2
d._name = 'd'
a[0] = b
a[1] = c
a[2] = d
print(a)
print(b)

nodes = []
edges = []
topo = []
visited = set()
def generate_graph(t: Tensor):
    if t not in visited:
        visited.add(t)
        nodes.append(t._name)
        for child in t._operands:
            edges.append((child._name, t._name))
        for child in t._operands:
            generate_graph(child)
        topo.append(t._name)


generate_graph(a)
print(topo)

import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes to the graph
G.add_nodes_from(nodes)

# Add edges to the graph
G.add_edges_from(edges)

pos = nx.planar_layout(G)

nx.draw_networkx_labels(G, pos=pos, font_size=12)

# Draw the graph
nx.draw(G, pos=pos)

# Show the graph
plt.show()