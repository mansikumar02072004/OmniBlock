import networkx as nx
import pickle

# Load the graph
with open('data/domain_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

print('Success! Nodes:', G.number_of_nodes())
print('Edges:', G.number_of_edges())
print('Sample edges:', list(G.edges())[:5])
