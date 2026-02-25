'''import networkx as nx
from node2vec import Node2Vec
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# Import the extracted list
from known_trackers_from_layer1 import KNOWN_TRACKERS as known_trackers

# Load the graph
with open('data/domain_graph.gpickle', 'rb') as f:
    G = pickle.load(f)



# Prepare data for classification
nodes = list(G.nodes())
labels = [1 if node in known_trackers else 0 for node in nodes]  # 1 = tracker, 0 = benign

# Generate Node2Vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = np.array([model.wv[node] for node in nodes])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model for daemon use
with open("data/trained_model.pkl", "wb") as f:
    pickle.dump((model, clf), f)
print("Trained model saved to data/trained_model.pkl for daemon use")

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2%}")

# Classify all nodes
predictions = clf.predict(embeddings)
classified = {node: (bool(pred), "ML predicted tracker" if pred else "ML predicted benign") for node, pred in zip(nodes, predictions)}

# Save results
with open("data/advanced_classified.pkl", "wb") as f:
    pickle.dump(classified, f)

trackers_count = sum(1 for v in classified.values() if v[0])
print(f"Advanced classification complete.")
print(f"Trackers detected: {trackers_count} / {len(classified)} total domains")
'''
import networkx as nx
from node2vec import Node2Vec
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Import the full auto-updated Layer 1 blocklist as ground truth
from known_trackers_from_layer1 import KNOWN_TRACKERS as known_trackers

# Load the graph
with open('data/domain_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Prepare data for classification with overlap fix
nodes = list(G.nodes())
labels = [0 for _ in nodes]
positive_count = 0

for i, node in enumerate(nodes):
    if node in known_trackers:
        labels[i] = 1
        positive_count += 1
        print(f"Labeled {node} as tracker (direct match)")
    else:
        # Rule 1: High degree (many connections → suspicious)
        if G.degree(node) > 30:
            labels[i] = 1
            positive_count += 1
            print(f"Labeled {node} as tracker (high degree)")
        else:
            # Rule 2: Connected to known trackers (neighbors)
            neighbors = list(G.neighbors(node))
            if any(t in known_trackers for t in neighbors):
                labels[i] = 1
                positive_count += 1
                print(f"Labeled {node} as tracker (connected to known tracker)")
            else:
                # Rule 3: Betweenness centrality (bridges ad networks) - optional, slow
                try:
                    bc = nx.betweenness_centrality(G, k=100)  # sample for speed
                    if bc.get(node, 0) > 0.01:
                        labels[i] = 1
                        positive_count += 1
                        print(f"Labeled {node} as tracker (high centrality)")
                except:
                    pass  # Skip if graph too large

print(f"Graph: {len(nodes)} nodes, {G.number_of_edges()} edges")
print(f"Known trackers in graph (after overlap + rules): {positive_count} / {len(nodes)}")

# Generate Node2Vec embeddings
print("Generating Node2Vec embeddings...")
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = np.array([model.wv[node] for node in nodes])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train Random Forest classifier
print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2%}")

# Classify all nodes (hybrid prediction)
predictions = clf.predict(embeddings)
classified = {}
for i, node in enumerate(nodes):
    pred = predictions[i]
    # Optional: Combine with rule-based reason if you want more detail
    classified[node] = (bool(pred), "ML predicted tracker" if pred else "ML predicted benign")

# Save results
with open("data/advanced_classified.pkl", "wb") as f:
    pickle.dump(classified, f)

# Save trained model for daemon
with open("data/trained_model.pkl", "wb") as f:
    pickle.dump((model, clf), f)
print("Trained model saved to data/trained_model.pkl for daemon use")

trackers_count = sum(1 for v in classified.values() if v[0])
print(f"Advanced classification complete.")
print(f"Trackers detected: {trackers_count} / {len(classified)} total domains")
