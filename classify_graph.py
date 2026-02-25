import networkx as nx
import pickle

# Load the graph
with open('data/domain_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Known tracker domains (seed list – expand from your blocklist later)
known_trackers = {
    "doubleclick.net", "googlesyndication.com", "googletagmanager.com", 
    "google-analytics.com", "facebook.net", "fbcdn.net", "adservice.google.com",
    "pubmatic.com", "rubiconproject.com", "outbrain.com", "taboola.com",
    "bat.bing.com", "static.xx.fbcdn.net", "rum.hlx.page"  # from your edges
}

# Function to classify a domain as tracker
def is_tracker(domain, graph):
    if domain in known_trackers:
        return True, "Known tracker"
    
    # Rule 1: High degree (many connections → suspicious)
    if graph.degree(domain) > 5:
        return True, "High degree"
    
    # Rule 2: Connected to known trackers (neighbors)
    neighbors = list(graph.neighbors(domain))
    if any(t in known_trackers for t in neighbors):
        return True, "Connected to known tracker"
    
    # Rule 3: Betweenness centrality (bridges ad networks)
    try:
        bc = nx.betweenness_centrality(graph, k=100)  # sample for speed
        if bc.get(domain, 0) > 0.01:
            return True, "High centrality"
    except:
        pass  # Skip if graph too large
    
    return False, "Benign"

# Test on sample domains from your graph
sample_domains = list(G.nodes())[:10]  # First 10 nodes
classified = {}
for domain in sample_domains:
    is_ad, reason = is_tracker(domain, G)
    classified[domain] = (is_ad, reason)
    print(f"{domain}: {'TRACKER' if is_ad else 'BENIGN'} ({reason})")

# Save classification results
with open("data/classified_domains.pkl", "wb") as f:
    pickle.dump(classified, f)

trackers_count = sum(1 for v in classified.values() if v[0])
print(f"\nClassification complete. Saved to data/classified_domains.pkl")
print(f"Trackers found: {trackers_count} / {len(classified)} total domains")
