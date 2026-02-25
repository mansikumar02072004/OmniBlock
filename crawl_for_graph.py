import httpx
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import networkx as nx
import pickle  # This is the key import for saving

# Load top domains from your CSV (skip header if any)
df = pd.read_csv("data/top-1m.csv", header=None, names=["rank", "domain"])
domains = df["domain"].head(5000).tolist()  # Start with top 50 for testing

edges = set()  # Use set to avoid duplicates

print("Starting crawl of top domains...")

for domain in domains:
    try:
        url = f"https://{domain}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = httpx.get(url, headers=headers, timeout=8, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")

        print(f"[OK] Crawled {domain} (status {response.status_code})")

        # Extract resource domains from script, img, link, iframe, source, etc.
        for tag in soup.find_all(["script", "img", "link", "iframe", "source", "video", "audio"]):
            src = tag.get("src") or tag.get("href") or tag.get("data-src") or tag.get("data-lazy-src")
            if src:
                if src.startswith("//"):
                    src = "https:" + src
                if "http" in src:
                    try:
                        dst = httpx.URL(src).host
                        if dst and dst != domain and "." in dst:  # valid domain, not self
                            edges.add((domain, dst))
                            print(f"  Edge: {domain} → {dst}")
                    except:
                        pass

        time.sleep(random.uniform(1.5, 3.5))  # polite delay

    except Exception as e:
        print(f"[Error] {domain}: {e}")

# Save raw edges
with open("data/crawled_edges.txt", "w") as f:
    for src, dst in sorted(edges):
        f.write(f"{src} {dst}\n")

print(f"\nCollected {len(edges)} unique edges. Saved to data/crawled_edges.txt")

# Build NetworkX graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Save graph using pickle directly (works on NetworkX 3.0+)
print("Saving graph...")
with open("data/domain_graph.gpickle", "wb") as f:
    pickle.dump(G, f, protocol=4)  # protocol=4 is efficient and compatible

print(f"Graph built and saved:")
print(f"  Nodes (domains): {G.number_of_nodes()}")
print(f"  Edges (connections): {G.number_of_edges()}")
print("Graph saved to data/domain_graph.gpickle")
