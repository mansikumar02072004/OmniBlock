#!/usr/bin/env python3
# graphguard_daemon.py - Real-time Layer 3 blocking using tshark + GraphGuard model
# Run with: sudo ~/OmniBlock/env/bin/python graphguard_daemon.py

import subprocess
import time
import pickle
import networkx as nx
from node2vec import Node2Vec
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import logging
# Import the extracted list
from known_trackers_from_layer1 import KNOWN_TRACKERS

# ======================
# Configuration
# ======================
INTERFACE = "wlo1"                  # Your hotspot interface
GRAPH_FILE = "data/domain_graph.gpickle"
MODEL_FILE = "data/model.pkl"
LOG_FILE = "daemon.log"

'''KNOWN_TRACKERS = {
    # From your output and common trackers
    "doubleclick.net", "googlesyndication.com", "googletagmanager.com",
    "google-analytics.com", "facebook.net", "fbcdn.net", "adservice.google.com",
    "pubmatic.com", "rubiconproject.com", "outbrain.com", "taboola.com",
    "bat.bing.com", "static.xx.fbcdn.net", "rum.hlx.page",
    "www.googletagservices.com", "c.amazon-adsystem.com", "ib.adnxs.com",
    "pixel.quantserve.com", "trc.taboola.com", "aax.amazon-adsystem.com",
    "www.google-analytics.com", "analytics.google.com", "c.bing.com",
    "www.clarity.ms", "js-sec.indexww.com", "comcluster.cxense.com",
    "smetrics.cnn.com", "api.id5-sync.com", "ssum-sec.casalemedia.com",
    "onsiterecs.api.boomtrain.com", "pixel.adsafeprotected.com", "be.a4.v.fwmrm.net",
    "pixel-us-east.rubiconproject.com", "staticjs.adsafeprotected.com",
    "dt.adsafeprotected.com", "jsconfig.adsafeprotected.com", "js-agent.newrelic.com",
    "data.pendo.io", "bam.nr-data.net", "p1.parsely.com", "api.bounceexchange.com",
    "events.bouncex.net", "api.permutive.com", "eq97f.publishers.tremorhub.com",
    "live.rezync.com", "c2.piano.io", "secure.cdn.fastclick.net", "view.cdnbasket.net",
    "page.cdnbasket.net", "data.cdnbasket.net", "ids.cdnwidget.com", "e.cdnwidget.com",
    "api.btloader.com", "btloader.com", "mabping.chartbeat.net", "ping.chartbeat.net",
    "mab.chartbeat.com", "tag.wknd.ai", "experience.piano.io", "secure-us.imrworldwide.com",
    "cadmus.script.ac", "api.id5-sync.com", "ssum-sec.casalemedia.com", "onsiterecs.api.boomtrain.com",
    "pixel.adsafeprotected.com", "be.a4.v.fwmrm.net", "prod.us-east-1.cxm-bcn.publisher-services.amazon.dev",
    "events.brightline.tv", "zion.api.cnn.io", "e.cdnwidget.com", "eq97f.publishers.tremorhub.com",
    "pixel-us-east.rubiconproject.com", "staticjs.adsafeprotected.com", "dt.adsafeprotected.com",
    "jsconfig.adsafeprotected.com", "js-agent.newrelic.com", "data.pendo.io", "bam.nr-data.net",
    "p1.parsely.com", "api.bounceexchange.com", "events.bouncex.net", "api.permutive.com",
    "live.rezync.com", "c2.piano.io", "secure.cdn.fastclick.net", "view.cdnbasket.net",
    "page.cdnbasket.net", "data.cdnbasket.net", "ids.cdnwidget.com", "e.cdnwidget.com",
    "api.btloader.com", "btloader.com", "mabping.chartbeat.net", "ping.chartbeat.net"
}'''

# ======================
# Logging setup
# ======================
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logging.info("GraphGuard daemon started")

# ======================
# Load & (re)train model
# ======================
logging.info("Loading graph...")
with open(GRAPH_FILE, 'rb') as f:
    G = pickle.load(f)

nodes = list(G.nodes())
labels = [1 if node in KNOWN_TRACKERS else 0 for node in nodes]

logging.info("Generating Node2Vec embeddings...")
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = np.array([model.wv[node] for node in nodes])

logging.info("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(embeddings, labels)

# Save updated model
with open(MODEL_FILE, "wb") as f:
    pickle.dump((model, clf), f)
logging.info(f"Model re-trained. Trackers in seed: {sum(labels)} / {len(labels)}")

# ======================
# Classification function
# ======================
def is_tracker(sni):
    """
    Hybrid tracker detection with clear debug messages:
    - Shows if blocked by rules or ML
    """
    # Approach 1: Static list (100% accurate)
    if sni in KNOWN_TRACKERS:
        return True, "Blocked by rules: Known tracker (static list)"

    # Approach 2: Neighbor + 2-hop + high degree (rule-based)
    if sni in G.nodes():
        # Direct neighbors (1-hop)
        neighbors = list(G.neighbors(sni))
        if any(neigh in KNOWN_TRACKERS for neigh in neighbors):
            return True, "Blocked by rules: Direct neighbor of known tracker (1-hop)"

        # 2-hop: neighbors of neighbors
        for neigh in neighbors:
            second_hop = list(G.neighbors(neigh))
            if any(sh in KNOWN_TRACKERS for sh in second_hop):
                return True, "Blocked by rules: Neighbor's neighbor of known tracker (2-hop)"

        # High degree (only if not benign CDN)
        benign_cdns = {"google.com", "www.google.com", "gstatic.com", "googleapis.com", "fonts.googleapis.com", "fonts.gstatic.com",
                       "youtube.com", "ytimg.com", "ggpht.com", "youtubei.googleapis.com", "oauthaccountmanager.googleapis.com",
                       "play.googleapis.com", "jnn-pa.googleapis.com", "app-analytics-services.com", "ampproject.org", "cdn.jsdelivr.net","cdnjs.cloudflare.com", "www.youtube.com" }
        if sni not in benign_cdns:
            degree = G.degree(sni)
            if degree > 20:
                return True, f"Blocked by rules: High degree ({degree} connections)"

    # Approach 3: ML prediction (only if rules didn't catch it)
    try:
        emb = model.wv[sni]
        pred = clf.predict([emb])[0]
        if pred:
            return True, "Blocked by ML: ML predicted tracker (Node2Vec + RF)"
    except KeyError:
        pass  # No embedding → skip

    return False, "Benign (no match)"
'''
def is_tracker(sni):
   
    """
    Hybrid tracker detection:
    1. Static list (100% accurate)
    2. Neighbor/2-hop check (rule-based, catches short cloaking chains)
    3. ML prediction (Node2Vec + Random Forest, high coverage)
    """
    # Approach 1: Static list (fast, 100% accurate)
    if sni in KNOWN_TRACKERS:
        return True, "Known tracker (static list)"

    # Approach 2: Neighbor + 2-hop check (rule-based, catches direct/short cloaking)
    if sni in G.nodes():
        # Direct neighbors (1-hop)
        neighbors = list(G.neighbors(sni))
        if any(neigh in KNOWN_TRACKERS for neigh in neighbors):
            return True, "Direct neighbor of known tracker (1-hop)"

        # 2-hop: neighbors of neighbors
        for neigh in neighbors:
            second_hop = list(G.neighbors(neigh))
            if any(sh in KNOWN_TRACKERS for sh in second_hop):
                return True, "Neighbor's neighbor of known tracker (2-hop)"

    # Approach 3: ML prediction (advanced coverage for longer/new cloaking)
    try:
        emb = model.wv[sni]
        pred = clf.predict([emb])[0]
        if pred:
            return True, "ML predicted tracker (Node2Vec + RF)"
    except KeyError:
        pass  # No embedding → skip

    return False, "Benign (no match)"'''

# ======================
# Drop function (nftables)
# ======================
'''def drop_sni(sni):
    rule = f"nft add rule ip graphguard forward meta l4proto tcp th dport 443 sni {sni} drop comment \"GraphGuard block {sni}\""
    result = subprocess.run(["sudo"] + rule.split(), capture_output=True, text=True)
    if result.returncode == 0:
        logging.info(f"Dropped {sni}")
        print(f"[BLOCKED] Dropped {sni}")
    else:
        logging.error(f"nft rule failed: {result.stderr}")
        print(f"[ERROR] nft rule failed for {sni}: {result.stderr}")'''
blocked_ips = set()  # Global or class variable

def drop_sni(sni):
    try:
        result = subprocess.check_output(["dig", "+short", sni], timeout=5).decode().strip()
        ips = [ip for ip in result.split('\n') if ip and '.' in ip]

        if not ips:
            print(f"[WARNING] No IPs resolved for {sni}")
            return

        for ip in ips:
            if ip in blocked_ips:
                print(f"[INFO] Already blocked IP {ip} for {sni}")
                continue

            rule = f"nft add rule ip nm-shared-wlo1 filter_forward ip daddr {ip} tcp dport 443 drop comment \"GraphGuard block {sni} ({ip})\""
            res = subprocess.run(["sudo"] + rule.split(), capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[BLOCKED] IP {ip} for {sni}")
                blocked_ips.add(ip)
            else:
                print(f"[ERROR] Failed to block IP {ip} for {sni}: {res.stderr}")
        
    except Exception as e:
        print(f"[ERROR] Failed to block {sni}: {e}")

# ======================
# Main loop: tshark SNI extraction
# ======================
logging.info(f"Starting daemon – listening on {INTERFACE}")
print(f"Starting GraphGuard daemon – listening on {INTERFACE}...")

while True:
    try:
        cmd = [
            "tshark",
            "-i", INTERFACE,
            "-Y", "tls.handshake.type == 1",
            "-T", "fields",
            "-e", "tls.handshake.extensions_server_name",
            "-l"  # line-buffered
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        for line in process.stdout:
            sni = line.strip()
            if sni:
                print(f"[SNI] {sni}")
                logging.info(f"SNI detected: {sni}")
                is_ad, reason = is_tracker(sni)
                if is_ad:
                    drop_sni(sni)
                    logging.info(f"Blocked {sni} ({reason})")
                    print(f"[BLOCKED] {sni} ({reason})")

        process.wait()
        logging.warning("tshark exited. Restarting in 5s...")
        print("tshark exited. Restarting in 5s...")
        time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Daemon stopped by user")
        print("\nDaemon stopped by user.")
        if 'process' in locals():
            process.terminate()
        break
    except Exception as e:
        logging.error(f"Daemon error: {e}")
        print(f"Error: {e}")
        time.sleep(5)
