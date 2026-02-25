import networkx as nx
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Load graph
with open('data/domain_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Your expanded known_trackers (from above)
known_trackers = {
    "doubleclick.net", "googlesyndication.com", "googletagmanager.com", 
    "google-analytics.com", "facebook.net", "fbcdn.net", "adservice.google.com",
    "pubmatic.com", "rubiconproject.com", "outbrain.com", "taboola.com",
    "bat.bing.com", "static.xx.fbcdn.net", "rum.hlx.page", "www.googletagservices.com",
    "c.amazon-adsystem.com", "ib.adnxs.com", "pixel.quantserve.com", "trc.taboola.com",
    "aax.amazon-adsystem.com", "www.google-analytics.com", "analytics.google.com",
    "c.bing.com", "www.clarity.ms", "js-sec.indexww.com", "comcluster.cxense.com",
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
}

nodes = list(G.nodes())
labels = [1 if node in known_trackers else 0 for node in nodes]

# Generate embeddings and train
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = np.array([model.wv[node] for node in nodes])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(embeddings, labels)

# Save updated model
with open("data/model.pkl", "wb") as f:
    pickle.dump((model, clf), f)

print(f"Model re-trained. Trackers in seed: {sum(labels)} / {len(labels)}")
print("Saved to data/model.pkl")
