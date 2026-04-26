import json

with open("arxiv-metadata-oai-snapshot_umap3d_index.json") as f:
    d = json.load(f)
    clusters = list(map(lambda x: x["cluster"], d))
    cset = set(clusters)
    print("Clusters:", cset)
    print("Count:", len(cset))
