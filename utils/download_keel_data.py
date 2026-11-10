import os, urllib.request, zipfile

BASE = "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced"
DATASETS = {
    "abalone-19_vs_10-11-12-13":       "imb_IRhigherThan9p3",
    "yeast-1-2-8-9_vs_7":              "imb_IRhigherThan9p1",
    "flare-F":                         "imb_IRhigherThan9p3",
    "led7digit-0-2-4-5-6-7-8-9_vs_1":  "imb_IRhigherThan9p2",
    "glass1":                          "imb_IRlowerThan9",
}

os.makedirs("../data/keel", exist_ok=True)
for name, folder in DATASETS.items():
    url = f"{BASE}/{folder}/{name}.zip"
    out = f"../data/keel/{name}.zip"
    urllib.request.urlretrieve(url, out)
    print("Downloaded", name)