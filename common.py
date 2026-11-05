"""Shared loaders. Every corpus carries human labels only; no label is produced by this code."""
import os, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

def load_task(task):
    """Return a frame with text, y (int), split. 'joint' is the 1,320-paragraph set that
    carries commitment, specificity and sentiment labels for the same paragraph."""
    if task == "target":
        d = pd.read_csv(os.path.join(DATA, "targets_final.csv"))[["text", "target"]]
        d["y"] = d.target.map({"none": 0, "reduction": 1, "net-zero": 2})
        rng = np.random.default_rng(2026)
        d["split"] = "train"
        for k in range(3):                       # stratified 80/20 split, fixed seed
            idx = d.index[d.y == k].to_numpy().copy(); rng.shuffle(idx)
            d.loc[idx[: int(round(0.2 * len(idx)))], "split"] = "test"
        return d[["text", "y", "split"]]
    if task == "joint":
        c = pd.read_csv(os.path.join(DATA, "commit.csv")).rename(columns={"label": "commit"})
        s = pd.read_csv(os.path.join(DATA, "specific.csv")).rename(columns={"label": "specific"})
        m = pd.read_csv(os.path.join(DATA, "sentiment.csv")).rename(columns={"label": "sentiment"})
        d = c.merge(s[["text", "specific"]], on="text").merge(m[["text", "sentiment"]], on="text")
        return d
    d = pd.read_csv(os.path.join(DATA, {"claim": "claim", "commit": "commit", "specific": "specific",
                                        "sentiment": "sentiment"}[task] + ".csv"))
    d = d.rename(columns={"label": "y"})
    if task == "claim":
        d.loc[d.split == "validation", "split"] = "train"   # validation folded into training pool
    return d[["text", "y", "split"]]
