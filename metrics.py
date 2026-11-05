"""Evaluation metrics and bootstrap utilities."""
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score

def ece(P, y, bins=10):
    conf, pred = P.max(1), P.argmax(1); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def brier(P, y):
    Y = np.eye(P.shape[1])[y]; return ((P - Y) ** 2).sum(1).mean()

def auroc(P, y):
    if P.shape[1] == 2: return roc_auc_score(y, P[:, 1])
    return roc_auc_score(y, P, multi_class="ovr", average="macro")

def all_metrics(P, y):
    pred = P.argmax(1)
    return {"macroF1": f1_score(y, pred, average="macro"), "MCC": matthews_corrcoef(y, pred),
            "BalAcc": balanced_accuracy_score(y, pred), "AUROC": auroc(P, y),
            "ECE": ece(P, y), "Brier": brier(P, y)}

def _key(P, y, key):
    if key == "macroF1": return f1_score(y, P.argmax(1), average="macro")
    return all_metrics(P, y)[key]

def boot_ci(P, y, key, B=2000, seed=7):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < P.shape[1]: continue
        vals.append(_key(P[i], y[i], key))
    return np.percentile(vals, [2.5, 97.5])

def paired_boot_p(P1, P2, y, key="macroF1", B=2000, seed=11):
    """Two-sided paired bootstrap p-value for H0: metric(P1) = metric(P2)."""
    rng = np.random.default_rng(seed); n = len(y)
    d0 = _key(P1, y, key) - _key(P2, y, key); ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < P1.shape[1]: continue
        ds.append(_key(P1[i], y[i], key) - _key(P2[i], y[i], key))
    ds = np.array(ds)
    return d0, min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, k in enumerate(o):
        run = max(run, (m - r) * p[k]); adj[k] = min(1, run)
    return adj
