"""Event-time profile for the 2018 cohort (never-treated controls), true vs plug-in vs rectified."""
import json, numpy as np
import simulate_assurance as S
from common import OUT
YEARS = S.YEARS; g0 = 2018; base = np.where(YEARS == g0 - 1)[0][0]
EV = [e for e in range(-3, 8) if e != -1]
def contrasts(Y, g):
    out = {}
    for e in EV:
        t = np.where(YEARS == g0 + e)[0][0]; out[e] = Y[:, t] - Y[:, base]
    return out
def est(zt, zh, grp, gold, tuned):
    m = grp & ~np.isnan(zh); u, l = m & ~gold, m & gold & ~np.isnan(zt)
    lam = 1.0
    if tuned:
        v = np.var(zh[m], ddof=1); lam = float(np.clip(np.cov(zt[l], zh[l])[0, 1] / ((1 + l.sum() / u.sum()) * v), 0, 1)) if v > 0 else 0.0
    r = zt[l] - lam * zh[l]; return lam * zh[u].mean() + r.mean(), lam ** 2 * zh[u].var(ddof=1) / u.sum() + r.var(ddof=1) / l.sum()
res = {}
rng = np.random.default_rng(777)
for world, y in (("quiet", "M"), ("closing", "S")):
    acc = {k: {e: [] for e in EV} for k in ("truth", "plugin", "ppi")}
    for _ in range(200):
        s = S.simulate(rng, world); g = s["g"]; gold = np.zeros(len(g), bool)
        for c in S.COH:
            idx = np.where(g == c)[0]; gold[rng.choice(idx, int(0.2 * len(idx)), replace=False)] = True
        Yt, Yh = s["true"][y], s["hat"][y]
        if y == "S":   # shortfall: use firm-level values where defined, mean-imputed by firm pre-period
            Yt = np.where(np.isnan(Yt), np.nanmean(Yt[:, :base + 1], 1, keepdims=True), Yt); Yh = np.where(np.isnan(Yh), np.nanmean(Yh[:, :base + 1], 1, keepdims=True), Yh)
        ct, ch = contrasts(Yt, g), contrasts(Yh, g); tr_, ctl = g == g0, g == 0
        for e in EV:
            a = ct[e][tr_]; b = ct[e][ctl]; acc["truth"][e].append(np.nanmean(a) - np.nanmean(b))
            acc["plugin"][e].append(np.nanmean(ch[e][tr_]) - np.nanmean(ch[e][ctl]))
            e1, v1 = est(ct[e], ch[e], tr_, gold, True); e2, v2 = est(ct[e], ch[e], ctl, gold, True)
            acc["ppi"][e].append((e1 - e2, np.sqrt(v1 + v2)))
    res[world] = {"y": y, "ev": EV, "truth": [float(np.mean(acc["truth"][e])) for e in EV],
                  "plugin": [float(np.mean(acc["plugin"][e])) for e in EV],
                  "ppi": [float(np.mean([a for a, _ in acc["ppi"][e]])) for e in EV],
                  "ppi_lo": [float(np.percentile([a for a, _ in acc["ppi"][e]], 2.5)) for e in EV],
                  "ppi_hi": [float(np.percentile([a for a, _ in acc["ppi"][e]], 97.5)) for e in EV],
                  "ppi_cov": [float(np.mean([abs(a - np.mean(acc["truth"][e])) <= 1.96 * sd for a, sd in acc["ppi"][e]])) for e in EV]}
    print(world, {k: np.round(v, 3).tolist() for k, v in res[world].items() if k in ("truth", "plugin", "ppi", "ppi_cov")}, flush=True)
json.dump(res, open(f"{OUT}/sim_event.json", "w"), indent=1)
