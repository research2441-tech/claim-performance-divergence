"""Monte Carlo study of the assurance design (E4-E6). The data-generating process is declared, not
estimated from any real panel; measurement error rates are taken from the real E1 confusion matrices."""
import json, sys, numpy as np
from common import OUT
Z = 1.959963984540054
YEARS = np.arange(2012, 2026); AN = YEARS >= 2015                     # 2012-2014 burn-in for claims
COH = {2018: 200, 2024: 200, 0: 200}                                   # 0 = not treated by 2025
E1 = json.load(open(f"{OUT}/E1_results.json"))
def rates(task):
    """Error rates of the annotator with the highest test macro-F1 for this attribute in E1."""
    best = max(E1[task], key=lambda m: E1[task][m]["macroF1"])
    (tn, fp), (fn, tp) = E1[task][best]["cm"]; return tp / (tp + fn), fp / (fp + tn)
BEST = {t: max(E1[t], key=lambda m: E1[t][m]["macroF1"]) for t in ("commit", "specific")}
SE_COM, FP_COM = rates("commit"); SE_SPEC, FP_SPEC = rates("specific")
WORLD = {"null": (0, 0), "closing": (0.03, 0), "quiet": (0, 0.45), "mixed": (0.015, 0.25)}
def simulate(rng, world, N_scale=1, dse=0.05, gate=True):
    dC, dQ = WORLD[world]
    g = np.concatenate([np.full(k * N_scale, c) for c, k in COH.items()]); N = len(g); T = len(YEARS)
    gam = rng.normal(-0.02, 0.01, N); mu = rng.normal(12, 1, N)
    tt = YEARS[None, :] - 2012; post = (g[:, None] > 0) & (YEARS[None, :] >= g[:, None])
    ramp = np.where(post, np.minimum(YEARS[None, :] - g[:, None] + 1, 3), 0)
    lnE = mu[:, None] + gam[:, None] * tt - dC * ramp + rng.normal(0, 0.04, (N, T))
    lam = rng.gamma(3.0, 0.5, N); nu = rng.gamma(3.0, 0.6, N); om = np.abs(rng.normal(0.02, 0.015, N))
    K = rng.poisson(lam[:, None] * (1 - dQ * ramp / 3))                 # verifiable claims issued
    U = rng.poisson(nu[:, None] * (1 + 0.5 * dQ * ramp / 3))            # unverifiable claims issued
    # claim tables
    fi, ti = np.nonzero(K); reps = K[fi, ti]; cf, ct = np.repeat(fi, reps), np.repeat(ti, reps)
    rho = np.maximum(0.005, -gam[cf] + om[cf] + rng.normal(0, 0.005, len(cf)))
    ui, ut = np.nonzero(U); ur = U[ui, ut]; vf, vt = np.repeat(ui, ur), np.repeat(ut, ur)
    # measurement: detection of verifiable claims, false verifiable parses of vague claims
    sens = SE_COM * np.clip(SE_SPEC + dse * post[cf, ct], 0, 1)
    det = rng.random(len(cf)) < sens
    rho_hat = np.where(rng.random(len(cf)) < 0.10, rho * rng.uniform(0.5, 1.5, len(cf)), rho)
    fake = rng.random(len(vf)) < (FP_SPEC if gate else 1.0)
    vrho = rng.uniform(0.01, 0.08, len(vf))
    vdet = rng.random(len(vf)) < SE_COM                                   # vague claims seen as climate commitments
    def aggregate(f, t, r, keep):
        A = np.zeros((N, T)); M = np.zeros((N, T))
        for lag in (1, 2, 3):
            te = t + lag; ok = keep & (te < T)
            s = np.maximum(0, lnE[f[ok], te[ok]] - (lnE[f[ok], t[ok]] + lag * np.log1p(-r[ok])))
            np.add.at(A, (f[ok], te[ok]), s); np.add.at(M, (f[ok], te[ok]), 1)
        return A, M
    A, M = aggregate(cf, ct, rho, np.ones(len(cf), bool))
    Ad1, Md1 = aggregate(cf, ct, rho_hat, det); Ad2, Md2 = aggregate(vf, vt, vrho, fake & vdet)
    Ah, Mh = Ad1 + Ad2, Md1 + Md2
    def window(X):
        W = np.zeros_like(X, dtype=float)
        for lag in (1, 2, 3): W[:, lag:] += X[:, :-lag]
        return W
    Uw = window(U)
    Uh = np.zeros((N, T)); np.add.at(Uh, (vf[vdet & ~fake], vt[vdet & ~fake]), 1); Uh = window(Uh)
    Kmiss = np.zeros((N, T)); np.add.at(Kmiss, (cf[~det], ct[~det]), 1); Uh += window(Kmiss) * SE_COM
    S = np.where(M > 0, A / np.maximum(M, 1), np.nan); Sh = np.where(Mh > 0, Ah / np.maximum(Mh, 1), np.nan)
    Tx = Uw / np.maximum(Uw + M, 1); Txh = Uh / np.maximum(Uh + Mh, 1)
    return dict(g=g, true=dict(A=A, M=M, S=S, T=Tx), hat=dict(A=Ah, M=Mh, S=Sh, T=Txh))
def firm_stats(Y, g):
    """a_i^c = mean over post(c) of Y_it - Y_i,c-1 for each cohort c, per firm (nan-aware for S)."""
    out = {}
    for c in (2018, 2024):
        b = np.where(YEARS == c - 1)[0][0]; p = (YEARS >= c)
        base = Y[:, b]
        if np.isnan(Y).any():                                            # shortfall: pre mean over years with claims
            pre = (YEARS < c) & AN; base = np.nanmean(Y[:, pre], 1)
        out[c] = np.nanmean(Y[:, p], 1) - base
    return out
def att_cs(Y, g, idx=None):
    st = firm_stats(Y, g); w = {c: COH[c] * (YEARS >= c).sum() for c in (2018, 2024)}; ws = sum(w.values())
    w = {c: v / ws for c, v in w.items()}
    est, var = 0, 0
    ctrl = g == 0; cvec = sum(w[c] * st[c] for c in w)
    for c in w:
        z = st[c][g == c]; z = z[~np.isnan(z)]; est += w[c] * z.mean(); var += w[c] ** 2 * z.var(ddof=1) / len(z)
    z = cvec[ctrl]; z = z[~np.isnan(z)]; est -= z.mean(); var += z.var(ddof=1) / len(z)
    return est, np.sqrt(var)
def att_cs_ppi(Yt, Yh, g, gold, tuned=False):
    stt, sth = firm_stats(Yt, g), firm_stats(Yh, g)
    w = {c: COH[c] * (YEARS >= c).sum() for c in (2018, 2024)}; ws = sum(w.values()); w = {c: v / ws for c, v in w.items()}
    def part(zt, zh, grp):
        m = grp & ~np.isnan(zh); u, l = m & ~gold, m & gold & ~np.isnan(zt)
        lam = 1.0
        if tuned:
            v = np.var(zh[m], ddof=1); lam = float(np.clip(np.cov(zt[l], zh[l])[0, 1] / ((1 + l.sum() / u.sum()) * v), 0, 1)) if v > 0 else 0.0
        rect = zt[l] - lam * zh[l]; return lam * zh[u].mean() + rect.mean(), lam ** 2 * zh[u].var(ddof=1) / u.sum() + rect.var(ddof=1) / l.sum()
    est, var = 0, 0
    for c in w:
        e, v = part(stt[c], sth[c], g == c); est += w[c] * e; var += w[c] ** 2 * v
    ct_ = sum(w[c] * stt[c] for c in w); ch_ = sum(w[c] * sth[c] for c in w)
    e, v = part(ct_, ch_, g == 0); return est - e, np.sqrt(var + v)
def att_twfe(Y, g):
    N, T = Y.shape; D = ((g[:, None] > 0) & (YEARS[None, :] >= g[:, None])).astype(float)
    m = AN[None, :] & ~np.isnan(Y); Yv = np.where(m, Y, 0.0)
    def demean(X):
        X = np.where(m, X, 0.0)
        for _ in range(30):
            X = X - np.where(m, (X.sum(1) / m.sum(1))[:, None], 0); X = X - np.where(m, (X.sum(0) / np.maximum(m.sum(0), 1))[None, :], 0)
        return X
    yd, dd = demean(Yv), demean(D); beta = (yd * dd).sum() / (dd * dd).sum()
    e = (yd - beta * dd) * m; sc = (dd * e).sum(1)
    return beta, np.sqrt((sc ** 2).sum()) / (dd * dd).sum()
def one_rep(rng, world, q=0.20, dse=0.05, gate=True):
    s = simulate(rng, world, dse=dse, gate=gate); g = s["g"]
    gold = np.zeros(len(g), bool)
    for c in COH:
        idx = np.where(g == c)[0]; gold[rng.choice(idx, max(2, int(round(q * len(idx)))), replace=False)] = True
    r = {}
    for y in ("A", "M", "S", "T"):
        Yt, Yh = s["true"][y], s["hat"][y]
        r[y] = {"oracle_cs": att_cs(Yt, g), "oracle_twfe": att_twfe(Yt, g), "plugin_cs": att_cs(Yh, g), "plugin_twfe": att_twfe(Yh, g),
                "gold_cs": att_cs(Yt[gold], g[gold]), "ppi_cs": att_cs_ppi(Yt, Yh, g, gold),
                "ppit_cs": att_cs_ppi(Yt, Yh, g, gold, tuned=True)}
    # channel shares from plug-in-free (PPI) estimates
    pre = (YEARS < 2024) & AN
    M0 = np.nanmean(s["true"]["M"][(g > 0)][:, pre]); S0 = np.nanmean(s["true"]["S"][(g > 0)][:, pre])
    a, m_, sh = r["A"]["ppi_cs"][0], r["M"]["ppi_cs"][0], r["S"]["ppi_cs"][0]
    r["share"] = {"claim": S0 * m_ / a if abs(a) > 1e-9 else np.nan, "perf": M0 * sh / a if abs(a) > 1e-9 else np.nan}
    return r
def truth(world, reps=20):
    rng = np.random.default_rng(99); acc = {y: [] for y in "AMST"}; m0, s0 = [], []
    pre = (YEARS < 2024) & AN
    for _ in range(reps):
        s = simulate(rng, world, N_scale=10, dse=0.0)
        for y in "AMST": acc[y].append(att_cs(s["true"][y], s["g"])[0])
        m0.append(np.nanmean(s["true"]["M"][s["g"] > 0][:, pre])); s0.append(np.nanmean(s["true"]["S"][s["g"] > 0][:, pre]))
    t = {y: float(np.mean(v)) for y, v in acc.items()}
    t["share_claim"] = float(np.mean(s0) * t["M"] / t["A"]) if abs(t["A"]) > 1e-3 else float("nan")
    t["share_perf"] = float(np.mean(m0) * t["S"] / t["A"]) if abs(t["A"]) > 1e-3 else float("nan")
    return t
if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    tr = {}
    for w in WORLD:
        tr[w] = truth(w, 4); print("truth", w, tr[w], flush=True)
    res = {"truth": tr, "rates": [SE_COM, FP_COM, SE_SPEC, FP_SPEC], "best": BEST, "main": {}, "ablation_gate": {}, "sens": {}}
    rng = np.random.default_rng(20260508)
    for w in WORLD:
        res["main"][w] = [one_rep(rng, w) for _ in range(R)]; print("main", w, flush=True)
        res["ablation_gate"][w] = [one_rep(rng, w, gate=False) for _ in range(R // 4)]
    for q in (0.10, 0.20, 0.30):
        for dse in (0.0, 0.05, 0.10):
            res["sens"][f"{q}|{dse}"] = [one_rep(rng, "quiet", q=q, dse=dse) for _ in range(R // 4)]
        print("sens", q, flush=True)
    json.dump(res, open(f"{OUT}/sim_raw.json", "w"), default=float)
