"""E3: coverage and width of 95% intervals for population quantities whose true values are known
from human labels. Gold samples of size n are drawn repeatedly; the remaining texts carry only
machine predictions (cross-fitted encoder probabilities or zero-shot LLM probabilities)."""
import json, os, numpy as np
from common import load_task, OUT
from ppi import mean_classical, mean_naive, mean_ppi, ols, ols_ppi, Z
R, NS = 2000, [50, 100, 200, 400]
rng = np.random.default_rng(20260508)
d = load_task("joint")
def P(name): return np.load(f"{OUT}/{name}.npy")
pc, ps, pt = P("oof_climatebert_lr_commit")[:, 1], P("oof_climatebert_lr_specific")[:, 1], P("oof_climatebert_lr_sentiment")
setups = {}
c, s, t = d.commit.to_numpy(float), d.specific.to_numpy(float), d.sentiment.to_numpy()
setups["theta1_enc"] = ("mean", c * (1 - s), pc * (1 - ps), ((pc > .5) & (ps <= .5)).astype(float))
xc, xs = P("oof_lexicon_commit")[:, 1], P("oof_lexicon_specific")[:, 1]
setups["theta1_lex"] = ("mean", c * (1 - s), xc * (1 - xs), ((xc > .5) & (xs <= .5)).astype(float))
LLMF = f"{OUT}/llm_Qwen2.5-0.5B-Instruct_test_commit.npy"
te = (d.split == "test").to_numpy()
if os.path.exists(LLMF):     # zero-shot LLM annotator: population = the 320 test paragraphs it labelled
    lc, ls = P("llm_Qwen2.5-0.5B-Instruct_test_commit")[:, 1], P("llm_Qwen2.5-0.5B-Instruct_test_specific")[:, 1]
    setups["theta1_llm"] = ("mean", (c * (1 - s))[te], lc * (1 - ls), ((lc > .5) & (ls <= .5)).astype(float))
cl = load_task("claim"); pcl = P("oof_climatebert_lr_claim")[:, 1]
setups["theta3_claim"] = ("mean", cl.y.to_numpy(float), pcl, (pcl > .5).astype(float))
tg = load_task("target"); ptg = P("oof_climatebert_lr_target")
setups["theta4_netzero"] = ("mean", (tg.y.to_numpy() == 2).astype(float), ptg[:, 2], (ptg.argmax(1) == 2).astype(float))
# regression: specificity on commitment and tone (linear probability model), all variables predicted off-gold
X = np.c_[np.ones(len(d)), c, (t == 2), (t == 0)].astype(float)
Xf = np.c_[np.ones(len(d)), pc, pt[:, 2], pt[:, 0]]
Xh = np.c_[np.ones(len(d)), (pc > .5), pt.argmax(1) == 2, pt.argmax(1) == 0].astype(float)
setups["theta2_beta_commit"] = ("ols", (X, s), (Xf, ps), (Xh, (ps > .5).astype(float)))
Xl = np.c_[np.ones(len(d)), xc, pt[:, 2], pt[:, 0]]; Xlh = np.c_[np.ones(len(d)), xc > .5, pt.argmax(1) == 2, pt.argmax(1) == 0].astype(float)
setups["theta2_beta_lex"] = ("ols", (X, s), (Xl, xs), (Xlh, (xs > .5).astype(float)))
if os.path.exists(LLMF):
    Xq = np.c_[np.ones(te.sum()), lc, pt[te, 2], pt[te, 0]]; Xqh = np.c_[np.ones(te.sum()), lc > .5, pt[te].argmax(1) == 2, pt[te].argmax(1) == 0].astype(float)
    setups["theta2_beta_llm"] = ("ols", (X[te], s[te]), (Xq, ls), (Xqh, (ls > .5).astype(float)))
out = {}
for key, (kind, truth, soft, hard) in setups.items():
    theta = truth.mean() if kind == "mean" else ols(*truth)[0][1]
    out[key] = {"theta": theta, "N": len(truth[1]) if kind == "ols" else len(truth)}
    Ntot = out[key]["N"]
    for n in [m for m in NS if m <= Ntot // 3]:
        acc = {m: [] for m in ["classical", "naive", "ppi", "ppi_tuned"]}
        for _ in range(R):
            g = rng.choice(Ntot, n, replace=False); u = np.setdiff1d(np.arange(Ntot), g)
            if kind == "mean":
                r = {"classical": mean_classical(truth[g]), "naive": mean_naive(hard),
                     "ppi": mean_ppi(truth[g], soft[g], soft[u]), "ppi_tuned": mean_ppi(truth[g], soft[g], soft[u], tuned=True)}
            else:
                (Xt, yt), (Xs, ys), (Xhh, yhh) = truth, soft, hard
                b, se = ols(Xt[g], yt[g]); bn, sen = ols(Xhh, yhh); bp, sep = ols_ppi(Xt[g], yt[g], Xs[g], ys[g], Xs[u], ys[u])
                r = {"classical": (b[1], se[1]), "naive": (bn[1], sen[1]), "ppi": (bp[1], sep[1]), "ppi_tuned": (bp[1], sep[1])}
            for m, (e, se) in r.items():
                acc[m].append((e, se, abs(e - theta) <= Z * se))
        out[key][n] = {m: {"bias": float(np.mean([a[0] for a in v]) - theta), "rmse": float(np.sqrt(np.mean([(a[0] - theta) ** 2 for a in v]))),
                           "width": float(np.mean([2 * Z * a[1] for a in v])), "coverage": float(np.mean([a[2] for a in v]))} for m, v in acc.items()}
        print(key, n, {m: (round(x["coverage"], 3), round(x["width"], 4), round(x["bias"], 4)) for m, x in out[key][n].items()}, flush=True)
json.dump(out, open(f"{OUT}/E3_results.json", "w"), indent=1, default=float)
