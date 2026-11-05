"""E1 test-split comparison with bootstrap CIs and Holm-adjusted paired tests, and E2 proxy validity
on the 1,320 paragraphs that carry human commitment, specificity and tone labels."""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import chi2_contingency
from common import load_task, OUT
from metrics import all_metrics, boot_ci, paired_boot_p, holm
import lexicon
TASKS = ["claim", "commit", "specific", "target"]
MODELS = ["lexicon", "tfidf_lr", "tfidf_svm", "minilm_lr", "climatebert_lr", "llm_zeroshot", "published_ft"]
def ytest(task):
    d = load_task("joint") if task in ("commit", "specific") else load_task(task)
    col = task if task in ("commit", "specific") else "y"
    return d[d.split == "test"][col].to_numpy(), d[d.split == "test"].text.tolist()
def probs(model, task):
    f = {"llm_zeroshot": f"llm_Qwen2.5-0.5B-Instruct_test_{task}.npy", "published_ft": f"pub_{task}.npy"}.get(model, f"p_{model}_{task}.npy")
    f = os.path.join(OUT, f); return np.load(f) if os.path.exists(f) else None
rows, res = [], {}
for task in TASKS:
    y, _ = ytest(task); res[task] = {}
    avail = {m: probs(m, task) for m in MODELS}; avail = {m: P for m, P in avail.items() if P is not None}
    best = max(avail, key=lambda m: all_metrics(avail[m], y)["macroF1"])
    pv = {}
    for m, P in avail.items():
        r = all_metrics(P, y); lo, hi = boot_ci(P, y, "macroF1", B=1000)
        r.update(F1lo=lo, F1hi=hi)
        if m != best: pv[m] = paired_boot_p(avail[best], P, y, B=1000)[1]
        res[task][m] = r
    adj = dict(zip(pv, holm(list(pv.values())))) if pv else {}
    for m in avail:
        res[task][m]["p_holm_vs_best"] = adj.get(m, np.nan); res[task][m]["is_best"] = m == best
        cm = confusion_matrix(y, avail[m].argmax(1)); res[task][m]["cm"] = cm.tolist()
json.dump(res, open(f"{OUT}/E1_results.json", "w"), indent=1, default=float)
# ---------- E2: proxy validity against human labels ----------
d = load_task("joint"); tx = d.text.tolist()
prox = {"Flesch reading ease": np.array([lexicon.flesch(t) for t in tx]),
        "Numeric density": lexicon.numeric_density(tx),
        "Vague-word rate": -lexicon.vague_rate(tx),
        "Opportunity tone (human)": (d.sentiment == 2).astype(float).to_numpy(),
        "Length (words)": np.array([len(t.split()) for t in tx], float),
        "Heuristic greenwash flag": -lexicon.heuristic_greenwash_flag(tx)}
oof = np.load(f"{OUT}/oof_climatebert_lr_specific.npy")[:, 1]; prox["Cross-fitted encoder"] = oof
E2 = {}
rng = np.random.default_rng(3)
for k, v in prox.items():
    a = roc_auc_score(d.specific, v); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(v), len(v)); bs.append(roc_auc_score(d.specific.iloc[i], v[i]))
    E2[k] = [a, *np.percentile(bs, [2.5, 97.5])]
ct = pd.crosstab(d.commit, d.specific); chi = chi2_contingency(ct)
tone = pd.crosstab(d.sentiment, d.specific); chi2t = chi2_contingency(tone)
cv = lambda tab, c: np.sqrt(c[0] / (tab.values.sum() * (min(tab.shape) - 1)))
E2["_cells"] = ct.values.tolist(); E2["_chi_commit_spec"] = [chi[0], chi[1], cv(ct, chi)]
E2["_tone_spec"] = tone.values.tolist(); E2["_chi_tone_spec"] = [chi2t[0], chi2t[1], cv(tone, chi2t)]
flag = lexicon.heuristic_greenwash_flag(tx)
E2["_flag_rate"] = flag.mean(); E2["_flag_by_cell"] = {f"{k[0]}{k[1]}": v for k, v in d.assign(f=flag).groupby(["commit", "specific"]).f.mean().to_dict().items()}
json.dump(E2, open(f"{OUT}/E2_results.json", "w"), indent=1, default=str)
print(json.dumps({t: {m: round(r["macroF1"], 3) for m, r in res[t].items()} for t in res}, indent=0))
print({k: np.round(v, 3).tolist() for k, v in E2.items() if not k.startswith("_")}); print(E2["_cells"], E2["_chi_commit_spec"], E2["_chi_tone_spec"])
