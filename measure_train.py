"""E1 measurement models trained on the training split; also writes cross-fitted (5-fold,
out-of-fold) probabilities over each full corpus for prediction-powered inference."""
import time, json, numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from common import load_task, OUT
import lexicon
SEED = 2026
def corpus(task):
    if task in ("commit", "specific", "sentiment"):
        d = load_task("joint"); y = d[task].to_numpy(); emb = "joint"
    else:
        d = load_task(task); y = d.y.to_numpy(); emb = task
    E = {k: np.load(f"{OUT}/emb_{k}_{emb}.npy") for k in ("minilm", "climatebert")}
    return d.text.tolist(), y, d.split.to_numpy(), E
def models():
    lr = lambda: LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], cv=5, max_iter=4000, random_state=SEED)
    return {
      "lexicon": ("lex", lambda: make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, max_iter=2000))),
      "tfidf_lr": ("text", lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr())),
      "tfidf_svm": ("text", lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                                   CalibratedClassifierCV(LinearSVC(C=0.5, random_state=SEED), cv=5))),
      "minilm_lr": ("minilm", lambda: make_pipeline(StandardScaler(), lr())),
      "climatebert_lr": ("climatebert", lambda: make_pipeline(StandardScaler(), lr())),
    }
def feats(kind, task, texts, E, idx):
    if kind == "text": return [texts[i] for i in idx]
    if kind == "lex": return lexicon.scores(task, [texts[i] for i in idx]).reshape(-1, 1)
    return E[kind][idx]
timing = {}
for task in ["claim", "commit", "specific", "target", "sentiment"]:
    texts, y, split, E = corpus(task)
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    for name, (kind, mk) in models().items():
        if task == "sentiment" and name not in ("climatebert_lr", "tfidf_lr"): continue
        t0 = time.time(); m = mk().fit(feats(kind, task, texts, E, tr), y[tr]); fit = time.time() - t0
        t0 = time.time(); P = m.predict_proba(feats(kind, task, texts, E, te)); inf = (time.time() - t0) / len(te)
        np.save(f"{OUT}/p_{name}_{task}.npy", P); timing[f"{name}|{task}"] = [fit, inf]
        if name in ("climatebert_lr", "tfidf_lr", "lexicon"):          # cross-fitted predictions over the full corpus
            oof = np.zeros((len(y), len(np.unique(y))))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(texts, y):
                oof[b] = mk().fit(feats(kind, task, texts, E, a), y[a]).predict_proba(feats(kind, task, texts, E, b))
            np.save(f"{OUT}/oof_{name}_{task}.npy", oof)
        print(task, name, f"fit {fit:.1f}s", flush=True)
json.dump(timing, open(f"{OUT}/timing_measure.json", "w"), indent=1)
