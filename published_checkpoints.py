"""Score official test splits with the task-specific ClimateBERT checkpoints released by the
corpus authors (strongest published comparators). Not used for the target task, whose released
checkpoint was trained on a split that cannot be reconstructed."""
import time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import load_task, OUT
torch.set_num_threads(2)
CK = {"claim": "climatebert/environmental-claims",
      "commit": "climatebert/distilroberta-base-climate-commitment",
      "specific": "climatebert/distilroberta-base-climate-specificity"}
for task, repo in CK.items():
    d = load_task(task); d = d[d.split == "test"]
    tok = AutoTokenizer.from_pretrained(repo); mdl = AutoModelForSequenceClassification.from_pretrained(repo).eval()
    print(task, mdl.config.id2label, flush=True)
    P, t0 = [], time.time()
    for i in range(0, len(d), 16):
        b = tok(d.text.tolist()[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            P.append(torch.softmax(mdl(**b).logits, -1).numpy())
    P = np.vstack(P)
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    pos = lab.get("yes", lab.get("specific", 1))           # column of the positive class
    np.save(f"{OUT}/pub_{task}.npy", np.c_[1 - P[:, pos], P[:, pos]])
    print(task, P.shape, f"{time.time()-t0:.0f}s", flush=True)
