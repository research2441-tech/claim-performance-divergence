"""Frozen sentence representations from a general and a climate-domain encoder (mean pooling)."""
import sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from common import load_task, OUT
torch.set_num_threads(2)
ENC = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "climatebert": "climatebert/distilroberta-base-climate-f"}
texts = {}
for t in ["claim", "joint", "target"]:
    texts[t] = load_task(t).text.tolist()
for name, repo in ENC.items():
    tok = AutoTokenizer.from_pretrained(repo); mdl = AutoModel.from_pretrained(repo).eval()
    for t, tx in texts.items():
        t0 = time.time(); embs = []
        for i in range(0, len(tx), 32):
            b = tok(tx[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).numpy())
        E = np.vstack(embs).astype(np.float32); np.save(f"{OUT}/emb_{name}_{t}.npy", E)
        print(name, t, E.shape, f"{time.time() - t0:.0f}s", flush=True)
