"""Download the human-labelled ClimateBERT corpora and write one tidy CSV per task."""
import os, pandas as pd
from datasets import load_dataset
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
TASKS = {"claim": "climatebert/environmental_claims",
         "commit": "climatebert/climate_commitments_actions",
         "specific": "climatebert/climate_specificity",
         "sentiment": "climatebert/climate_sentiment",
         "detect": "climatebert/climate_detection"}
for t, repo in TASKS.items():
    ds = load_dataset(repo)
    frames = []
    for split in ds:
        df = ds[split].to_pandas(); df["split"] = split; frames.append(df)
    df = pd.concat(frames, ignore_index=True)[["text", "label", "split"]]
    df.to_csv(os.path.join(OUT, f"{t}.csv"), index=False)
    print(t, len(df), df.groupby("split").size().to_dict(), df.label.value_counts().to_dict())
import urllib.request
url = "https://huggingface.co/datasets/climatebert/netzero_reduction_data/resolve/main/targets_final.csv"
dst = os.path.join(OUT, "targets_final.csv")
if not os.path.exists(dst): urllib.request.urlretrieve(url, dst)
tg = pd.read_csv(dst)[["text", "target"]]
print("target raw", len(tg), "unique texts", tg.text.nunique())
