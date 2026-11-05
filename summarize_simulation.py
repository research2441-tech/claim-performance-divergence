"""Summaries of the Monte Carlo replications: bias, RMSE, coverage and rejection rates."""
import json, numpy as np
from common import OUT
Z = 1.959963984540054
raw = json.load(open(f"{OUT}/sim_raw.json")); tr = raw["truth"]
def summ(reps, world, y, est):
    v = np.array([r[y][est] for r in reps], float); v = v[~np.isnan(v).any(1)]
    th = 0.0 if est == "never" else tr[world][y]
    cov = np.mean(np.abs(v[:, 0] - th) <= Z * v[:, 1]); rej = np.mean(np.abs(v[:, 0]) > Z * v[:, 1])
    return {"mean": v[:, 0].mean(), "bias": v[:, 0].mean() - th, "rmse": np.sqrt(np.mean((v[:, 0] - th) ** 2)),
            "sd": v[:, 0].std(), "se": v[:, 1].mean(), "coverage": cov, "reject": rej, "truth": th, "R": len(v)}
EST = ["oracle_cs", "oracle_twfe", "plugin_cs", "plugin_twfe", "gold_cs", "ppi_cs", "ppit_cs"]
out = {"truth": tr, "rates": raw["rates"], "best": raw.get("best"), "main": {}, "gate": {}, "sens": {}, "share": {}}
for w, reps in raw["main"].items():
    out["main"][w] = {y: {e: summ(reps, w, y, e) for e in EST} for y in "AMST"}
    sh = np.array([[r["share"]["claim"], r["share"]["perf"]] for r in reps], float)
    out["share"][w] = {"claim_median": float(np.nanmedian(sh[:, 0])), "claim_iqr": np.nanpercentile(sh[:, 0], [25, 75]).tolist(),
                       "perf_median": float(np.nanmedian(sh[:, 1])), "perf_iqr": np.nanpercentile(sh[:, 1], [25, 75]).tolist()}
for w, reps in raw["ablation_gate"].items():
    out["gate"][w] = {y: {e: summ(reps, w, y, e) for e in ("plugin_cs", "ppi_cs")} for y in "AMS"}
for k, reps in raw["sens"].items():
    out["sens"][k] = {y: {e: summ(reps, "quiet", y, e) for e in ("plugin_cs", "ppi_cs", "ppit_cs", "gold_cs")} for y in "MA"}
json.dump(out, open(f"{OUT}/sim_summary.json", "w"), indent=1, default=float)
for w in out["main"]:
    for y in "AMST":
        print(w, y, "truth", round(tr[w][y], 4), {e: (round(s["bias"], 3), round(s["coverage"], 2), round(s["reject"], 2)) for e, s in out["main"][w][y].items()})
print(out["share"])
