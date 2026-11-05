"""Result figures (grayscale, column width). Legends are placed outside the data area."""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import OUT
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "mathtext.fontset": "stix", "legend.frameon": False})
NAMES = {"lexicon": "Dictionary", "tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "minilm_lr": "General encoder",
         "climatebert_lr": "Climate encoder", "llm_zeroshot": "Zero-shot LLM", "published_ft": "Published fine-tuned"}
TASKN = {"claim": "(a) Environmental claim", "commit": "(b) Commitment", "specific": "(c) Specificity", "target": "(d) Target type"}
def fig_e1():
    r = json.load(open(f"{OUT}/E1_results.json")); order = [m for m in NAMES]
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.3), sharex=False)
    for ax, t in zip(axs.flat, TASKN):
        ms = list(order); y = np.arange(len(ms))[::-1]
        for yy, m in zip(y, ms):
            if m not in r[t]:
                ax.text(0.34, yy, "not available", va="center", fontsize=5.6, color="0.35", style="italic"); continue
            x = r[t][m]; ax.plot([x["F1lo"], x["F1hi"]], [yy, yy], color="0.35", lw=0.9)
            ax.plot(x["macroF1"], yy, "o", ms=3.6, mfc="black" if x["is_best"] else "white", mec="black", mew=0.8)
        ax.set_yticks(y); ax.set_yticklabels([NAMES[m] for m in ms], fontsize=6.2); ax.set_ylim(-0.6, len(ms) - 0.4)
        ax.set_title(TASKN[t], fontsize=7, pad=3); ax.set_xlim(0.3, 1.0); ax.grid(axis="x", color="0.88", lw=0.5)
        ax.tick_params(axis="x", labelsize=6.2)
        if t in ("claim", "commit"): pass
    for ax in axs[1]: ax.set_xlabel("Test macro-F1 (95% bootstrap CI)", fontsize=6.6)
    for ax in axs[:, 1]: ax.set_yticklabels([])
    fig.tight_layout(w_pad=0.6, h_pad=0.9); fig.savefig(f"{FIG}/fig2_annotators.png", dpi=600); plt.close(fig)
def fig_e2():
    r = json.load(open(f"{OUT}/E2_results.json")); ks = [k for k in r if not k.startswith("_")]
    ks = sorted(ks, key=lambda k: r[k][0]); fig, ax = plt.subplots(figsize=(3.5, 1.9)); y = np.arange(len(ks))
    for yy, k in zip(y, ks):
        a, lo, hi = r[k]; ax.plot([lo, hi], [yy, yy], color="0.35", lw=1.0)
        ax.plot(a, yy, "s" if k == "Cross-fitted encoder" else "o", ms=4, mfc="black" if k == "Cross-fitted encoder" else "white", mec="black", mew=0.8)
    ax.axvline(0.5, color="0.4", ls=(0, (3, 2)), lw=0.7); ax.text(0.505, len(ks) - 0.45, "chance", fontsize=6, color="0.3", va="center")
    ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=6.4); ax.set_xlim(0.35, 1.0); ax.set_ylim(-0.6, len(ks) - 0.1)
    ax.set_xlabel("AUROC for human-coded specificity (95% CI)", fontsize=6.6); ax.grid(axis="x", color="0.88", lw=0.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_proxies.png", dpi=600); plt.close(fig)
def fig_e3():
    r = json.load(open(f"{OUT}/E3_results.json"))
    panels = [("theta1_enc", "(a) Unverifiable-commitment share"), ("theta3_claim", "(b) Environmental-claim share"),
              ("theta4_netzero", "(c) Net-zero target share"), ("theta2_beta_commit", "(d) Commitment coefficient")]
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.0)); ns = [50, 100, 200, 400]
    sty = {"classical": ("o", "white", "-", "Gold only"), "ppi": ("s", "black", "-", "Rectified (PPI)"), "ppi_tuned": ("^", "0.6", (0, (2, 1.5)), "Power-tuned PPI")}
    for ax, (k, ttl) in zip(axs.flat, panels):
        for m, (mk, fc, ls, lab) in sty.items():
            if k.startswith("theta2") and m == "ppi_tuned": continue
            ax.plot(ns, [r[k][str(n)][m]["width"] for n in ns], ls=ls, color="black", lw=0.8, marker=mk, ms=3.4, mfc=fc, mec="black", mew=0.7, label=lab)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(ns); ax.set_xticklabels(ns, fontsize=6.2); ax.tick_params(axis="y", labelsize=6.2)
        ax.set_title(ttl, fontsize=6.8, pad=3); ax.grid(color="0.9", lw=0.5); ax.set_ylim(bottom=0)
    for ax in axs[1]: ax.set_xlabel("Gold sample size $n$", fontsize=6.6)
    for ax in axs[:, 0]: ax.set_ylabel("Mean 95% CI width", fontsize=6.6)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8, h_pad=0.8); fig.savefig(f"{FIG}/fig4_ppi_width.png", dpi=600); plt.close(fig)
def fig_sim():
    raw = json.load(open(f"{OUT}/sim_raw.json")); tr = raw["truth"]
    est = [("plugin_twfe", "Plug-in\nTWFE"), ("plugin_cs", "Plug-in\ngroup-time"), ("gold_cs", "Gold\nonly"), ("ppit_cs", "Rectified\ngroup-time")]
    cells = [("closing", "M", "(a) Closing the gap: claim mass $M$"), ("closing", "S", r"(b) Closing the gap: shortfall $\bar{S}$"),
             ("quiet", "M", "(c) Going quiet: claim mass $M$"), ("quiet", "S", r"(d) Going quiet: shortfall $\bar{S}$")]
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.6))
    for ax, (w, y, ttl) in zip(axs.flat, cells):
        data = [[r[y][e][0] for r in raw["main"][w] if r[y][e][0] == r[y][e][0]] for e, _ in est]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.9),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        for i, b in enumerate(bp["boxes"]): b.set_facecolor(["0.85", "0.85", "white", "0.55"][i])
        ax.axhline(tr[w][y], color="black", ls=(0, (3, 2)), lw=0.8); ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels([l for _, l in est], fontsize=5.6); ax.tick_params(axis="y", labelsize=6.2)
        ax.set_title(ttl, fontsize=6.6, pad=3)
    for ax in axs[:, 0]: ax.set_ylabel("Estimated assurance effect", fontsize=6.4)
    fig.text(0.5, 0.005, "Dashed line: true effect from a ten-times larger panel; 400 replications per world", ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.7, h_pad=0.9); fig.savefig(f"{FIG}/fig5_simulation.png", dpi=600); plt.close(fig)
def fig_sens():
    s = json.load(open(f"{OUT}/sim_summary.json"))["sens"]; qs, ds = [0.1, 0.2, 0.3], [0.0, 0.05, 0.1]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, e, ttl in zip(axs, ("plugin_cs", "ppit_cs"), ("(a) Plug-in group-time", "(b) Rectified group-time")):
        M = np.array([[s[f"{q}|{d}"]["M"][e]["coverage"] for d in ds] for q in qs])
        ax.imshow(M, cmap="Greys", vmin=0, vmax=2.4, aspect="auto")
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.4, color="black")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{d:.2f}" for d in ds], fontsize=6.2); ax.set_yticks(range(3))
        ax.set_yticklabels([f"{q:.2f}" for q in qs], fontsize=6.2); ax.set_xlabel(r"Post-assurance shift $\Delta_{se}$", fontsize=6.4)
        ax.set_title(ttl, fontsize=6.7, pad=3)
    axs[0].set_ylabel("Gold share $q$", fontsize=6.4)
    fig.tight_layout(w_pad=1.0); fig.savefig(f"{FIG}/fig6_sensitivity.png", dpi=600); plt.close(fig)
def fig_shares():
    raw = json.load(open(f"{OUT}/sim_raw.json")); tr = raw["truth"]
    worlds = [("closing", "Closing\nthe gap"), ("mixed", "Mixed"), ("quiet", "Going\nquiet")]
    fig, ax = plt.subplots(figsize=(3.5, 2.0)); pos = []
    for k, (w, lab) in enumerate(worlds):
        for j, key in enumerate(("claim", "perf")):
            v = np.array([r["share"][key] for r in raw["main"][w]], float); v = v[np.isfinite(v)]
            b = ax.boxplot([v], positions=[k * 3 + j], widths=0.7, patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
            b["boxes"][0].set_facecolor("0.45" if key == "claim" else "white")
            t = tr[w]["share_claim" if key == "claim" else "share_perf"]
            ax.plot([k * 3 + j - 0.42, k * 3 + j + 0.42], [t, t], color="black", ls=(0, (2, 1.5)), lw=0.9)
    ax.set_xticks([0.5, 3.5, 6.5]); ax.set_xticklabels([l for _, l in worlds], fontsize=6.4); ax.axhline(0, color="0.6", lw=0.4)
    ax.set_ylim(-1.2, 2.2); ax.set_ylabel("Share of change in divergence", fontsize=6.6); ax.tick_params(axis="y", labelsize=6.2)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(fc="0.45", ec="black", lw=0.6, label=r"Claim share $\pi_Q$"), Patch(fc="white", ec="black", lw=0.6, label=r"Performance share $\pi_C$"),
                       Line2D([0], [0], color="black", ls=(0, (2, 1.5)), lw=0.9, label="True share")], loc="upper center", ncol=3, fontsize=6.2,
              bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{FIG}/fig7_shares.png", dpi=600); plt.close(fig)

def fig_event():
    r = json.load(open(f"{OUT}/sim_event.json"))
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.85))
    for ax, (w, ttl) in zip(axs, (("quiet", "(a) Going quiet: claim mass $M$"), ("closing", r"(b) Closing the gap: shortfall $\bar{S}$"))):
        x = r[w]; ev = np.array(x["ev"])
        ax.fill_between(ev, x["ppi_lo"], x["ppi_hi"], color="0.85", lw=0, label="Rectified 95% range")
        ax.plot(ev, x["truth"], color="black", lw=1.0, ls=(0, (3, 2)), label="True effect")
        ax.plot(ev, x["ppi"], "s-", color="black", lw=0.8, ms=2.6, mfc="black", label="Rectified")
        ax.plot(ev, x["plugin"], "o-", color="0.45", lw=0.8, ms=2.6, mfc="white", label="Plug-in")
        ax.axvline(-0.5, color="0.5", lw=0.5); ax.axhline(0, color="0.6", lw=0.4)
        ax.set_title(ttl, fontsize=6.5, pad=3); ax.tick_params(labelsize=6); ax.set_xlabel("Years since assurance", fontsize=6.3); ax.set_xticks([-3, -2, 0, 1, 2, 3, 4, 5, 6, 7])
    axs[0].set_ylabel("Effect, 2018 cohort", fontsize=6.3)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.8); fig.savefig(f"{FIG}/fig8_event.png", dpi=600); plt.close(fig)

def fig_calib():
    from common import load_task
    d = load_task("joint"); te = d[d.split == "test"]
    files = {"climatebert_lr": "p_climatebert_lr_{}.npy", "tfidf_lr": "p_tfidf_lr_{}.npy", "published_ft": "pub_{}.npy", "llm_zeroshot": "llm_Qwen2.5-0.5B-Instruct_test_{}.npy"}
    sty = {"climatebert_lr": ("s", "black", "-"), "tfidf_lr": ("^", "0.5", "-"), "published_ft": ("o", "white", "-"), "llm_zeroshot": ("D", "0.75", (0, (2, 1.5)))}
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.3))
    for ax, t, ttl in zip(axs, ("commit", "specific"), ("(a) Commitment", "(b) Specificity")):
        y = te[t].to_numpy()
        for m, f in files.items():
            p = np.load(f"{OUT}/" + f.format(t))[:, 1]; bins = np.linspace(0, 1, 6); idx = np.clip(np.digitize(p, bins) - 1, 0, 4)
            xs = [p[idx == k].mean() for k in range(5) if (idx == k).sum() >= 8]; ys = [y[idx == k].mean() for k in range(5) if (idx == k).sum() >= 8]
            mk, fc, ls = sty[m]
            ax.plot(xs, ys, marker=mk, ls=ls, color="black", lw=0.7, ms=3, mfc=fc, mec="black", mew=0.6, label=NAMES[m])
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=6.6, pad=3); ax.tick_params(labelsize=6); ax.set_xlabel("Mean predicted probability", fontsize=6.2)
    axs[0].set_ylabel("Observed share (human)", fontsize=6.2)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, fontsize=5.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.0, 1, 0.87), w_pad=1.0); fig.savefig(f"{FIG}/fig9_calibration.png", dpi=600); plt.close(fig)

def fig_cov():
    r = json.load(open(f"{OUT}/E3_results.json"))
    lab = {"theta1_enc": "Unverif. share, encoder", "theta1_lex": "Unverif. share, dictionary", "theta1_llm": "Unverif. share, LLM",
           "theta3_claim": "Claim share, encoder", "theta4_netzero": "Net-zero share, encoder", "theta2_beta_commit": "Coefficient, encoder",
           "theta2_beta_lex": "Coefficient, dictionary", "theta2_beta_llm": "Coefficient, LLM"}
    ks = [k for k in lab if k in r]; y = np.arange(len(ks))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.2)); h = 0.26
    for off, m, fc, name in ((h, "classical", "white", "Gold only"), (0, "naive", "0.75", "Plug-in"), (-h, "ppi", "0.25", "Rectified (PPI)")):
        vals = [r[k]["100"][m]["coverage"] for k in ks]
        ax.barh(y + off, vals, height=h, color=fc, edgecolor="black", lw=0.5, label=name)
        for yy, v in zip(y + off, vals):
            if v < 0.05: ax.text(0.012, yy, "0.00 (plug-in)", va="center", fontsize=5.4, color="black")
    ax.axvline(0.95, color="black", ls=(0, (3, 2)), lw=0.8); ax.set_xlim(0, 1.0); ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in ks], fontsize=6.0)
    ax.set_xlabel("Empirical coverage of nominal 95% intervals, $n=100$", fontsize=6.4); ax.tick_params(axis="x", labelsize=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.14), ncol=3, fontsize=6.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_coverage.png", dpi=600); plt.close(fig)

if __name__ == "__main__":
    for f in sys.argv[2:] or ["e1", "e2", "e3", "sim", "sens", "shares", "event", "calib", "cov"]:
        {"e1": fig_e1, "e2": fig_e2, "e3": fig_e3, "sim": fig_sim, "sens": fig_sens, "shares": globals()["fig_shares"], "event": globals()["fig_event"], "calib": globals()["fig_calib"], "cov": globals()["fig_cov"]}[f](); print("ok", f)
