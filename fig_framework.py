"""Fig. 1: framework diagram (grayscale). Boxes are placed on a grid and checked for overlap."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import itertools, sys
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 6.6, "mathtext.fontset": "stix"})
W, H = 3.5, 5.3
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 152); ax.axis("off")
B = {}  # name: (x, y, w, h)
def box(name, x, y, w, h, text, fill="#FFFFFF", bold=False, lw=0.8, ls="-"):
    B[name] = (x, y, w, h)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fill, ec="black", lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontweight="bold" if bold else "normal", linespacing=1.25)
def anchor(name, side, frac=0.5):
    x, y, w, h = B[name]
    return {"top": (x + w * frac, y + h + 0.4), "bottom": (x + w * frac, y - 0.4), "left": (x - 0.4, y + h * frac), "right": (x + w + 0.4, y + h * frac)}[side]
def arrow(p, q, dashed=False, rad=0.0, label=None, loff=(0, 0)):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="black", ls=(0, (3, 2)) if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label: ax.text((p[0] + q[0]) / 2 + loff[0], (p[1] + q[1]) / 2 + loff[1], label, ha="center", va="center", fontsize=5.8, style="italic",
                      bbox=dict(fc="white", ec="none", pad=0.6))
def poly(points, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color="black", lw=0.8, ls=(0, (3, 2)) if dashed else "-")
    arrow(points[-2], points[-1], dashed=dashed)
# spine x=24..72; left column x=1..17; right column x=78..99
box("rep", 24, 136, 48, 11, "Management reports $R_{it}$\n(sustainability statement)", fill="#EDEDED")
box("ann", 24, 117, 48, 12, "Codebook-constrained annotator $f$\n(encoder probe or zero-shot LLM)")
box("gate", 24, 99, 48, 11, "Verifiability gate $v_c$\n(commitment, specific, scope, tuple)")
box("unv", 78, 99, 21, 11, "Unverifiable\nclaims $U_{it}$")
box("ets", 1, 80, 16, 12, "Verified\nemissions\n$\\ell_{it}$", fill="#EDEDED")
box("tup", 24, 80, 48, 12, "Tuples, annual rate $\\rho_c$,\nannounced path and shortfall $s_{c,t}$")
box("stat", 24, 62, 48, 11, "Firm-year statistics\n$M_{it}$, $A_{it}$, $\\bar{S}_{it}$, $Q_{it}$")
box("gold", 1, 44, 16, 11, "Human gold\ncoding $\\mathcal{G}$", fill="#EDEDED")
box("ppi", 24, 44, 48, 11, "Rectified firm statistics\n$\\hat{a}_i^{g}$ corrected by $a_i^{g}-\\hat{a}_i^{g}$")
box("ass", 78, 26, 21, 11, "Assurance\ntiming $G_i$", fill="#EDEDED")
box("cs", 24, 26, 48, 11, "Staggered group-time estimator\n$\\hat{\\theta}^{\\mathrm{PP}}$ with 95% intervals")
box("dec", 24, 4, 48, 15, "Channel decomposition\nclaim channel $\\pi_Q$: going quiet\nperformance channel $\\pi_C$: closing the gap", lw=1.2)
arrow(anchor("rep", "bottom"), anchor("ann", "top"))
arrow(anchor("ann", "bottom"), anchor("gate", "top"))
arrow(anchor("gate", "bottom"), anchor("tup", "top"), label="$v_c=1$", loff=(5.5, 0))
arrow(anchor("gate", "right"), anchor("unv", "left"))
arrow(anchor("ets", "right"), anchor("tup", "left"))
arrow(anchor("tup", "bottom"), anchor("stat", "top"))
poly([anchor("unv", "bottom"), (88.5, 67.5), anchor("stat", "right")])
arrow(anchor("stat", "bottom"), anchor("ppi", "top"))
arrow(anchor("gold", "right"), anchor("ppi", "left"), dashed=True)
arrow(anchor("ppi", "bottom"), anchor("cs", "top"))
arrow(anchor("ass", "left"), anchor("cs", "right"))
arrow(anchor("cs", "bottom"), anchor("dec", "top"))
# overlap check (with a 1-unit margin)
bad = [(a, b) for a, b in itertools.combinations(B, 2)
       if not (B[a][0] + B[a][2] + 1 <= B[b][0] or B[b][0] + B[b][2] + 1 <= B[a][0] or B[a][1] + B[a][3] + 1 <= B[b][1] or B[b][1] + B[b][3] + 1 <= B[a][1])]
if bad: sys.exit(f"overlapping boxes: {bad}")
fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
for t in ax.texts:
    bb = inv.transform(t.get_window_extent(r).get_points())
    for n, (x, y, w, h) in B.items():
        cx, cy = t.get_position()
        if x <= cx <= x + w and y <= cy <= y + h and (bb[0][0] < x + 0.3 or bb[1][0] > x + w - 0.3 or bb[0][1] < y or bb[1][1] > y + h):
            sys.exit(f"text overflows box {n}: {t.get_text()!r}")
out = sys.argv[1] if len(sys.argv) > 1 else "fig1_framework.png"
fig.savefig(out, dpi=600); print("ok", out)
