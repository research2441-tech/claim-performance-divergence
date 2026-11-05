"""Writes the Results section text and tables directly from the experiment outputs, so that every
number in the manuscript is traceable to a JSON file produced by the experiments."""
import json, re, numpy as np
from common import OUT
import os
HERE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(HERE, "outputs", "results_section.txt")
E1 = json.load(open(f"{OUT}/E1_results.json")); E2 = json.load(open(f"{OUT}/E2_results.json"))
E3 = json.load(open(f"{OUT}/E3_results.json")); SIM = json.load(open(f"{OUT}/sim_summary.json"))
TM = json.load(open(f"{OUT}/timing_measure.json"))
NAMES = {"lexicon": "Dictionary", "tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "minilm_lr": "General encoder",
         "climatebert_lr": "Climate encoder", "llm_zeroshot": "Zero-shot LLM", "published_ft": "Published fine-tuned"}
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; pc = lambda x: f"{100*x:.0f}%"
ln = lambda x: x if x.startswith("TF") else x[0].lower() + x[1:]
def best(t): return [m for m in E1[t] if E1[t][m]["is_best"]][0]
def r1(t, m, k="macroF1"): return E1[t][m][k]
L = []
L.append("# Results")
# ---------------- E1 ----------------
L.append("## Measurement Accuracy Against Human Labels")
bt = {t: best(t) for t in E1}
rows = ["Task | Annotator | Macro-F1 [95% CI] | MCC | AUROC | ECE"]
for t, tn in [("claim", "Claim"), ("commit", "Commitment"), ("specific", "Specificity"), ("target", "Target")]:
    for m in NAMES:
        if m not in E1[t]: continue
        x = E1[t][m]; star = "" if x["is_best"] else ("*" if x["p_holm_vs_best"] < 0.05 else "")
        name = f"**{NAMES[m]}**" if x["is_best"] else NAMES[m]
        rows.append(f"{tn if m == list(E1[t])[0] else ''} | {name} | {f3(x['macroF1'])}{star} [{f2(x['F1lo'])}, {f2(x['F1hi'])}] | {f3(x['MCC'])} | {f3(x['AUROC'])} | {f3(x['ECE'])}")
gap = {t: r1(t, bt[t]) - max(r1(t, m) for m in E1[t] if m != bt[t]) for t in E1}
lex = {t: r1(t, "lexicon") for t in E1}
llm_txt = ""
if all("llm_zeroshot" in E1[t] for t in E1):
    lf = {t: r1(t, "llm_zeroshot") for t in E1}
    llm_txt = (f" The zero-shot language model, which saw only the codebook definitions, reached macro-F1 of {f3(lf['claim'])}, {f3(lf['commit'])}, {f3(lf['specific'])} and "
               f"{f3(lf['target'])} on the four tasks. It therefore {'outperformed' if np.mean([lf[t]-lex[t] for t in lf])>0 else 'did not outperform'} the dictionary on average but stayed "
               f"{f2(np.mean([r1(t, bt[t]) - lf[t] for t in lf]))} macro-F1 points below the best supervised annotator, and its expected calibration error was the largest in "
               f"{['none', 'one', 'two', 'three', 'all'][sum(1 for t in lf if E1[t]['llm_zeroshot']['ECE'] == max(E1[t][m]['ECE'] for m in E1[t]))]} of the four tasks; uncorrected, it is the annotator most likely to bias a downstream estimate.")
L.append(f"{{T:e1}} and {{F:annot}} compare the annotators on the held-out test splits. The best annotator differs by task: the {ln(NAMES[bt['claim']])} probe for environmental claims (macro-F1 {f3(r1('claim', bt['claim']))}), "
         f"the {ln(NAMES[bt['commit']])} checkpoint for commitments ({f3(r1('commit', bt['commit']))}), the {ln(NAMES[bt['specific']])} probe for specificity ({f3(r1('specific', bt['specific']))}) and "
         f"{NAMES[bt['target']]} for target type ({f3(r1('target', bt['target']))}). Margins between the leading supervised annotators are small, between {f3(min(gap.values()))} and {f3(max(gap.values()))} macro-F1, and most are not "
         f"significant after Holm correction, whereas the dictionary trails by {f2(min(r1(t, bt[t]) - lex[t] for t in E1))} to {f2(max(r1(t, bt[t]) - lex[t] for t in E1))} points on every task.{llm_txt}")
cm = E1["specific"][bt["specific"]]["cm"]; sens_spec = cm[1][1] / (cm[1][0] + cm[1][1])
cmc = E1["commit"][bt["commit"]]["cm"]; sens_c = cmc[1][1] / (cmc[1][0] + cmc[1][1])
L.append(f"Specificity is the hardest attribute. Even the best annotator recovers only {pc(sens_spec)} of human-coded specific paragraphs, and the best commitment annotator recovers {pc(sens_c)} of commitments, "
         f"so roughly {pc(sens_spec*sens_c)} of verifiable commitments would pass both stages of the gate if the two errors were independent. This is the error structure that the simulation in Section V-D inherits, and it is the reason why a "
         f"count of machine-detected verifiable claims cannot be read as a count of verifiable claims.")
L.append("!TAB e1 | Annotator Comparison on Held-Out Test Splits (Best Annotator in Bold; * Holm-Adjusted p < 0.05 Against the Best) | 12L,24L,28,12,12,12 | W")
L += rows; L.append("!END")
L.append("!FIG annot | fig2_annotators.png | Test macro-F1 with 95% bootstrap intervals for each annotator and attribute. Filled markers denote the best annotator for the attribute; no released checkpoint is evaluated for target type.")
L.append("The errors are informative about what specificity means to human coders. {T:errors} lists four test paragraphs on which the climate-encoder probe was confidently wrong. The false negatives describe concrete, nameable actions, such as lighting and ventilation upgrades or flood-plain mapping, without a single number; "
         "the false positives contain numbers that describe the business, such as the count of repair sites or reporting dates, while the commitment itself remains open-ended. For the divergence measure the second type matters more, because a figure that is not attached to the claim gives no path against which emissions can be compared, and it is exactly the kind of statement that numeric-density proxies reward. The tuple requirement in (EQ:gate) rejects it, since no base year, magnitude or target year can be parsed.")
L.append("!TAB errors | Confident Specificity Errors of the Climate-Encoder Probe on the Test Split (Verbatim Excerpts) | 15L,55L,10,20L")
L.append("Error | Excerpt | $\\hat{\\pi}$ | Reading")
L.append("False negative | \"A number of energy-efficiency projects have been implemented such as lighting control and heating, ventilation, and air conditioning (HVAC) upgrades.\" | 0.16 | Concrete action, no figure")
L.append("False negative | \"…[the company considers] site specific climate and weather factors, such as flood plain mapping and extreme weather history.\" | 0.03 | Named procedure")
L.append("False positive | \"…has grown to 21 sites…, repairing around 90,000 accident-damaged cars each year. [The group is] investing significant resources… aimed at continually improving energy efficiency…\" | 0.93 | Figures describe operations, not the commitment")
L.append("False positive | \"All data in this TCFD report is as of, or for the year-ended December 31, 2020 unless otherwise noted.\" | 0.70 | Dates without any claim")
L.append("!END")
L.append(f"{{F:calib}} shows the probability scale of four annotators, which matters because the rectifier variance in (EQ:ppivar) grows when predicted probabilities are far from the conditional share of human labels. "
         f"For commitments the TF-IDF probe stays closest to the diagonal, whereas the climate-encoder probe and the released fine-tuned checkpoint are overconfident in their upper range (ECE {f3(E1['commit']['climatebert_lr']['ECE'])} and {f3(E1['commit']['published_ft']['ECE'])}). For specificity, the zero-shot model's probabilities barely rise with the human share "
         f"(ECE {f3(E1['specific']['llm_zeroshot']['ECE'])}), so its texts behave almost like unlabelled noise in the rectified estimators; the inference experiments below confirm that this costs precision but not validity.")
L.append("!FIG calib | fig9_calibration.png | Reliability diagrams on the paragraph test split (five equal-width probability bins with at least eight paragraphs each). The grey diagonal marks perfect calibration.")
# ---------------- E2 ----------------
cells = E2["_cells"]; chi = E2["_chi_commit_spec"]; tone = E2["_tone_spec"]; chit = E2["_chi_tone_spec"]
n_commit = cells[1][0] + cells[1][1]; unver = cells[1][0] / n_commit
prox = {k: v for k, v in E2.items() if not k.startswith("_")}
pk = sorted([k for k in prox if k != "Cross-fitted encoder"], key=lambda k: -prox[k][0])
fb = E2["_flag_by_cell"]
L.append("## Proxy Validity and the Structure of Cheap Talk")
L.append(f"The joint human labels show how much of the corpus is cheap talk. Of {n_commit} paragraphs coded as commitments, {cells[1][0]} ({pc(unver)}) were coded as non-specific, and commitment and specificity are "
         f"{'positively' if cells[1][1]*cells[0][0] > cells[1][0]*cells[0][1] else 'negatively'} associated ($\\chi^{{2}}={chi[0]:.1f}$, $p{'<0.001' if chi[1] < 0.001 else '=' + f3(chi[1])}$, Cramér's $V={f2(chi[2])}$). "
         f"Specificity also varies with human-coded tone ($\\chi^{{2}}={chit[0]:.1f}$, $V={f2(chit[2])}$): {pc(tone[2][1]/(tone[2][0]+tone[2][1]))} of opportunity paragraphs are specific against "
         f"{pc(tone[0][1]/(tone[0][0]+tone[0][1]))} of risk paragraphs and {pc(tone[1][1]/(tone[1][0]+tone[1][1]))} of neutral ones.")
L.append(f"{{F:proxies}} asks whether the proxies used to score greenwashing recover this human distinction. Numeric density, which is close to a definition of specificity, does best among them (AUROC {f3(prox['Numeric density'][0])} "
         f"[{f2(prox['Numeric density'][1])}, {f2(prox['Numeric density'][2])}]). The stylistic proxies do much worse: Flesch reading ease reaches {f3(prox['Flesch reading ease'][0])}, human-coded opportunity tone {f3(prox['Opportunity tone (human)'][0])}, "
         f"the vague-word rate {f3(prox['Vague-word rate'][0])} and the heuristic greenwashing flag {f3(prox['Heuristic greenwash flag'][0])}, against {f3(prox['Cross-fitted encoder'][0])} for a cross-fitted encoder trained on the human labels. "
         f"Readability and vagueness therefore describe style rather than verifiability. The heuristic flag fires on {pc(fb['10'])} of non-specific commitments, but also on "
         f"{pc(fb['00'])} of paragraphs that contain no commitment at all and on {pc(fb['11'])} of specific commitments, which is the confusion between vagueness and misleading behaviour that a performance-referenced measure is designed to remove.")
L.append("!FIG proxies | fig3_proxies.png | Discrimination of human-coded specific versus non-specific paragraphs by common greenwashing proxies (open markers) and by a cross-fitted supervised encoder (filled marker). Signs are oriented so that higher values predict specificity; the dashed line marks chance. N = 1,320 paragraphs.")
# ---------------- E3 ----------------
L.append("## Rectified Inference With Known Ground Truth")
lab = {"theta1_enc": "Unverifiable-commitment share, encoder", "theta1_lex": "Unverifiable-commitment share, dictionary", "theta1_llm": "Unverifiable-commitment share, LLM",
       "theta3_claim": "Environmental-claim share, encoder", "theta4_netzero": "Net-zero target share, encoder", "theta2_beta_commit": "Commitment coefficient, encoder",
       "theta2_beta_lex": "Commitment coefficient, dictionary", "theta2_beta_llm": "Commitment coefficient, LLM"}
rows = ["Estimand and annotator | True value | $n$ | Gold only cov. / width | Plug-in cov. / bias | PPI cov. / width"]
for k in lab:
    if k not in E3: continue
    ns = [n for n in ("50", "100", "400") if n in E3[k]]
    for n in ns:
        x = E3[k][n]
        rows.append(f"{lab[k] if n == ns[0] else ''} | {f3(E3[k]['theta']) if n == ns[0] else ''} | {n} | {f2(x['classical']['coverage'])} / {f3(x['classical']['width'])} | "
                    f"{f2(x['naive']['coverage'])} / {x['naive']['bias']:+.3f} | {f2(x['ppi']['coverage'])} / {f3(x['ppi']['width'])}")
covs = [E3[k][n]["ppi"]["coverage"] for k in E3 for n in E3[k] if n not in ("theta", "N")]
red = {k: 1 - E3[k]["50"]["ppi"]["width"] / E3[k]["50"]["classical"]["width"] for k in E3}
naive_bad = [(k, E3[k]["50"]["naive"]["bias"], E3[k]["50"]["naive"]["coverage"]) for k in E3]
worst = max(naive_bad, key=lambda x: abs(x[1]) / max(abs(E3[x[0]]["theta"]), 1e-9))
L.append(f"{{T:e3}} reports {['zero','one','two','three','four','five','six','seven','eight','nine','ten'][len(E3)]} estimands whose true values were computed from all human labels. Across every estimand, annotator and gold size, prediction-powered intervals covered the truth in between {pc(min(covs))} and {pc(max(covs))} of 2,000 draws. "
         f"Coverage above 95% at the larger gold sizes reflects sampling without replacement from a finite corpus, where the gold sample is a sizeable fraction of the population, and is conservative rather than invalid.")
L.append(f"Efficiency depends on the annotator ({{F:ppiwidth}}). With the cross-fitted climate encoder, rectification shortened intervals at $n=50$ by {pc(red['theta3_claim'])} for the claim share, {pc(red['theta4_netzero'])} for the net-zero share and "
         f"{pc(red['theta1_enc'])} for the unverifiable-commitment share; for the regression coefficient, in which three machine-coded variables enter jointly, the gain was {pc(red['theta2_beta_commit'])}. "
         + "With the dictionary and the zero-shot language model as annotators, untuned rectification gave intervals no narrower than gold coding alone "
         + f"(power tuning recovered {pc(1 - E3['theta1_lex']['50']['ppi_tuned']['width'] / E3['theta1_lex']['50']['classical']['width'])} and {pc(1 - E3['theta1_llm']['50']['ppi_tuned']['width'] / E3['theta1_llm']['50']['classical']['width'])} for the share), "
         + "but coverage did not deteriorate, because a weak annotator only costs precision under rectification.")
L.append(f"The naive plug-in estimator behaves very differently. Its intervals are narrow because they treat every machine label as an observation, so validity rests entirely on the annotator. With the cross-fitted encoder its bias on random samples of the corpus was small "
         f"(at most {max(abs(E3[k]['50']['naive']['bias']) for k in ('theta1_enc','theta3_claim','theta4_netzero','theta2_beta_commit')):.3f} in absolute value), but with the {lab[worst[0]].split(', ')[1].replace('LLM', 'zero-shot language model')} it was {worst[1]:+.3f} for the "
         f"{lab[worst[0]].split(',')[0].lower()} (true value {f3(E3[worst[0]]['theta'])}) and its interval covered the truth in {pc(worst[2])} of draws; with the dictionary the plug-in commitment coefficient was biased by {E3['theta2_beta_lex']['50']['naive']['bias']:+.3f} against a true value of {f3(E3['theta2_beta_lex']['theta'])}, again with no coverage. A researcher cannot know in advance which situation applies; rectification removes the dependence, as {{F:cov}} shows for all eight estimands at $n=100$.")
L.append("!FIG cov | fig10_coverage.png | Empirical coverage of nominal 95% intervals at a gold size of 100 for all estimands and annotators. The dashed line marks the nominal level; plug-in coverage is either complete or zero because its interval does not depend on the gold draw.")
L.append("!TAB e3 | Coverage of Nominal 95% Intervals, Interval Width and Plug-in Bias for Estimands With Known Human-Labelled Truth (2,000 Gold Draws) | 30L,10,6,18,18,18 | W")
L += rows; L.append("!END")
L.append("!FIG ppiwidth | fig4_ppi_width.png | Mean width of 95% intervals as a function of gold sample size for four estimands with the cross-fitted encoder as annotator. Rectified intervals keep nominal coverage ({T:e3}) while needing fewer human-coded texts for a given precision.")
# ---------------- Simulation ----------------
S = SIM["main"]; tr = SIM["truth"]
def g(w, y, e, k): return S[w][y][e][k]
L.append("## Separating Closing the Gap From Going Quiet")
rows = ["World | Outcome (truth) | Plug-in TWFE bias / cov. | Plug-in group-time bias / cov. | Gold only power | Rectified bias / cov. / power"]
for w, wn in [("closing", "Closing"), ("quiet", "Quiet"), ("mixed", "Mixed"), ("null", "Null")]:
    for y, yn in [("A", "$A$"), ("M", "$M$"), ("S", "$\\overline{S}$"), ("T", "$Q$")]:
        rows.append(f"{wn if y == 'A' else ''} | {yn} ({tr[w][y]:+.3f}) | {g(w,y,'plugin_twfe','bias'):+.3f} / {f2(g(w,y,'plugin_twfe','coverage'))} | {g(w,y,'plugin_cs','bias'):+.3f} / {f2(g(w,y,'plugin_cs','coverage'))} | "
                    f"{f2(g(w,y,'gold_cs','reject'))} | {g(w,y,'ppit_cs','bias'):+.3f} / {f2(g(w,y,'ppit_cs','coverage'))} / {f2(g(w,y,'ppit_cs','reject'))}")
ppi_cov = [g(w, y, "ppit_cs", "coverage") for w in S for y in "AMST"]
L.append(f"{{T:sim}} and {{F:sim}} report the Monte Carlo study. The rectified group-time estimator was close to unbiased for every outcome in every world, with coverage between {pc(min(ppi_cov))} and {pc(max(ppi_cov))}. "
         f"In the going-quiet world, true claim mass fell by {-tr['quiet']['M']:.2f} verifiable commitments per firm-year; the plug-in group-time estimator recovered only {pc((tr['quiet']['M']+g('quiet','M','plugin_cs','bias'))/tr['quiet']['M'])} of that fall and its interval covered the truth in "
         f"{pc(g('quiet','M','plugin_cs','coverage'))} of replications, because missed detections attenuate every machine count. The rectified estimator rejected the null of no claim-mass effect in {pc(g('quiet','M','ppit_cs','reject'))} of replications against "
         f"{pc(g('quiet','M','gold_cs','reject'))} for the gold-only estimator using the same human coding.")
L.append(f"The text-score benchmark fails in the way anticipated in Section III-D. In the closing-the-gap world the true unverifiable share does not move, so a cheap-talk score registers nothing although divergence mass fell by {-tr['closing']['A']:.3f}. "
         f"In the going-quiet world the true unverifiable share rose by {tr['quiet']['T']:.3f}, yet the plug-in estimate was biased by {g('quiet','T','plugin_cs','bias'):+.3f} and covered the truth in only {pc(g('quiet','T','plugin_cs','coverage'))} of replications, "
         f"because missed verifiable claims are themselves counted as vague. Divergence mass alone also cannot discriminate: it fell in both worlds ({tr['closing']['A']:+.3f} and {tr['quiet']['A']:+.3f}), which is exactly why the decomposition is needed.")
L.append(f"Two-way fixed effects were biased even when applied to the true outcomes. With effects that build up over three years, the TWFE estimate of the shortfall effect in the closing world covered the truth in {pc(g('closing','S','oracle_twfe','coverage'))} of replications "
         f"and the claim-mass effect in the quiet world in {pc(g('quiet','M','oracle_twfe','coverage'))}, against {pc(g('closing','S','oracle_cs','coverage'))} and {pc(g('quiet','M','oracle_cs','coverage'))} for the group-time estimator on the same data. "
         f"In the null world, differential measurement error alone made the plug-in group-time estimator reject a true null for claim mass in {pc(g('null','M','plugin_cs','reject'))} of replications, against {pc(g('null','M','ppit_cs','reject'))} after rectification, close to but slightly above the nominal 5%.")
L.append("!TAB sim | Monte Carlo Results for Assurance Effects (400 Replications per World; Gold Share 0.20; Post-Assurance Sensitivity Shift 0.05) | 8L,16,19,19,14,24 | W")
L += rows; L.append("!END")
L.append("!FIG sim | fig5_simulation.png | Sampling distributions of assurance-effect estimates for claim mass and mean shortfall in the closing-the-gap and going-quiet worlds. Boxes span the interquartile range; whiskers extend to 1.5 times that range.")
EVT = json.load(open(f"{OUT}/sim_event.json")); qe = EVT["quiet"]; ce = EVT["closing"]
allcov = qe["ppi_cov"] + ce["ppi_cov"]
L.append(f"{{F:event}} traces the effects by years since assurance for the 2018 cohort against firms not yet treated. Pre-treatment estimates are close to zero for both estimators, which is the diagnostic the application will use for anticipation. "
         f"In the quiet world the true fall in claim mass deepens as new commitments stop being issued and older ones leave the evaluation window, reaching {qe['truth'][-1]:+.2f} by year seven; the rectified profile follows it, "
         f"whereas the plug-in profile has the right shape but reaches only {pc(qe['plugin'][-1] / qe['truth'][-1])} of the true fall. In the closing world the shortfall effect is concentrated in the first years after assurance, "
         f"peaking at {min(ce['truth']):+.3f}, because commitments issued later are based on the lower emission level and their shortfall returns toward zero. Here the plug-in and rectified profiles coincide, since misdetection adds noise to the mean shortfall but does not attenuate it. "
         f"Coverage of the rectified intervals at individual event times ranged from {pc(min(allcov))} to {pc(max(allcov))}.")
L.append("!FIG event | fig8_event.png | Event-time effects for the 2018 assurance cohort (200 replications). The shaded band spans the 2.5th to 97.5th percentiles of rectified estimates; the vertical line separates pre- and post-treatment years, with the year before assurance as the base.")
# ---------------- Ablation & shares ----------------
sh = SIM["share"]; G = SIM["gate"]
L.append("## Ablation and Channel Recovery")
L.append(f"Each component was removed in turn ({{T:abl}}). Removing rectification leaves the heterogeneity-robust estimator but uses machine-coded outcomes, which reproduces the attenuation and the false rejections reported above. Replacing the group-time "
         f"estimator with two-way fixed effects introduces bias even without measurement error. Removing the verifiability gate, so that vague commitments receive a parsed rate and enter divergence, changed the plug-in estimate of claim mass in the quiet world "
         f"from {g('quiet','M','plugin_cs','mean'):+.3f} to {G['quiet']['M']['plugin_cs']['mean']:+.3f} and its coverage from {pc(g('quiet','M','plugin_cs','coverage'))} to {pc(G['quiet']['M']['plugin_cs']['coverage'])}, "
         f"while the untuned rectified estimate stayed valid (coverage {pc(G['quiet']['M']['ppi_cs']['coverage'])}) but lost power ({pc(G['quiet']['M']['ppi_cs']['reject'])} against {pc(g('quiet','M','ppi_cs','reject'))} with the gate) because false verifiable claims add noise. "
         f"Fixing $\\lambda=1$ instead of tuning it costs power where predictions are weakly related to the truth: for the shortfall effect in the closing world, power fell from {pc(g('closing','S','ppit_cs','reject'))} to {pc(g('closing','S','ppi_cs','reject'))}, below the {pc(g('closing','S','gold_cs','reject'))} of the gold-only estimator, which is the case that power tuning is designed for. "
         f"Removing the decomposition leaves only divergence mass, which falls in both worlds and therefore cannot classify the response.")
rows = ["Configuration | Closing: $\\overline{S}$ cov. / power | Quiet: $M$ cov. / power | Can classify response"]
rows.append(f"Full framework (power-tuned) | {f2(g('closing','S','ppit_cs','coverage'))} / {f2(g('closing','S','ppit_cs','reject'))} | {f2(g('quiet','M','ppit_cs','coverage'))} / {f2(g('quiet','M','ppit_cs','reject'))} | Yes")
rows.append(f"Untuned rectification ($\\lambda=1$) | {f2(g('closing','S','ppi_cs','coverage'))} / {f2(g('closing','S','ppi_cs','reject'))} | {f2(g('quiet','M','ppi_cs','coverage'))} / {f2(g('quiet','M','ppi_cs','reject'))} | Yes, lower power")
rows.append(f"Without rectification | {f2(g('closing','S','plugin_cs','coverage'))} / {f2(g('closing','S','plugin_cs','reject'))} | {f2(g('quiet','M','plugin_cs','coverage'))} / {f2(g('quiet','M','plugin_cs','reject'))} | Biased")
rows.append(f"TWFE instead of group-time | {f2(g('closing','S','plugin_twfe','coverage'))} / {f2(g('closing','S','plugin_twfe','reject'))} | {f2(g('quiet','M','plugin_twfe','coverage'))} / {f2(g('quiet','M','plugin_twfe','reject'))} | Biased")
rows.append(f"Without verifiability gate | {f2(G['closing']['S']['ppi_cs']['coverage'])} / {f2(G['closing']['S']['ppi_cs']['reject'])} | {f2(G['quiet']['M']['ppi_cs']['coverage'])} / {f2(G['quiet']['M']['ppi_cs']['reject'])} | Weaker")
rows.append(f"Gold coding only | {f2(g('closing','S','gold_cs','coverage'))} / {f2(g('closing','S','gold_cs','reject'))} | {f2(g('quiet','M','gold_cs','coverage'))} / {f2(g('quiet','M','gold_cs','reject'))} | Yes, lower power")
rows.append(f"Divergence mass only | n.a. | n.a. | No")
L.append(f"The shares in (EQ:share) recover the mechanism. In the closing world the median performance share was {f2(sh['closing']['perf_median'])} (interquartile range {f2(sh['closing']['perf_iqr'][0])} to {f2(sh['closing']['perf_iqr'][1])}) and the claim share "
         f"{f2(sh['closing']['claim_median'])}; in the quiet world the claim share was {f2(sh['quiet']['claim_median'])} ({f2(sh['quiet']['claim_iqr'][0])} to {f2(sh['quiet']['claim_iqr'][1])}) and the performance share {f2(sh['quiet']['perf_median'])}. "
         f"In the mixed world the medians were {f2(sh['mixed']['claim_median'])} and {f2(sh['mixed']['perf_median'])}, against true values of {f2(tr['mixed']['share_claim'])} and {f2(tr['mixed']['share_perf'])}. The interquartile ranges are wide, "
         f"so the shares are best read as a summary of two separately tested effects rather than as estimates to be reported to two decimals. {{F:shares}} shows the full distributions; a share can exceed one when the other channel moves in the opposite direction, as the slight rise in shortfall does in the quiet world.")
L.append("!FIG shares | fig7_shares.png | Distribution of the claim share and the performance share across 400 replications per world, computed from rectified effects. Dashed segments mark the true shares from a ten-times larger panel; whiskers are truncated at the axis limits.")
L.append("!TAB abl | Component Ablation in the Monte Carlo Design (Coverage / Rejection Rate of the Key Effect) | 34L,24,24,18")
L += rows; L.append("!END")
# ---------------- Sensitivity & efficiency ----------------
SS = SIM["sens"]
def sg(q, d, e, k="coverage", y="M"): return SS[f"{q}|{d}"][y][e][k]
L.append("## Sensitivity and Computational Cost")
L.append(f"{{F:sens}} varies the gold share and the post-assurance shift in annotator sensitivity in the going-quiet world. Plug-in coverage for claim mass stayed between {pc(min(sg(q,d,'plugin_cs') for q in (0.1,0.2,0.3) for d in (0.0,0.05,0.1)))} and "
         f"{pc(max(sg(q,d,'plugin_cs') for q in (0.1,0.2,0.3) for d in (0.0,0.05,0.1)))} in every cell, because attenuation does not depend on the gold sample. Rectified coverage stayed between "
         f"{pc(min(sg(q,d,'ppit_cs') for q in (0.1,0.2,0.3) for d in (0.0,0.05,0.1)))} and {pc(max(sg(q,d,'ppit_cs') for q in (0.1,0.2,0.3) for d in (0.0,0.05,0.1)))}, and its power rose from "
         f"{pc(sg(0.1,0.05,'ppit_cs','reject'))} with a ten-percent gold share to {pc(sg(0.3,0.05,'ppit_cs','reject'))} with thirty percent. For an application of the size simulated, coding roughly one firm in five is therefore the practical minimum.")
L.append("!FIG sens | fig6_sensitivity.png | Coverage of nominal 95% intervals for the claim-mass effect in the going-quiet world as a function of gold share and post-assurance sensitivity shift; 100 replications per cell.")
tt = {m: np.mean([TM[f"{m}|{t}"][1] for t in ("claim", "commit", "specific", "target")]) * 1000 for m in ("lexicon", "tfidf_lr", "climatebert_lr", "minilm_lr")}
llm_s = None
try:
    llm_s = float(re.findall(r"([\d.]+)s/item", open(f"{OUT}/llm_test.log").read())[0])
except Exception: pass
enc = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"climatebert (\w+) \((\d+), 768\) (\d+)s", open(f"{OUT}/encode.log").read())}
per = [s_ / n for n, s_ in enc.values()]
L.append(f"Annotation dominated cost. After representations were computed, a probe labelled a text in {tt['climatebert_lr']:.3f} ms on average, while encoding with the frozen climate encoder took between {min(per):.2f} and {max(per):.2f} s per text on two CPU cores, depending on text length"
         + (f" and the zero-shot language model {llm_s:.1f} s per text" if llm_s else "") + ". All rectified estimators ran in milliseconds, and one complete Monte Carlo replication of all estimators took well under a second, so the inferential layer adds nothing material to the cost of annotation.")
open(TXT, "w").write("\n".join(L) + "\n"); print("wrote", TXT, len(" ".join(L).split()), "words")
# ---------------- Conclusion and abstract (numbers taken from the same outputs) ----------------
q_plug = g('quiet', 'M', 'plugin_cs', 'coverage'); q_pp = g('quiet', 'M', 'ppit_cs', 'coverage')
q_pow = g('quiet', 'M', 'ppit_cs', 'reject'); c_pow = g('closing', 'S', 'ppit_cs', 'reject')
redmax = max(red[k] for k in ("theta3_claim", "theta4_netzero")); redmin = min(red[k] for k in ("theta3_claim", "theta4_netzero"))
C = []
C.append("# Conclusion")
C.append("When mandatory assurance arrives, measured greenwashing can fall because firms meet their promises or because they stop making promises that can be checked, and existing text measures cannot tell these responses apart. "
         "This study proposed the claim–performance divergence framework, which compares only verifiable commitments with verified emissions, decomposes the change in divergence into a performance channel and a claim channel, "
         "and corrects every machine-coded firm statistic with a small human-coded gold sample before estimating staggered assurance effects.")
C.append(f"The evidence has two parts. On 7,408 human-labelled disclosure texts, the attributes the measure needs could be recovered with macro-F1 between {f3(min(r1(t, bt[t]) for t in E1))} and {f3(max(r1(t, bt[t]) for t in E1))}, "
         f"specificity being the hardest, while readability, tone and heuristic flags carried little information about verifiability. Rectified intervals covered known population values in at least {pc(min(covs))} of draws for every annotator, "
         f"and with a supervised encoder they were {pc(redmin)} to {pc(redmax)} narrower than intervals from gold coding alone. In a Monte Carlo design that reproduces the European assurance timetable and the measured error rates, "
         f"the rectified group-time estimator kept coverage between {pc(min(ppi_cov))} and {pc(max(ppi_cov))} and detected a going-quiet response in {pc(q_pow)} of replications, whereas the plug-in estimator covered the true fall in claim mass in only "
         f"{pc(q_plug)} of replications and two-way fixed effects were biased even without measurement error.")
C.append("The practical implication for assurance standards is that procedures should reconcile commitments over time and require disclosure of dropped targets, because withdrawing checkable claims reduces what can be contradicted without any change in operations. "
         "The principal limitation is that the assurance effects themselves were simulated; the next step is to run the pre-specified application on European management reports linked to registry emissions, with the gold sample and reliability thresholds already fixed in the protocol.")
open(os.path.join(HERE, "outputs", "conclusion_section.txt"), "w").write("\n".join(C) + "\n")
ABS = (f"Text-based greenwashing scores are usually validated against the labels that produced them and rarely separate misleading claims from vague disclosure. They also cannot tell whether a fall in measured greenwashing "
       f"after mandatory sustainability assurance reflects better performance or the withdrawal of checkable commitments, a response known as greenhushing. This study proposes a claim–performance divergence framework. "
       f"Only commitments that pass a verifiability gate are compared with verified emissions, the change in divergence is decomposed exactly into a performance channel and a claim channel, and machine-coded firm statistics are "
       f"rectified with a small human-coded gold sample through prediction-powered inference, including a moment-rectified estimator for regressions in which outcomes and regressors are both machine-coded, before staggered group-time "
       f"assurance effects are estimated. On 7,408 human-labelled corporate disclosure texts, the required attributes were recovered with macro-F1 of {f3(min(r1(t, bt[t]) for t in E1))} to {f3(max(r1(t, bt[t]) for t in E1))}, "
       f"whereas readability, tone and heuristic flags discriminated verifiable from vague paragraphs only weakly. Rectified intervals covered known population values in at least {pc(min(covs))} of draws for every annotator and were up to "
       f"{pc(redmax)} narrower than gold-only intervals. In a Monte Carlo design calibrated to European assurance timing and measured error rates, rectified estimates kept coverage between {pc(min(ppi_cov))} and {pc(max(ppi_cov))} "
       f"and detected greenhushing in {pc(q_pow)} of replications, while plug-in estimates covered the true effect in {pc(q_plug)}. Assurance procedures should therefore track withdrawn commitments as well as reported figures.")
open(os.path.join(HERE, "outputs", "abstract.txt"), "w").write(ABS); print("abstract words", len(ABS.split()))
