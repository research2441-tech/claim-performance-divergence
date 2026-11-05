"""Transparent keyword rules derived from the codebook wording; used as the dictionary baseline
and as the 'heuristic greenwashing flag' that re-creates the labelling logic criticised in review."""
import re, numpy as np
GREEN = r"\b(sustainab\w*|green\w*|environment\w*|eco\w*|clean\w*|renewabl\w*|low[- ]carbon|carbon[- ]neutral\w*|net[- ]zero|climate[- ]friendly|responsib\w*)\b"
COMMIT = r"\b(target\w*|commit\w*|pledg\w*|aim\w*|plan\w*|will|intend\w*|goal\w*|strive|ambition\w*|implemented|installed|invested|launched|reduced)\b"
VAGUE = r"\b(strive|aim|seek|endeavou?r|committed to|dedicated|aspire|where possible|continue to|strong|significant|leading|best|responsible|sustainable)\b"
NUM = r"(\d+(\.\d+)?\s?%|\b(19|20)\d{2}\b|\b\d[\d,\.]*\s?(t|kt|mt|tco2e?|mwh|gwh|kwh|tonnes?|tons?|million|billion)\b)"
NZ = r"\b(net[- ]?zero|carbon[- ]neutral\w*|climate[- ]neutral\w*|zero emissions)\b"
RED = r"\b(reduc\w*|cut\w*|lower\w*|decreas\w*)\b"
def count(p, t): return len(re.findall(p, t.lower()))
def scores(task, texts):
    s = []
    for t in texts:
        if task == "claim": s.append(count(GREEN, t))
        elif task == "commit": s.append(count(COMMIT, t))
        elif task == "specific": s.append(count(NUM, t) - 0.5 * count(VAGUE, t))
        elif task == "target":
            s.append(2 if count(NZ, t) else (1 if (count(RED, t) and count(NUM, t)) else 0))
    return np.array(s, float)
def heuristic_greenwash_flag(texts):
    """Flag = green or commitment language, no quantification, at least one vague marker."""
    return np.array([float(count(GREEN, t) + count(COMMIT, t) > 0 and count(NUM, t) == 0 and count(VAGUE, t) > 0)
                     for t in texts])
def numeric_density(texts):
    return np.array([len(re.findall(r"\d+", t)) / max(1, len(t.split())) for t in texts])
def vague_rate(texts):
    return np.array([count(VAGUE, t) / max(1, len(t.split())) for t in texts])
def _syllables(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w: return 0
    groups = re.findall(r"[aeiouy]+", w); n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1: n -= 1
    return max(1, n)
def flesch(text):
    """Flesch reading ease with a vowel-group syllable counter."""
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z]+", text)
    if not words: return 0.0
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words)
