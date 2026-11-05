"""Codebook-constrained zero-shot labelling with a small open instruction-tuned LLM.
The model sees only the codebook definition and the text; option probabilities are read from
next-token logits of the option letters, so every label comes with a probability."""
import sys, time, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import load_task, OUT
torch.set_num_threads(2)
REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
SCOPE = sys.argv[2] if len(sys.argv) > 2 else "test"
CODEBOOK = {
 "claim": ("An environmental claim is a statement in which a company states or implies that its products, "
           "services, operations or the company as a whole are environmentally friendly, sustainable or "
           "beneficial for the environment. Factual descriptions without such a positive implication are not claims.",
           ["A: not an environmental claim", "B: environmental claim"]),
 "commit": ("A climate commitment or action is a paragraph that states a target, pledge or plan to address "
            "climate change, or reports an action already taken or under way. General discussion of climate "
            "risk, markets or policy without the company committing to or doing something is not a commitment.",
            ["A: no commitment or action", "B: commitment or action"]),
 "specific": ("A specific paragraph gives concrete details that an outsider could verify: quantities, dates, "
              "baselines, named projects, measured results or clearly bounded scopes. A non-specific paragraph "
              "uses general statements, intentions or boilerplate that cannot be checked.",
              ["A: non-specific", "B: specific"]),
 "target": ("Classify the emission target stated in the text. A net-zero target commits to net-zero emissions or "
            "carbon neutrality. A reduction target commits to cutting emissions by an amount or share without "
            "reaching net zero. Otherwise there is no target.",
            ["A: no target", "B: reduction target", "C: net-zero target"]),
}
tok = AutoTokenizer.from_pretrained(REPO)
mdl = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()
def score(task, text):
    rule, opts = CODEBOOK[task]
    msg = [{"role": "system", "content": "You are a careful annotator of corporate disclosures. Answer with one letter."},
           {"role": "user", "content": f"Codebook: {rule}\n\nText: \"{text[:1500]}\"\n\nOptions:\n" + "\n".join(opts) +
            "\nAnswer with the letter only."}]
    prompt = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = mdl(ids).logits[0, -1]
    letters = [tok.encode(L, add_special_tokens=False)[0] for L in "ABC"[:len(opts)]]
    p = torch.softmax(logits[letters], 0).numpy()
    return p
jobs = []
if SCOPE == "test":
    for task in ["claim", "target", "commit", "specific"]:
        d = load_task(task); jobs.append((task, d[d.split == "test"]))
else:  # full joint set for prediction-powered inference with an untrained annotator
    d = load_task("joint")
    jobs += [("commit", d), ("specific", d)]
tag = REPO.split("/")[-1]
for task, d in jobs:
    t0 = time.time(); P = np.vstack([score(task, x) for x in d.text])
    np.save(f"{OUT}/llm_{tag}_{SCOPE}_{task}.npy", P)
    print(task, len(d), f"{(time.time()-t0)/len(d):.2f}s/item", flush=True)
