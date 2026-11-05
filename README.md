# Claim–performance divergence: code for "Closing the Gap or Going Quiet?"

All files sit in one folder. Running `bash run_all.sh` downloads the public human-labelled corpora,
reproduces every experiment and writes results to `outputs/` and figures to `figures/`.

## What each script does
| Script | Purpose |
|---|---|
| `prepare_data.py` | Downloads the ClimateBERT environmental-claims, commitment, specificity, sentiment and net-zero target corpora (human labels only) |
| `common.py` | Loaders and fixed splits (stratified 80/20 split with seed 2026 for the target corpus) |
| `encode.py` | Frozen mean-pooled representations from a general sentence encoder and the ClimateBERT encoder |
| `lexicon.py` | Dictionary baseline, heuristic greenwashing flag, Flesch reading ease, numeric density, vague-word rate |
| `measure_train.py` | Dictionary, TF-IDF, and probe annotators; 5-fold cross-fitted predictions for inference |
| `published_checkpoints.py` | Scores the official test splits with the released task-specific ClimateBERT checkpoints |
| `zeroshot_llm.py` | Codebook-constrained zero-shot labelling with Qwen2.5-0.5B-Instruct (option-letter probabilities) |
| `metrics.py` | Macro-F1, MCC, balanced accuracy, AUROC, ECE, Brier; bootstrap CIs, paired bootstrap tests, Holm correction |
| `evaluate_measurement.py` | E1 annotator comparison and E2 proxy validity |
| `ppi.py` | Prediction-powered mean, power-tuned mean, and moment-rectified OLS with sandwich variance |
| `experiment_ppi_real.py` | E3 coverage and width against human-labelled truth |
| `simulate_assurance.py` | E4–E6 Monte Carlo panel: staggered assurance, group-time and TWFE estimators, rectification, ablations, sensitivity |
| `summarize_simulation.py` | Bias, RMSE, coverage and rejection rates |
| `simulate_event_time.py` | Event-time profile for the 2018 cohort |
| `fig_framework.py`, `plots.py` | Grayscale figures (overlap-checked framework diagram, result plots) |
| `make_results.py` | Writes the results text and tables directly from the JSON outputs |

## Notes
* No label is created by this code; every accuracy and coverage number is measured against labels released by the corpus authors.
* The simulation is a declared data-generating process. Its error rates are read from the E1 confusion matrices.
* The corpora carry their own licences (CC BY-NC-SA 4.0 for the ClimateBERT task sets; Apache-2.0 for the target data).
