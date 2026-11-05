#!/bin/bash
# Reproduces every table and figure. CPU only; about 2-3 hours on two cores, most of it the zero-shot LLM.
set -e
cd "$(dirname "$0")"; mkdir -p outputs figures
python3 prepare_data.py
python3 encode.py                      > outputs/encode.log 2>&1
python3 measure_train.py               > outputs/measure.log 2>&1
python3 published_checkpoints.py       > outputs/published.log 2>&1
python3 zeroshot_llm.py Qwen/Qwen2.5-0.5B-Instruct test > outputs/llm_test.log 2>&1
python3 evaluate_measurement.py        > outputs/evaluate.log 2>&1
python3 experiment_ppi_real.py         > outputs/ppi_real.log 2>&1
python3 simulate_assurance.py 400      > outputs/simulation.log 2>&1
python3 summarize_simulation.py        > outputs/simulation_summary.log 2>&1
python3 simulate_event_time.py         > outputs/event_time.log 2>&1
python3 fig_framework.py figures/fig1_framework.png
python3 plots.py figures
python3 make_results.py
echo "done: see outputs/ and figures/"
