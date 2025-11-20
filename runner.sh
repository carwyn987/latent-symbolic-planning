#!/bin/bash

set -euo pipefail

# -------------------------------
# Parameter Sweeps
# -------------------------------

DEBUG_LEVELS=(0)
NUM_CLUSTERS=(10 20 30)
NUM_STEPS=(10000 20000)
CLUSTERING_METHODS=("kmeans")
NUM_ACT_APPLY=(10 20)
HARDCODE_START_GOAL=("True")
FULL_REPLAN=("True")

# Base output directory
BASE_OUT="results"

mkdir -p "$BASE_OUT"

# -------------------------------
# Iterate Over All Combinations
# -------------------------------

for dbg in "${DEBUG_LEVELS[@]}"; do
for nc in "${NUM_CLUSTERS[@]}"; do
for ns in "${NUM_STEPS[@]}"; do
for cm in "${CLUSTERING_METHODS[@]}"; do
for naa in "${NUM_ACT_APPLY[@]}"; do
for hcg in "${HARDCODE_START_GOAL[@]}"; do
for fr in "${FULL_REPLAN[@]}"; do

    # Construct output directory name
    OUTDIR="${BASE_OUT}/dbg${dbg}_clust${nc}_steps${ns}_${cm}_act${naa}_h${hcg}_fr${fr}"
    mkdir -p "$OUTDIR"

    echo "===================================================="
    echo "Running:"
    echo "  debug=$dbg"
    echo "  num_clusters=$nc"
    echo "  num_steps=$ns"
    echo "  method=$cm"
    echo "  num_act_apply=$naa"
    echo "  hardcode_start_goal=$hcg"
    echo "  full_replan=$fr"
    echo "  output=$OUTDIR"
    echo "===================================================="

    python main.py \
        -d "$dbg" \
        -c "$nc" \
        -s "$ns" \
        -m "$cm" \
        -a "$naa" \
        -g "$hcg" \
        -f "$fr" \
        -o "$OUTDIR"

done; done; done; done; done; done; done