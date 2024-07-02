#!/bin/bash

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA="$PROJECT"/cmc_data
RUN1="$PROJECT"/cmc_results/abide_i
RUN2="$PROJECT"/cmc_results/hcp
OUTS1="$RUN1/terminal_outputs.txt"
OUTS2="$RUN2/terminal_outputs.txt"
DATA1="$DATA/ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__autism.parquet"
DATA2="$DATA/HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
mkdir -p "$RUN1"
mkdir -p "$RUN2"

"$PROJECT"/.venv/bin/python df-analyze.py \
    --df "$DATA1" \
    --mode classify \
    --target TARGET__CLS__autism \
    --classifiers knn lgbm rf lr sgd dummy \
    --norm robust \
    --nan median \
    --feat-select filter embed wrap \
    --embed-select lgbm linear \
    --wrapper-select step-up \
    --wrapper-model linear \
    --filter-method assoc pred \
    --filter-assoc-cont-classify mut_info \
    --filter-assoc-cat-classify mut_info \
    --filter-pred-classify acc \
    --n-feat-filter 20 \
    --n-feat-wrapper 20 \
    --redundant-wrapper-selection \
    --redundant-threshold 0.01 \
    --htune-trials 100 \
    --htune-cls-metric acc \
    --test-val-size 0.25 \
    --outdir "$RUN1" 2>&1 | tee "$OUTS1"

"$PROJECT"/.venv/bin/python df-analyze.py \
    --df "$DATA2" \
    --mode classify \
    --target TARGET__REG__int_g_like \
    --classifiers knn lgbm rf lr sgd dummy \
    --norm robust \
    --nan median \
    --feat-select filter embed wrap \
    --embed-select lgbm linear \
    --wrapper-select step-up \
    --wrapper-model linear \
    --filter-method assoc pred \
    --filter-assoc-cont-classify mut_info \
    --filter-assoc-cat-classify mut_info \
    --filter-pred-classify acc \
    --n-feat-filter 20 \
    --n-feat-wrapper 20 \
    --redundant-wrapper-selection \
    --redundant-threshold 0.01 \
    --htune-trials 100 \
    --htune-cls-metric acc \
    --test-val-size 0.25 \
    --outdir "$RUN2" 2>&1 | tee "$OUTS2"