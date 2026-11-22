#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
"$PY" -m wildfeatures.cli data --n-children "${N_CHILDREN:-48}" --seed "${SEED:-42}"
"$PY" -m wildfeatures.cli features
"$PY" -m wildfeatures.cli train
"$PY" -m wildfeatures.cli submit
"$PY" -m wildfeatures.cli validate
