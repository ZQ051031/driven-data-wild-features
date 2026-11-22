PY = .venv/bin/python
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
.PHONY: setup data features train submit test all
setup:
	python3 -m venv .venv
	$(PY) -m pip install -e .
data:
	$(PY) -m wildfeatures.cli data
features:
	$(PY) -m wildfeatures.cli features
train:
	$(PY) -m wildfeatures.cli train
submit:
	$(PY) -m wildfeatures.cli submit
	$(PY) -m wildfeatures.cli validate
test:
	$(PY) -m pytest -q
all:
	bash scripts/run_pipeline.sh
