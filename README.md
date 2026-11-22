# Wild Features — synthetic-only local sandbox

**DATA PROHIBITION: Never upload competition data to third-party services that
retain data, including ChatGPT, Claude or Gemini. This project was built without
visiting the competition site, accessing data pages, reading/downloading real
competition data, or downloading pretrained models. All demonstrations and tests
use generated synthetic data only. No command submits anything.**

Context supplied by the user: DrivenData “Where the Wild Features Are” predicts
US kindergarten end-of-year reading scores from beginning-of-year screening
audio and screening scores. Prize pool: **$50,000** (1st $20k, 2nd $15k,
3rd $10k, 4th $5k). Phase 1 model deadline: **2027-01-20 07:59 Beijing time**;
Phase 2 report deadline: **2027-02-09 07:59 Beijing time**. These details have
not been checked against the competition website; the user must confirm them.

## Setup and reproduce

Python 3.10+ and a C-compatible audio runtime are required (tested on Ubuntu,
Python 3.12). SoundFile wheels include libsndfile; LightGBM may need the OS
OpenMP runtime. No network is used after dependency installation.

```bash
make setup  # creates repo-local .venv and installs requirements
source .venv/bin/activate
pytest -q
bash scripts/run_pipeline.sh
```

Default: 48 children, two 16 kHz WAV recordings each, seed 42, typically well
under a few minutes. Override with `N_CHILDREN=80 SEED=123 bash scripts/run_pipeline.sh`.
Individual targets: `make data features train submit test`; `make all` runs the
pipeline. CLI help: `.venv/bin/python -m wildfeatures.cli --help`.
`data --n-children 48 --seed 42 --out data/synthetic` controls generation.

The generator creates a new directory with `SYNTHETIC_ONLY.txt`; an existing
directory without that marker is refused, even if empty. Do not pre-create
`data/synthetic`. Every metadata/label row has `is_synthetic=True`, and a data
README explicitly identifies synthesis. Re-running with a new size in the same
marked directory can leave unused WAVs; only metadata-listed recordings are read.
Use a fresh output directory for clean experiments. Never put real files in a
marked synthetic directory. The marker is a safety convention, not a data detector.

Outputs (ignored by git): `outputs/features.csv`, `model.pkl`, `metrics.json`,
`submission.csv`. Only load trusted local pickle models. Submission CSV defaults
to `id,score`; **these are placeholders, not a verified competition format**.
`submit` and `validate` accept `--id-column` and `--score-column`; both operate
on the `test` split and validate finite numeric scores, unique matching ids and
row count. Validation failures exit nonzero. `submit` writes a local CSV only.

## Method and interfaces

- `wildfeatures.data.generate`: latent child ability drives three clearly named
  synthetic screening scores, final score, speaking rate and pauses. Harmonic
  voiced syllables have formant-like resonances, pitch contours, envelopes,
  silence and noise. These are not natural child speech or phonetic transcripts.
- `features.acoustic.extract_recording`: librosa MFCC mean/std and spectral
  features; NumPy/SciPy normalized-autocorrelation pitch (75–500 Hz, correlation
  threshold 0.65), RMS, duration, pitch/energy-based voiced fraction and pause
  lengths/count, spectral-flux onset rate,
  spectral centroid/bandwidth/rolloff/flatness, ZCR. Voicing and onset rate are
  heuristics, not clinical fluency/phoneme measurements. Silence yields finite
  features; empty audio is rejected. Pitch, onset peak finding and zero crossings
  avoid librosa's Numba gufunc paths (`localmin`, `localmax`, `peak_pick`).
- `build_features`: mean/std across a child's recordings, then child-level
  screening join; no ids, splits or synthetic markers are model inputs.
- `wildfeatures.model`: Ridge (fold-local imputation/scaling), LightGBM or
  automatic sklearn HistGradientBoosting fallback, equal-weight ensemble.
  GroupKFold uses child ids on training children only; preprocessing fits within
  folds. Validation children remain separate and choose the saved model, which
  stays fitted on training only. CV predictions are pooled for RMSE, MAE, R2,
  Spearman and Kendall. Undefined correlations serialize as JSON null.
- `wildfeatures.submission.validate`: reusable independent format validator.

## User-only local real-data adaptation (NOT tested here)

The user downloads data themselves under the competition rules, works on their
own machine without uploading data to AI or other retaining services, confirms
the official format themselves, and submits themselves. **Do not provide real
data or logs containing it to this assistant.** No downloader or uploader exists.

Prepare metadata with one row per recording, a stable child id, a local audio
path, consistent per-child screening scores, and a child-disjoint `split` column
containing `train`, `validation`, or `test`. Copy/edit
`configs/local_columns.example.json`: `columns` maps canonical field names
(`id`, `audio_path`, `split`) to your CSV column names; `tabular` is a list of
numeric screening columns in that CSV. The metadata filename is configurable.
Prepare a separate labels CSV with canonical `id,score` (one labeled child per row).
No real-schema assumptions have been verified.

```bash
# USER executes locally only; replace all paths and column mappings
.venv/bin/python -m wildfeatures.cli features --real-local --root /local/data --config configs/my_columns.json --out outputs/local_features.csv
.venv/bin/python -m wildfeatures.cli train --real-local --root /local/data --labels labels.csv --features outputs/local_features.csv --out outputs/local
.venv/bin/python -m wildfeatures.cli submit --features outputs/local_features.csv --model outputs/local/model.pkl --out outputs/local/predictions.csv --id-column id --score-column score
```

Real mode is an explicit opt-in that bypasses synthetic markers. It does not
access any remote service. Keep real data, derived features, predictions, models
and identifiers out of git. Ignore rules cover repo `data/` and `outputs/` only;
do not place sensitive files elsewhere in the repo. Fit preprocessing on train
only, audit leakage and selection bias, and verify all competition constraints.

## Licenses

Project code: MIT (`LICENSE`). Core dependencies: NumPy/pandas/SciPy/
scikit-learn/librosa — BSD-3-Clause; SoundFile — BSD-3-Clause (bundled libsndfile
LGPL-2.1-or-later); LightGBM — MIT; pytest — MIT. Transitive dependencies retain
their respective licenses. Optional future embeddings, not implemented or
downloaded: OpenAI Whisper code/weights (MIT), or specifically licensed
Apache-2.0 wav2vec2 checkpoints (verify each checkpoint separately; the family
name alone does not establish a license). They are not needed by this pipeline.
See `docs/phase2_report.md` for synthetic results, limitations and report template.
