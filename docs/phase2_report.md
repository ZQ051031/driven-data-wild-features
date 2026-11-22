# Phase 2 report template — SYNTHETIC demonstration only

**Not a competition submission. No real competition data was accessed, read,
downloaded or uploaded. No pretrained models were downloaded. All results below
come from generated tones and artificial screening/target scores, not children.**

## 1. Administrative summary [user completes locally]

- Team / authors: [fill in]
- Final competition model / submission identifier: [fill in locally]
- Official format and rules verification: [user confirms independently]
- User-supplied Phase 1 deadline: 2027-01-20 07:59 Beijing time.
- User-supplied Phase 2 deadline: 2027-02-09 07:59 Beijing time.
- Prize pool: $50k ($20k / $15k / $10k / $5k). Not independently verified here.

## 2. Data and governance

This sandbox generates 48 artificial child ids with two speech-like recordings
per child (96 mono WAVs at 16 kHz). Latent ability controls speaking rate, pauses,
three artificial screening scores and the artificial end-of-year target.
Fundamental pitch varies by child; tasks perturb pitch and formant-like filtering.
Harmonics, syllable envelopes, pauses and noise approximate selected acoustic
properties, not natural language or genuine kindergarten recordings.

Metadata and labels carry `is_synthetic=True`; the generated directory contains
`SYNTHETIC_ONLY.txt` and a README. Synthetic generation and raw feature reads
refuse existing unmarked data directories. Test targets are withheld. Never put
real data in marked directories. Markers are conventions, not content detection.

Competition rules prohibit uploading data to retaining third-party services,
including ChatGPT, Claude and Gemini. Real-data adaptation is a user-only local
activity; the user obtains data, checks requirements, runs locally and submits
themselves. This project has no download or submission transport.

Real-data inventory, permitted uses, consent/privacy controls, exclusions,
recording quality and demographics: **[user completes locally; untested here]**.

## 3. Method and feature design

Recordings are loaded locally and resampled to 16 kHz. Librosa provides 13 MFCC
means/stds and spectral centroid, bandwidth, rolloff and flatness means/stds.
NumPy computes RMS and zero-crossing summaries in centered 2048-sample frames
with a 512-sample hop. Pitch uses SciPy autocorrelation normalized by overlapping
frame energies, searches 75–500 Hz, and accepts local peaks with correlation
at least 0.65 and at least 95% of the best correlation. The earliest qualifying
lag avoids selecting a multiple-period peak. Frames below RMS 0.02 are unvoiced.
Child voicing also requires RMS above 15% of the recording maximum.

Features include duration, voiced fraction, pause count/mean duration and an
onset-rate proxy. Onsets are positive spectral-flux peaks found by SciPy,
restricted to rising frame energy to reject offsets, with 0.1 s minimum spacing
and relative/absolute prominence thresholds. These are heuristics, not clinical
fluency, phonetic accuracy or true speaking-rate measurements. No YIN,
`localmin`, `localmax` or `peak_pick` librosa gufunc paths are used.

Per-recording acoustic features aggregate by child using mean/std; single-record
standard deviations become zero. Three synthetic screening scores are joined.
Identifiers, split assignments and synthetic markers never enter model inputs.

Real-data column mappings, missingness decisions, task-specific ablations and
quality audits: **[user fills in locally]**. CSV is the supported feature format.

## 4. Models

- Ridge baseline: fold-local mean imputation, standard scaling, alpha 10.
- LightGBM: 80 estimators, 7 leaves, minimum 5 child samples, random state 42,
  one worker. If unavailable, sklearn HistGradientBoosting uses 80 iterations,
  7 leaves and minimum 5 samples. No XGBoost dependency is needed.
- Simple ensemble: equal-weight average of Ridge and boosting predictions.

The lowest validation RMSE selects the saved model; it remains trained on the
training split only. This run selected **LightGBM (`boosting`)**. The artifact
contains the fitted estimator and ordered feature names. Only trusted local
pickle artifacts should be loaded.

## 5. Validation protocol

Seed 42 assigns whole children to 30 training, 9 validation and 9 test ids before
recording aggregation. Training evaluation pools out-of-fold predictions from
three-fold GroupKFold grouped by child. Imputation/scaling fits within each
training fold. No child's recordings cross partitions. The separate validation
set selects among the three candidates, so its results are selection-biased,
not an unbiased final generalization estimate. Test predictions are checked for
format only; no test accuracy is reported.

Metrics: RMSE, MAE, R², Spearman rank correlation and Kendall tau. Undefined
correlations become JSON null. Negative R² is allowed. Proposed real evaluation:
**[user defines school/site-aware splits, uncertainty, subgroup audits, and a
truly untouched final holdout locally]**.

## 6. Actual SYNTHETIC results

Generated by the full default `bash scripts/run_pipeline.sh`, seed 42, with the
NumPy/SciPy pitch/onset implementation. Values below rounded to six decimals;
full precision is retained in ignored `outputs/metrics.json`.

| Model | Split | RMSE | MAE | R² | Spearman | Kendall |
|---|---|---:|---:|---:|---:|---:|
| Ridge | Grouped CV | 7.306987 | 6.467108 | 0.798192 | 0.899444 | 0.737931 |
| Ridge | Validation | 6.144917 | 4.804441 | 0.746979 | 0.950000 | 0.833333 |
| LightGBM | Grouped CV | 8.801730 | 7.216204 | 0.707182 | 0.852280 | 0.659770 |
| LightGBM | Validation | 4.194645 | 3.136074 | 0.882100 | 0.983333 | 0.944444 |
| Ensemble | Grouped CV | 7.261937 | 5.728605 | 0.800673 | 0.904338 | 0.747126 |
| Ensemble | Validation | 4.292208 | 3.498054 | 0.876551 | 0.966667 | 0.888889 |

Local prediction generation produced `id,score` for 9 synthetic test children.
The validator passed required columns, finite numeric scores, unique ids,
matching id sets and row count. The column names are placeholders; the user must
confirm the official format. **Nothing was submitted anywhere.**

## 7. Limitations and interpretation

These numbers demonstrate software execution only. They are not evidence of
competition performance: audio and screening scores share a deliberately simple
latent cause, the sample is tiny, artificial targets lack real measurement bias,
and no speech content, site effects or demographic variation is modeled.
Validation selection further inflates apparent performance. Autocorrelation can
make octave errors and is vulnerable to noise; onset/pause measures depend on
thresholds and recording conditions. No real schema or data has been tested.
No fairness, educational efficacy, diagnostic validity or deployment claim is
made. Ablations, nested tuning and confidence intervals remain future work.

## 8. Reproducibility

```bash
make setup
source .venv/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q
bash scripts/run_pipeline.sh
```

Verified: 21 tests passed, including tone/noise/silence/error pitch cases,
onset bursts, forbidden gufunc entry-point guards, generator reproducibility,
feature aggregation, optional boosting fallback, validator negative cases,
nonzero CLI validation failure and a small end-to-end smoke test. Three
third-party audioread deprecation warnings appear on Python 3.12.

Environment: Ubuntu, Python 3.12; NumPy 2.2.6, SciPy 1.18.1, pandas 3.1.1,
scikit-learn 1.9.1, librosa 0.11.0, SoundFile 0.14.0, LightGBM 4.7.0,
Numba 0.61.2, pytest 9.2.0. Dependency constraints are in `requirements.txt`;
not every transitive dependency is pinned, so exact cross-version bitwise
reproduction is not guaranteed. Initial YIN extraction crashed in a native
Numba gufunc; the replacement avoids that path without further crash debugging.
Generated data, features, models and predictions stay outside git via ignores.

## 9. Licenses and external resources

Project code: MIT. NumPy, pandas, SciPy, scikit-learn and librosa: BSD-3-Clause;
SoundFile: BSD-3-Clause, bundled libsndfile: LGPL-2.1-or-later; LightGBM and
pytest: MIT; Numba: BSD-2-Clause. Transitive dependencies retain their licenses.
No pretrained models or external datasets are used. Optional future research,
not implemented, downloaded or tested: Whisper (MIT code/weights) or specifically
Apache-2.0-licensed wav2vec2 checkpoints; verify the exact checkpoint's license,
training provenance and competition eligibility before any user-local use.

## 10. Final real-data report checklist [user only]

- [ ] Confirm official dates, rules, report requirements and CSV column names.
- [ ] Fill permitted local data inventory and privacy controls without uploading it.
- [ ] Audit label availability, school/child leakage, missingness and confounders.
- [ ] Report actual local validation protocol, uncertainty, ablations and limitations.
- [ ] Document final environment, code revision, model selection and licenses.
- [ ] Validate local predictions and submit personally through official channels.
