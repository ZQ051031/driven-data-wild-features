import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def metrics(y, prediction):
    def finite(value):
        return float(value) if np.isfinite(value) else None
    variable = np.ptp(np.asarray(y)) > 0 and np.ptp(np.asarray(prediction)) > 0
    return dict(rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mae=float(mean_absolute_error(y, prediction)), r2=finite(r2_score(y, prediction)),
                spearman=finite(spearmanr(y, prediction).statistic) if variable else None,
                kendall=finite(kendalltau(y, prediction).statistic) if variable else None)


def estimators(seed=42):
    ridge = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10))
    try:
        from lightgbm import LGBMRegressor
        tree = LGBMRegressor(n_estimators=80, num_leaves=7, min_child_samples=5,
                             verbosity=-1, random_state=seed, n_jobs=1)
        backend = "lightgbm"
    except ImportError:
        tree = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7,
                                            min_samples_leaf=5, random_state=seed)
        backend = "sklearn_hist_gradient_boosting"
    return {"ridge": ridge, "boosting": tree,
            "ensemble": VotingRegressor([("ridge", clone(ridge)), ("boosting", clone(tree))])}, backend


def train(features, labels, out, folds=3):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = features.merge(labels[["id", "score"]], on="id", validate="one_to_one")
    columns = [c for c in features if c not in ("id", "split", "is_synthetic")]
    training, validation = data[data.split == "train"], data[data.split == "validation"]
    if len(training) < folds or len(validation) < 2:
        raise ValueError("Insufficient training children or validation children")
    models, backend = estimators()
    report = {"backend": backend, "train_children": len(training), "validation_children": len(validation), "models": {}}
    for name, model in models.items():
        prediction = cross_val_predict(model, training[columns], training.score,
                                       groups=training.id, cv=GroupKFold(folds))
        model.fit(training[columns], training.score)
        report["models"][name] = {"grouped_cv": metrics(training.score, prediction),
                                  "validation": metrics(validation.score, model.predict(validation[columns]))}
    selected = min(models, key=lambda k: report["models"][k]["validation"]["rmse"])
    report["selected"] = selected
    report["note"] = "Validation used for model selection; not an unbiased final estimate. Synthetic results only when using synthetic data."
    with (out / "model.pkl").open("wb") as f:
        pickle.dump({"model": models[selected], "columns": columns}, f)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
