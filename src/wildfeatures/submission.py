import numpy as np
import pandas as pd


def validate(frame, expected_ids, id_column="id", score_column="score"):
    if not {id_column, score_column}.issubset(frame.columns):
        raise ValueError("Missing required submission columns")
    ids = pd.Series(list(expected_ids))
    if len(ids) == 0:
        raise ValueError("No expected test ids")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Expected ids must be unique and non-null")
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError("Submission ids must be unique and non-null")
    if len(frame) != len(ids) or set(frame[id_column]) != set(ids):
        raise ValueError("Row count or ids do not match input")
    if not pd.api.types.is_numeric_dtype(frame[score_column]) or pd.api.types.is_bool_dtype(frame[score_column]):
        raise ValueError("Scores must have numeric dtype")
    if not np.isfinite(frame[score_column].to_numpy(dtype=float)).all():
        raise ValueError("Scores contain NaN or infinity")
    return True
