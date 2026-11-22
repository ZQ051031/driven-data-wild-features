import builtins
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from features.acoustic import audio_frames, build_features, estimate_pitch, extract_recording, onset_rate
from wildfeatures.data import generate, synthetic_directory
from wildfeatures.model import estimators, train
from wildfeatures.submission import validate


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("parent") / "synthetic", 10, 7)


def test_generator(dataset, tmp_path):
    metadata = pd.read_csv(dataset / "metadata.csv")
    assert len(metadata) == 20
    assert metadata.is_synthetic.all()
    assert metadata.groupby("id").split.nunique().eq(1).all()
    assert set(metadata.split) == {"train", "validation", "test"}
    other = generate(tmp_path / "repeat", 10, 7)
    assert (other / "metadata.csv").read_bytes() == (dataset / "metadata.csv").read_bytes()
    wav = metadata.audio_path.iloc[0]
    assert (other / wav).read_bytes() == (dataset / wav).read_bytes()
    assert set(pd.read_csv(dataset / "labels.csv").id).isdisjoint(metadata.loc[metadata.split == "test", "id"])
    with pytest.raises(ValueError, match="unmarked"):
        generate(tmp_path, 10)
    with pytest.raises(ValueError):
        synthetic_directory(tmp_path)
    with pytest.raises(ValueError):
        generate(tmp_path / "invalid", 0)


def test_audio_errors_and_silence(tmp_path):
    sf.write(tmp_path / "empty.wav", np.array([]), 16000)
    with pytest.raises(ValueError, match="Empty"):
        extract_recording(tmp_path / "empty.wav")
    sf.write(tmp_path / "silence.wav", np.zeros(16000), 16000)
    values = extract_recording(tmp_path / "silence.wav")
    assert values["voiced_ratio"] == 0
    assert np.isfinite(list(values.values())).all()
    assert values["pitch_mean"] == values["onset_rate"] == 0


@pytest.mark.parametrize("frequency", [80, 200, 480])
def test_pitch_tones(frequency):
    sr = 16000
    tone = .5 * np.sin(2 * np.pi * frequency * np.arange(sr) / sr)
    pitch = estimate_pitch(audio_frames(tone)[2:-2], sr)
    assert np.allclose(pitch, frequency, rtol=.02)


def test_pitch_unvoiced_and_errors():
    frames = audio_frames(np.zeros(16000))
    assert not estimate_pitch(frames, 16000).any()
    noise = np.random.default_rng(9).normal(0, .1, (8, 2048))
    assert not estimate_pitch(noise, 16000).any()
    assert estimate_pitch(np.empty((0, 2048)), 16000).size == 0
    for invalid in (np.array([]), np.zeros((1, 2)), np.full((1, 2048), np.nan)):
        with pytest.raises(ValueError):
            estimate_pitch(invalid, 16000)
    with pytest.raises(ValueError):
        estimate_pitch(frames, 16000, fmin=600, fmax=500)


def test_onset_bursts():
    sr = 16000
    t = np.arange(sr) / sr
    envelope = ((t > .2) & (t < .35)) | ((t > .6) & (t < .75))
    tone = .5 * np.sin(2 * np.pi * 200 * t) * envelope
    assert onset_rate(audio_frames(tone), sr, 1.) == 2


def test_no_librosa_gufunc_entrypoints(dataset, monkeypatch):
    import librosa
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden librosa gufunc entry point")
    for module, names in ((librosa, ["yin", "pyin", "zero_crossings"]),
                          (librosa.util, ["localmin", "localmax", "peak_pick"]),
                          (librosa.onset, ["onset_detect"])):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    metadata = pd.read_csv(dataset / "metadata.csv")
    assert np.isfinite(list(extract_recording(dataset / metadata.audio_path.iloc[0]).values())).all()


@pytest.fixture(scope="module")
def features(dataset):
    return build_features(dataset)


def test_features(features, tmp_path):
    assert len(features) == 10
    assert features.id.is_unique
    assert np.isfinite(features.select_dtypes("number")).all().all()
    assert "mfcc_0_mean_mean" in features
    with pytest.raises(ValueError, match="unmarked"):
        build_features(tmp_path)


@pytest.mark.parametrize("bad", ["missing", "nan", "inf", "duplicate", "wrong_ids", "strings", "count", "empty", "bool"])
def test_invalid_submission(bad):
    frame = pd.DataFrame({"id": ["a", "b"], "score": [1., 2.]})
    if bad == "missing":
        frame = frame.drop(columns="score")
    elif bad in ("nan", "inf"):
        frame.loc[0, "score"] = np.nan if bad == "nan" else np.inf
    elif bad == "duplicate":
        frame.loc[1, "id"] = "a"
    elif bad == "wrong_ids":
        frame.loc[1, "id"] = "c"
    elif bad == "strings":
        frame["score"] = ["one", "two"]
    elif bad == "bool":
        frame["score"] = [True, False]
    else:
        frame = frame.iloc[:0 if bad == "empty" else 1]
    with pytest.raises(ValueError):
        validate(frame, ["a", "b"])


def test_validator_positive():
    assert validate(pd.DataFrame({"child": ["b", "a"], "value": [2, 1]}), ["a", "b"], "child", "value")
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"id": [], "score": []}), [])


def test_fallback(monkeypatch):
    original = builtins.__import__
    def without_lightgbm(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated optional dependency absence")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_lightgbm)
    models, backend = estimators()
    assert backend == "sklearn_hist_gradient_boosting"
    models["boosting"].fit(np.arange(30).reshape(-1, 1), np.arange(30))
    assert np.isfinite(models["boosting"].predict([[2]])).all()


def test_end_to_end(features, dataset, tmp_path):
    report = train(features, pd.read_csv(dataset / "labels.csv"), tmp_path)
    assert report["train_children"] == 6
    assert set(report["models"]) == {"ridge", "boosting", "ensemble"}
    with (tmp_path / "model.pkl").open("rb") as stream:
        artifact = pickle.load(stream)
    test = features[features.split == "test"]
    prediction = artifact["model"].predict(test[artifact["columns"]])
    assert validate(pd.DataFrame({"id": test.id, "score": prediction}), test.id)
    features.to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"id": test.id, "score": np.nan}).to_csv(tmp_path / "bad.csv", index=False)
    result = subprocess.run([sys.executable, "-m", "wildfeatures.cli", "validate", "--features", str(tmp_path / "features.csv"), "--submission", str(tmp_path / "bad.csv")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "NaN" in result.stderr
