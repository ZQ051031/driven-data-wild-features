from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

from wildfeatures.data import synthetic_directory


def audio_frames(y, frame_length=2048, hop_length=512):
    padded = np.pad(y, (frame_length // 2, frame_length // 2))
    return np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]


def estimate_pitch(frames, sr, fmin=75, fmax=500, threshold=.65):
    """Normalized autocorrelation pitch; unvoiced frames return zero Hz."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2 or frames.shape[1] < 2 or not np.isfinite(frames).all():
        raise ValueError("Pitch requires finite two-dimensional audio frames")
    if not 0 < fmin < fmax < sr / 2 or not 0 < threshold <= 1:
        raise ValueError("Invalid pitch frequency range or voicing threshold")
    first = int(np.ceil(sr / fmax))
    last = min(int(np.floor(sr / fmin)), frames.shape[1] - 2)
    if last <= first:
        raise ValueError("Audio frames too short for pitch range")
    pitch = np.zeros(len(frames))
    for i, frame in enumerate(frames):
        frame = frame - frame.mean()
        if np.sqrt(np.mean(frame ** 2)) < .02:
            continue
        correlation = correlate(frame, frame, mode="full", method="fft")[len(frame) - 1:]
        energy = np.r_[0., np.cumsum(frame ** 2)]
        lags = np.arange(first, last + 1)
        denominator = np.sqrt(energy[len(frame) - lags] * (energy[-1] - energy[lags]))
        normalized = correlation[lags] / np.maximum(denominator, np.finfo(float).eps)
        peaks, _ = find_peaks(np.r_[-np.inf, normalized, -np.inf])
        peaks = peaks - 1
        candidates = peaks[normalized[peaks] >= max(threshold, normalized.max() * .95)]
        if candidates.size:
            pitch[i] = sr / lags[candidates[0]]
    return pitch


def onset_rate(frames, sr, duration, hop_length=512):
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1))
    flux = np.maximum(np.diff(spectrum, axis=0), 0).mean(axis=1)
    rising_energy = np.diff(np.mean(frames ** 2, axis=1)) > 0
    flux = np.where(rising_energy, flux, 0.)
    if not flux.size or flux.max() <= np.finfo(float).eps:
        return 0.
    peaks, _ = find_peaks(flux, prominence=max(float(flux.max()) * .15, .01),
                          distance=max(1, int(.1 * sr / hop_length)))
    return len(peaks) / duration


def extract_recording(path):
    y, sr = librosa.load(path, sr=16000)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f"Empty or invalid audio: {path}")
    frames = audio_frames(y)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    pitch = estimate_pitch(frames, sr)
    voiced = (pitch > 0) & (rms > max(.02, float(rms.max()) * .15))
    pitch = pitch[voiced]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {f"mfcc_{i}_{stat}": float(fn(row)) for i, row in enumerate(mfcc)
                for stat, fn in (("mean", np.mean), ("std", np.std))}
    for name, values in {
        "rms": rms, "pitch": pitch,
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "flatness": librosa.feature.spectral_flatness(y=y),
        "zcr": np.mean(np.diff(np.signbit(frames), axis=1), axis=1),
    }.items():
        features[f"{name}_mean"] = float(np.mean(values)) if values.size else 0.
        features[f"{name}_std"] = float(np.std(values)) if values.size else 0.
    duration = len(y) / sr
    pauses = np.diff(np.r_[False, ~voiced, False].astype(int))
    lengths = (np.where(pauses == -1)[0] - np.where(pauses == 1)[0]) * 512 / sr
    features.update(duration=duration, voiced_ratio=float(voiced.mean()),
                    pause_count=len(lengths), pause_mean=float(lengths.mean()) if len(lengths) else 0.,
                    onset_rate=onset_rate(frames, sr, duration))
    return features


def build_features(root, config=None, synthetic=True):
    root = synthetic_directory(root) if synthetic else Path(root).resolve()
    config = config or {}
    metadata = pd.read_csv(root / config.get("metadata", "metadata.csv"))
    metadata = metadata.rename(columns={v: k for k, v in config.get("columns", {}).items()})
    if synthetic and ("is_synthetic" not in metadata or not metadata.is_synthetic.eq(True).all()):
        raise ValueError("Missing synthetic row markers")
    tabular = config.get("tabular", ["synthetic_letter_naming", "synthetic_phoneme_segments", "synthetic_sound_matching"])
    rows = []
    for row in metadata.to_dict("records"):
        audio = (root / row["audio_path"]).resolve()
        if synthetic and not audio.is_relative_to(root):
            raise ValueError("Synthetic audio must stay inside marked directory")
        rows.append(dict(id=row["id"], **extract_recording(audio)))
    acoustic = pd.DataFrame(rows).groupby("id").agg(["mean", "std"]).fillna(0)
    acoustic.columns = ["_".join(c) for c in acoustic.columns]
    if metadata.groupby("id")[tabular + ["split"]].nunique().gt(1).any().any():
        raise ValueError("Inconsistent child-level tabular values or splits")
    children = metadata.drop_duplicates("id")[["id", "split"] + tabular].set_index("id")
    result = children.join(acoustic).reset_index()
    if synthetic:
        result["is_synthetic"] = True
    return result
