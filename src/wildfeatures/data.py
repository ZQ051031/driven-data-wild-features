"""Deterministic synthetic-only data creation and safety boundaries."""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import iirpeak, lfilter

MARKER = "SYNTHETIC_ONLY.txt"


def synthetic_directory(path, create=False):
    path = Path(path).resolve()
    if create and not path.exists():
        path.mkdir(parents=True)
        (path / MARKER).write_text("SYNTHETIC data only. No competition data.\n")
    if not (path / MARKER).is_file():
        raise ValueError(f"Refusing unmarked synthetic directory: {path}")
    return path


def generate(out, n_children=48, seed=42):
    if n_children < 10:
        raise ValueError("At least 10 children required for train/validation/test splits")
    out = synthetic_directory(out, create=True)
    rng = np.random.default_rng(seed)
    sr = 16000
    records, labels = [], []
    order = rng.permutation(n_children)
    validation = set(order[:max(2, n_children // 5)])
    test = set(order[max(2, n_children // 5):2 * max(2, n_children // 5)])
    for child in range(n_children):
        ability = rng.normal()
        child_id = f"synthetic_{child:04d}"
        split = "validation" if child in validation else "test" if child in test else "train"
        scores = {f"synthetic_{name}": float(50 + 12 * ability + rng.normal(0, 5))
                  for name in ("letter_naming", "phoneme_segments", "sound_matching")}
        pitch = rng.uniform(180, 290)
        rate = np.clip(3 + ability * .45, 1.8, 4.5)
        for task in range(2):
            parts = []
            for syllable in range(5):
                duration = 0.65 / rate * rng.uniform(.8, 1.2)
                t = np.arange(int(sr * duration)) / sr
                f0 = pitch * (1 + .08 * np.sin(2 * np.pi * 3 * t) + .04 * task)
                phase = 2 * np.pi * np.cumsum(f0) / sr
                voice = sum(np.sin(k * phase) / k for k in range(1, 9))
                b, a = iirpeak(rng.choice([650, 1000, 1600]) / (sr / 2), 2)
                voice = .45 * voice + lfilter(b, a, voice)
                voice *= np.sin(np.linspace(0, np.pi, len(t))) ** .6
                pause = np.clip(.12 - ability * .025 + rng.normal(0, .015), .025, .23)
                parts.extend([voice, np.zeros(int(sr * pause))])
            audio = np.concatenate(parts)
            audio += rng.normal(0, .008, len(audio))
            audio = .85 * audio / max(np.max(np.abs(audio)), 1)
            filename = f"{child_id}_task{task}.wav"
            sf.write(out / filename, audio, sr)
            records.append(dict(id=child_id, task=f"synthetic_task_{task}", audio_path=filename,
                                split=split, is_synthetic=True, **scores))
        if split != "test":
            labels.append(dict(id=child_id, score=100 + 15 * ability + rng.normal(0, 4), is_synthetic=True))
    pd.DataFrame(records).to_csv(out / "metadata.csv", index=False)
    pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
    (out / "README.md").write_text("# SYNTHETIC ONLY\nGenerated speech-like tones, not children or competition recordings.\nTest targets intentionally withheld.\n")
    return out
