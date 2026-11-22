import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from features.acoustic import build_features
from wildfeatures.data import generate, synthetic_directory
from wildfeatures.model import train
from wildfeatures.submission import validate


def main():
    parser = argparse.ArgumentParser(description="Local synthetic pipeline; never downloads or submits data")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data.add_argument("--out", default="data/synthetic")
    data.add_argument("--n-children", type=int, default=48)
    data.add_argument("--seed", type=int, default=42)
    features = commands.add_parser("features")
    features.add_argument("--root", default="data/synthetic")
    features.add_argument("--out", default="outputs/features.csv")
    features.add_argument("--config")
    features.add_argument("--real-local", action="store_true", help="Explicit opt-in for user's own local execution only")
    fit = commands.add_parser("train")
    fit.add_argument("--features", default="outputs/features.csv")
    fit.add_argument("--root", default="data/synthetic")
    fit.add_argument("--labels", default="labels.csv")
    fit.add_argument("--out", default="outputs")
    fit.add_argument("--real-local", action="store_true")
    submit = commands.add_parser("submit")
    submit.add_argument("--features", default="outputs/features.csv")
    submit.add_argument("--model", default="outputs/model.pkl")
    submit.add_argument("--out", default="outputs/submission.csv")
    check = commands.add_parser("validate")
    check.add_argument("--submission", default="outputs/submission.csv")
    check.add_argument("--features", default="outputs/features.csv")
    for command in (submit, check):
        command.add_argument("--id-column", default="id")
        command.add_argument("--score-column", default="score")
    args = parser.parse_args()
    try:
        if args.command == "data":
            generate(args.out, args.n_children, args.seed)
            print(f"Generated SYNTHETIC data: {args.out}")
        elif args.command == "features":
            config = json.loads(Path(args.config).read_text()) if args.config else None
            result = build_features(args.root, config, synthetic=not args.real_local)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.out, index=False)
            print(f"Features: {len(result)} children -> {args.out}")
        elif args.command == "train":
            root = Path(args.root) if args.real_local else synthetic_directory(args.root)
            labels = pd.read_csv(root / args.labels)
            if not args.real_local and ("is_synthetic" not in labels or not labels.is_synthetic.eq(True).all()):
                raise ValueError("Missing synthetic label markers")
            features = pd.read_csv(args.features)
            if not args.real_local and ("is_synthetic" not in features or not features.is_synthetic.eq(True).all()):
                raise ValueError("Missing synthetic feature markers")
            print(json.dumps(train(features, labels, args.out), indent=2))
        else:
            features = pd.read_csv(args.features)
            test = features[features.split == "test"]
            if args.command == "submit":
                with open(args.model, "rb") as f:
                    artifact = pickle.load(f)
                result = pd.DataFrame({args.id_column: test.id,
                                       args.score_column: artifact["model"].predict(test[artifact["columns"]])})
                validate(result, test.id, args.id_column, args.score_column)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(args.out, index=False)
                print(f"Local predictions only: {args.out} ({len(result)} rows)")
            else:
                validate(pd.read_csv(args.submission), test.id, args.id_column, args.score_column)
                print(f"Submission validation OK ({len(test)} rows)")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
