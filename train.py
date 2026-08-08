"""Train and persist the best disease-prediction classifier."""

import argparse
from pathlib import Path

import joblib

from src.data import load_breast_cancer, load_csv
from src.modeling import evaluate_models, train_best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare and train medical-data classifiers.")
    parser.add_argument("--data", help="Optional path to a heart or diabetes CSV.")
    parser.add_argument("--output", default="artifacts/disease_model.joblib", help="Saved model path.")
    args = parser.parse_args()
    dataset = load_csv(args.data, Path(args.data).stem) if args.data else load_breast_cancer()
    comparison = evaluate_models(dataset.features, dataset.target)
    print(f"\n{dataset.name} model comparison (5-fold cross-validation):")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    result = train_best_model(dataset.features, dataset.target, comparison.iloc[0]["Model"])
    print(f"\nSelected: {result.model_name}")
    print("Hold-out metrics:", {name: round(value, 3) for name, value in result.metrics.items()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result.pipeline, "feature_names": list(dataset.features.columns), "dataset": dataset.name}, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
