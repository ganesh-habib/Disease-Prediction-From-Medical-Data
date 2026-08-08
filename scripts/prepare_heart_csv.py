"""Convert UCI's Cleveland heart-disease data into the project's CSV format."""

from pathlib import Path

import pandas as pd


COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SOURCE = Path("data/processed.cleveland.data")
OUTPUT = Path("data/heart.csv")


def main() -> None:
    frame = pd.read_csv(SOURCE, names=COLUMNS, na_values="?")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    # UCI encodes 0 as no disease and 1-4 as disease present.
    frame["target"] = (frame.pop("num") > 0).astype(int)
    frame.to_csv(OUTPUT, index=False)
    print(f"Created {OUTPUT} with {len(frame)} rows and {frame.shape[1] - 1} features.")
    print("Target counts:", frame["target"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
