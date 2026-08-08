"""Convert the Pima Indians diabetes source data to a documented CSV."""

from pathlib import Path

import pandas as pd


COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
MISSING_WHEN_ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

SOURCE = Path("data/pima-indians-diabetes.data.csv")
OUTPUT = Path("data/diabetes.csv")


def main() -> None:
    frame = pd.read_csv(SOURCE, names=COLUMNS)
    # A zero in these measurements represents an unavailable value, not a valid result.
    frame[MISSING_WHEN_ZERO] = frame[MISSING_WHEN_ZERO].replace(0, pd.NA)
    frame.to_csv(OUTPUT, index=False)
    print(f"Created {OUTPUT} with {len(frame)} rows and {frame.shape[1] - 1} features.")
    print("Outcome counts:", frame["Outcome"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
