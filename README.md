# Disease Prediction from Medical Data

An educational machine-learning project that estimates the probability of a binary disease outcome from structured patient data. It compares Logistic Regression, SVM, Random Forest, and XGBoost using stratified cross-validation.

## Included workflow

- Uses the breast-cancer dataset included with scikit-learn out of the box.
- Accepts a heart-disease or diabetes CSV. The target column must be named `target`, `Outcome`, `diagnosis`, `class`, or `label`.
- Treats missing numerical values with median imputation, encodes categorical features, and scales values where appropriate.
- Reports accuracy, precision, recall, F1, and ROC-AUC; selects the strongest ROC-AUC model and saves it.
- Provides a Streamlit interface for comparison and single-record risk estimates.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Train on a downloaded UCI-format CSV:

```powershell
python train.py --data data\heart.csv --output artifacts\heart_model.joblib
```

To create `data\\heart.csv` from the UCI Cleveland source file, run:

```powershell
python scripts\prepare_heart_csv.py
```

To create `data\\diabetes.csv` from the Pima Indians diabetes source file, run:

```powershell
python scripts\prepare_diabetes_csv.py
```

## Responsible use

This is a learning and decision-support prototype, not a medical device. Its estimates must not be used to diagnose, treat, or make clinical decisions without qualified medical oversight.
