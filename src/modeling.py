"""Reproducible pipelines for evaluating medical risk classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42


@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    test_features: pd.DataFrame
    test_target: pd.Series


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")


def classifiers() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM": CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9,
                                           colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_STATE)
    except ImportError:
        pass
    return models


def evaluate_models(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compare models using stratified cross-validation, avoiding optimistic train metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, classifier in classifiers().items():
        pipeline = Pipeline([("preprocess", build_preprocessor(features)), ("model", classifier)])
        scores = cross_validate(pipeline, features, target, cv=cv,
                                scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"})
        rows.append({"Model": name, **{metric.replace("test_", "").upper(): float(values.mean())
                                        for metric, values in scores.items() if metric.startswith("test_")}})
    return pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)


def train_best_model(features: pd.DataFrame, target: pd.Series, model_name: str) -> TrainingResult:
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE)
    pipeline = Pipeline([("preprocess", build_preprocessor(features)), ("model", classifiers()[model_name])])
    pipeline.fit(train_x, train_y)
    prediction = pipeline.predict(test_x)
    probability = pipeline.predict_proba(test_x)[:, 1]
    return TrainingResult(model_name, pipeline, {
        "Accuracy": accuracy_score(test_y, prediction), "Precision": precision_score(test_y, prediction, zero_division=0),
        "Recall": recall_score(test_y, prediction, zero_division=0), "F1": f1_score(test_y, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(test_y, probability),
    }, test_x, test_y)
