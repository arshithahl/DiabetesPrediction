#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train all models (KNN, SVM, Random Forest, Logistic Regression) on merged CSV datasets.

Usage (PowerShell):
  python prediction/ml_models/train_all_models.py --data-dir data --test-size 0.2 --seed 42

Requirements:
  - CSV files must share the same schema with columns:
    Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
  - Place one or more CSVs inside the given --data-dir folder. They will be concatenated.

Artifacts written to prediction/ml_models/:
  - knn.pkl, svm.pkl, rf.pkl, logreg.pkl
  - scaler_knn.pkl, scaler_svm.pkl, scaler_logreg.pkl (RF does not require a scaler)
  - models_summary.json (metrics per algorithm)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET_COLUMN = 'Outcome'


def load_and_merge_csvs(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted([p for p in data_dir.glob('*.csv') if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir!s}. Place CSVs with required columns.")
    dfs = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"File {csv} missing columns: {sorted(missing)}")
        dfs.append(df[FEATURE_COLUMNS + [TARGET_COLUMN]])
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged = merged.dropna(axis=0).reset_index(drop=True)
    return merged


def compute_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    except Exception:
        auc = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1': round(f1 * 100, 2),
        'roc_auc': round(float(auc), 3) if auc is not None else None,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }


def train_and_save_models(df: pd.DataFrame, out_dir: Path, seed: int, test_size: float):
    X = df[FEATURE_COLUMNS].astype(float).values
    y = df[TARGET_COLUMN].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # KNN (with scaler)
    scaler_knn = StandardScaler()
    X_train_knn = scaler_knn.fit_transform(X_train)
    X_test_knn = scaler_knn.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train)
    y_prob_knn = None
    try:
        y_prob_knn = knn.predict_proba(X_test_knn)[:, 1]
    except Exception:
        pass
    y_pred_knn = knn.predict(X_test_knn)
    summary['knn'] = compute_metrics(y_test, y_prob_knn, y_pred_knn)
    joblib.dump(knn, out_dir / 'knn.pkl')
    joblib.dump(scaler_knn, out_dir / 'scaler_knn.pkl')

    # SVM (with scaler, enable probability)
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_test_svm = scaler_svm.transform(X_test)
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    svm.fit(X_train_svm, y_train)
    y_prob_svm = svm.predict_proba(X_test_svm)[:, 1]
    y_pred_svm = svm.predict(X_test_svm)
    summary['svm'] = compute_metrics(y_test, y_prob_svm, y_pred_svm)
    joblib.dump(svm, out_dir / 'svm.pkl')
    joblib.dump(scaler_svm, out_dir / 'scaler_svm.pkl')

    # Random Forest (no scaler needed)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    summary['rf'] = compute_metrics(y_test, y_prob_rf, y_pred_rf)
    joblib.dump(rf, out_dir / 'rf.pkl')

    # Logistic Regression (with scaler)
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_test_lr = scaler_lr.transform(X_test)
    lr = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=None)
    lr.fit(X_train_lr, y_train)
    y_prob_lr = lr.predict_proba(X_test_lr)[:, 1]
    y_pred_lr = lr.predict(X_test_lr)
    summary['logreg'] = compute_metrics(y_test, y_prob_lr, y_pred_lr)
    joblib.dump(lr, out_dir / 'logreg.pkl')
    joblib.dump(scaler_lr, out_dir / 'scaler_logreg.pkl')

    # Write summary JSON
    with open(out_dir / 'models_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Merge CSV datasets and retrain models.')
    parser.add_argument('--data-dir', type=str, required=True, help='Folder containing CSV files to merge.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction (default 0.2).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default 42).')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / 'prediction' / 'ml_models'

    df = load_and_merge_csvs(Path(args.data_dir))
    summary = train_and_save_models(df, output_dir, seed=args.seed, test_size=args.test_size)

    print('Training complete. Metrics:')
    print(json.dumps(summary, indent=2))
    print(f'Artifacts saved to: {output_dir}')


if __name__ == '__main__':
    main()
