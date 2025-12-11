"""
early_warning_signal_generator.py
---------------------------------
Generates early-warning signals (EWS) for near-term cash shortfall risk
using weekly aggregates. Trains a classifier to predict whether the next
week's net cash will be negative (or below a configurable percentile),
provides threshold-based alerts, and exports a ranked alert list.

Models
------
- LogisticRegression (baseline, robust & interpretable)
- RandomForest (nonlinear)
- XGBoost (optional, if installed)
- SHAP values (optional; falls back to permutation importances if SHAP unavailable)

Output
------
- alerts.json with top-K alerting weeks, predicted probability, and label
- feature_importance.json (global) + per-instance explanations when enabled

CLI
---
python early_warning_signal_generator.py --weekly_parquet out/etl/cashflow_weekly.parquet \
    --out_json out/ews/alerts.json --threshold 0.6 --topk 20

Copyright (c) 2025
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

def build_features(df: pd.DataFrame, target_col: str = "net_cash", lags: int = 8) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy().sort_values("week_start")
    y = (df[target_col].shift(-1) < 0).astype(int)  # next-week negative
    X = pd.DataFrame(index=df.index)
    s = df[target_col].astype(float)
    for L in range(1, lags + 1):
        X[f"lag_{L}"] = s.shift(L)
    for w in (2,4,8):
        X[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        X[f"roll_std_{w}"] = s.shift(1).rolling(w).std()
    if "invoices" in df.columns:
        X["invoices"] = df["invoices"]
    if "late_rate" in df.columns:
        X["late_rate"] = df["late_rate"]
    X = X.dropna()
    y = y.loc[X.index]
    weeks = pd.to_datetime(df["week_start"]).loc[X.index]
    return X, y, weeks

def fit_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
    models = {}
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe_lr.fit(X, y)
    models["LogReg"] = pipe_lr

    rf = RandomForestClassifier(n_estimators=400, random_state=42, max_depth=None, min_samples_leaf=2)
    rf.fit(X, y)
    models["RF"] = rf

    if _HAS_XGB:
        models["XGB"] = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42
        ).fit(X, y)
    return models

def evaluate_models(models: Dict[str, any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for name, m in models.items():
        proba = m.predict_proba(X)[:,1] if hasattr(m, "predict_proba") else m.decision_function(X)
        metrics[name] = {
            "AUC": float(roc_auc_score(y, proba)),
            "AP": float(average_precision_score(y, proba))
        }
    return metrics

def compute_importance(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    if _HAS_SHAP:
        try:
            explainer = shap.Explainer(model, X, feature_names=X.columns)
            sv = explainer(X)
            imp = np.abs(sv.values).mean(axis=0)
            return {f: float(v) for f, v in zip(X.columns, imp)}
        except Exception:
            pass
    # Fallback to permutation importance
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    return {X.columns[i]: float(v) for i, v in enumerate(r.importances_mean)}

def generate_alerts(weekly_parquet: str, out_json: str, threshold: float = 0.6, topk: int = 20) -> Dict:
    df = pd.read_parquet(weekly_parquet)
    X, y, weeks = build_features(df)
    models = fit_models(X, y)
    metrics = evaluate_models(models, X, y)
    # pick best by AP
    best = max(metrics.items(), key=lambda kv: kv[1]["AP"])[0]
    model = models[best]

    proba = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X)
    alerts = []
    for wk, p, lab in zip(weeks, proba, y):
        if p >= threshold:
            alerts.append({"week": str(wk.date()), "risk_prob": float(p), "label": int(lab)})
    alerts = sorted(alerts, key=lambda a: a["risk_prob"], reverse=True)[:topk]

    # importance
    imp = compute_importance(model, X, y)

    out = {"model": best, "metrics": metrics[best], "threshold": float(threshold), "alerts": alerts, "feature_importance": imp}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out

def _parse_args():
    ap = argparse.ArgumentParser(description="Early-warning signal generator (weekly)")
    ap.add_argument("--weekly_parquet", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=20)
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    res = generate_alerts(args.weekly_parquet, args.out_json, threshold=args.threshold, topk=args.topk)
    print(json.dumps(res, indent=2))
