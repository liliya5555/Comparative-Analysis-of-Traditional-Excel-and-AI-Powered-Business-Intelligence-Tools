"""
cashflow_anomaly_detection.py
-----------------------------
Modular anomaly detection aligned with categories in the EI paper:
  - Duplicate Payments
  - Payment Delays
  - Pricing Discrepancies
  - Unauthorized Purchases
  - Fraud Indicators (multivariate)

Provides precision/recall/F1 evaluation if ground-truth labels are supplied.
Emits explainable flags per transaction plus weekly rollups for dashboards.

Dependencies: pandas, numpy, scikit-learn (for IsolationForest / LOF)

Copyright (c) 2025
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("cashflow.anomaly")

@dataclass
class DetectorConfig:
    iso_forest_estimators: int = 300
    iso_forest_contamination: float = 0.02
    lof_neighbors: int = 20
    delay_threshold_days: int = 0
    price_sigma: float = 3.0
    unauthorized_keywords: tuple = ("gift card", "misc", "other", "cash", "rounding")

class AnomalyDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self._iso = IsolationForest(
            n_estimators=self.cfg.iso_forest_estimators,
            contamination=self.cfg.iso_forest_contamination,
            random_state=42
        )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Basic assurances
        for c in ("invoice_id","invoice_date","due_date","payment_date","amount",
                  "customer_id","supplier_id","product_category","direction"):
            if c not in df.columns:
                df[c] = np.nan

        # feature table for multivariate
        feats = pd.DataFrame(index=df.index)
        feats["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        feats["days_to_pay"] = (pd.to_datetime(df["payment_date"]) - pd.to_datetime(df["invoice_date"])).dt.days.fillna(0)
        feats["days_past_due"] = (pd.to_datetime(df["payment_date"]) - pd.to_datetime(df["due_date"])).dt.days.fillna(0)
        # frequency encodings
        for col in ("customer_id", "supplier_id", "product_category"):
            counts = df[col].astype(str).value_counts()
            feats[f"{col}_freq"] = df[col].astype(str).map(counts).fillna(0)
        # temporal
        feats["month"] = pd.to_datetime(df["invoice_date"]).dt.month.fillna(0)
        feats["quarter"] = pd.to_datetime(df["invoice_date"]).dt.quarter.fillna(0)
        feats = feats.fillna(0)
        return df, feats

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df, feats = self._prepare(df)

        # Duplicate payments: same amount & supplier within small window
        window_days = 7
        df["dup_payment_flag"] = False
        key = df["supplier_id"].astype(str) + "|" + pd.to_numeric(df["amount"], errors="coerce").fillna(0).round(2).astype(str)
        order = pd.to_datetime(df["payment_date"])
        idx = df.index
        for k, grp in df.groupby(key):
            if len(grp) <= 1:
                continue
            gp = grp.sort_values("payment_date")
            dates = pd.to_datetime(gp["payment_date"])
            dup = (dates.diff().dt.days.abs() <= window_days).fillna(False)
            df.loc[gp.index, "dup_payment_flag"] = dup

        # Payment delays
        df["payment_delay_flag"] = (pd.to_datetime(df["payment_date"]) - pd.to_datetime(df["due_date"])).dt.days > self.cfg.delay_threshold_days

        # Pricing discrepancies: z-score by product category
        def _z_by_cat(group):
            m = group["amount"].mean()
            s = group["amount"].std(ddof=0) or 1.0
            return (group["amount"] - m) / s
        if "product_category" not in df.columns:
            df["product_category"] = "UNKNOWN"
        z = df.groupby("product_category", group_keys=False).apply(_z_by_cat)
        df["pricing_disc_flag"] = z.abs() >= self.cfg.price_sigma

        # Unauthorized purchases: keyword-based heuristic on free-text columns
        free_text = pd.Series("", index=df.index)
        for c in [c for c in df.columns if "desc" in c.lower() or "memo" in c.lower()]:
            free_text = free_text.str.cat(df[c].astype(str), sep=" ")
        df["unauth_flag"] = free_text.str.lower().str.contains("|".join(self.cfg.unauthorized_keywords), regex=True)

        # Fraud indicators: multivariate outlier
        feats = feats = self._prepare(df)[1]
        scores_iso = -self._iso.fit_predict(feats)  # 2 for outliers, 1 for inliers
        df["fraud_indicator_flag"] = (scores_iso > 1).astype(bool)

        # Roll-up category label (multi-hot to category names)
        def any_true(row, cols):
            return bool(np.any(row[cols].values))
        df["any_anomaly"] = df[["dup_payment_flag","payment_delay_flag","pricing_disc_flag","unauth_flag","fraud_indicator_flag"]].any(axis=1)

        return df

    @staticmethod
    def evaluate(df: pd.DataFrame, label_col: str = "is_anomaly") -> Dict:
        if label_col not in df.columns:
            raise ValueError(f"Ground-truth column '{label_col}' not present")
        y_true = df[label_col].astype(bool).values
        y_pred = df["any_anomaly"].astype(bool).values
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }

# --------------- CLI ---------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Anomaly detection for manufacturing cash-flow transactions")
    ap.add_argument("--transactions_parquet", required=True, help="Clean transactions parquet from ETL")
    ap.add_argument("--out", required=True, help="Path to write annotated parquet or json")
    ap.add_argument("--format", choices=["parquet","json"], default="parquet")
    ap.add_argument("--eval_label_col", default=None, help="Optional ground-truth label column for evaluation")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_parquet(args.transactions_parquet)
    det = AnomalyDetector()
    out_df = det.detect(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        out_df.to_parquet(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        out_df.to_json(args.out, orient="records", lines=False)
        print(f"Wrote {args.out}")
    if args.eval_label_col:
        metrics = det.evaluate(out_df, label_col=args.eval_label_col)
        print(json.dumps(metrics, indent=2))
