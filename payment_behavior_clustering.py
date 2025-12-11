"""
payment_behavior_clustering.py
------------------------------
Supplier/customer payment behavior clustering from transaction-level data.
Produces cluster assignments, centroids, and interpretability tables that
map to RFM-style features and delay/amount distributions.

Features
--------
- Entity choice: customer_id or supplier_id (auto-detected)
- RFM (Recency, Frequency, Monetary) + Delay statistics
- KMeans baseline; silhouette & Calinski-Harabasz indices
- Optional robust scaler and PCA for noise reduction
- Outputs Parquet with cluster labels + summary CSV/JSON

Dependencies: pandas, numpy, scikit-learn

CLI
---
python payment_behavior_clustering.py --transactions_parquet out/etl/transactions_clean.parquet \
    --entity supplier_id --n_clusters 6 --out_parquet out/cluster/assignments.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def build_entity_features(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    df["days_to_pay"] = (pd.to_datetime(df["payment_date"]) - pd.to_datetime(df["invoice_date"])).dt.days
    df["days_past_due"] = (pd.to_datetime(df["payment_date"]) - pd.to_datetime(df["due_date"])).dt.days
    now_ref = pd.to_datetime(df["invoice_date"]).max()

    g = df.groupby(entity)
    feats = pd.DataFrame({
        "freq": g["invoice_id"].nunique(),
        "monetary": g["amount"].sum(),
        "avg_amount": g["amount"].mean(),
        "std_amount": g["amount"].std().fillna(0),
        "avg_days_to_pay": g["days_to_pay"].mean(),
        "late_ratio": (g["days_past_due"].apply(lambda s: (s > 0).mean())),
        "recency_days": (now_ref - g["invoice_date"].max()).dt.days
    })
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats.reset_index()

def kmeans_cluster(feats: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, Dict]:
    X = feats.select_dtypes(include=[np.number]).values
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    # PCA optional (retain 95% variance up to 6 comps)
    pca = PCA(n_components=min(6, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Xp)

    # Metrics
    sil = silhouette_score(Xp, labels) if n_clusters > 1 else np.nan
    ch = calinski_harabasz_score(Xp, labels) if n_clusters > 1 else np.nan

    out = feats.copy()
    out["cluster"] = labels
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=[f"pc{i+1}" for i in range(Xp.shape[1])])
    return out, {"silhouette": float(sil), "calinski_harabasz": float(ch), "centers_pca": centers.to_dict(orient="records")}

def cluster_from_transactions(transactions_parquet: str, out_parquet: str, entity: str = "supplier_id", n_clusters: int = 6) -> Dict:
    df = pd.read_parquet(transactions_parquet)
    if entity not in df.columns:
        # fallback to customer_id if available
        entity = "customer_id" if "customer_id" in df.columns else "supplier_id"
    feats = build_entity_features(df, entity)
    labels, info = kmeans_cluster(feats, n_clusters=n_clusters)

    outp = Path(out_parquet); outp.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(outp, index=False)

    # simple summary
    summary = labels.groupby("cluster").agg({"freq":"mean","monetary":"mean","avg_days_to_pay":"mean","late_ratio":"mean"}).reset_index()
    summary_path = outp.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    return {"rows": int(len(labels)), "entity": entity, "out_parquet": str(outp), "summary_csv": str(summary_path), **info}

def _parse_args():
    ap = argparse.ArgumentParser(description="Payment behavior clustering")
    ap.add_argument("--transactions_parquet", required=True)
    ap.add_argument("--entity", default="supplier_id")
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--out_parquet", required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    result = cluster_from_transactions(args.transactions_parquet, args.out_parquet, entity=args.entity, n_clusters=args.n_clusters)
    print(result)
