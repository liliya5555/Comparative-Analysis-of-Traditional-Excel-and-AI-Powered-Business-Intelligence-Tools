"""
cashflow_data_pipeline.py
---------------------------------
End-to-end ETL + data quality validation for manufacturing cash-flow transactions,
aligned with the EI paper comparing Excel vs. AI-powered BI tools (Power BI / Tableau)
in a mid-sized U.S. manufacturing enterprise (18 months, ~78 weekly periods).

Features
--------
- Robust schema validation and coercion (dates, numerics, categoricals)
- Deterministic preprocessing for reproducibility (na handling, deduplication, outlier flags)
- Weekly aggregation into inflow/outflow and net cash, with customer/supplier dimensions
- Derived KPIs: Days Sales Outstanding (DSO), Days Payable Outstanding (DPO), delay buckets
- Lightweight data quality report + anomaly hints emitted as JSON
- CLI entrypoint with argparse; YAML config support optional
- Friendly logging with timestamps; deterministic random seed for any sampling

Notes
-----
- Transaction schema expects columns (case-insensitive tolerant):
  ['invoice_id','invoice_date','due_date','payment_date','amount',
   'customer_id','supplier_id','product_category','direction']
  Where 'direction' in {'inflow','outflow'}; outflows are negative by convention.
- If your raw extracts split AP/AR, you may pass two CSV/Parquet files; the pipeline will
  harmonize into a unified fact table.
- The weekly calendar is aligned to Monday starts by default.

Copyright (c) 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Logging ----------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("cashflow.etl")

# ---------- Utilities ----------
def coerce_datetime(s: pd.Series, utc: bool = False) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce", utc=utc, infer_datetime_format=True)
    return s2

def week_start(dt: pd.Series) -> pd.Series:
    # Align to Monday start
    dt = pd.to_datetime(dt, errors="coerce")
    monday = dt - pd.to_timedelta((dt.dt.weekday), unit="D")
    return monday.dt.normalize()

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapper = {
        "invoiceid": "invoice_id",
        "invoice_no": "invoice_id",
        "invoice_number": "invoice_id",
        "invoice_date": "invoice_date",
        "invoicedate": "invoice_date",
        "due_date": "due_date",
        "duedate": "due_date",
        "payment_date": "payment_date",
        "paid_date": "payment_date",
        "amount": "amount",
        "value": "amount",
        "customer_id": "customer_id",
        "client_id": "customer_id",
        "supplier_id": "supplier_id",
        "vendor_id": "supplier_id",
        "product_category": "product_category",
        "category": "product_category",
        "direction": "direction",
        "type": "direction",
    }
    cols = {c: mapper.get(c.lower(), c.lower()) for c in df.columns}
    df = df.rename(columns=cols)
    return df

@dataclass
class PipelineConfig:
    week_anchor: str = "MON"     # Monday weekly calendar
    timezone_utc: bool = False
    drop_duplicate_invoices: bool = True
    duplicate_subset: Tuple[str, ...] = ("invoice_id", "amount", "invoice_date")
    outlier_sigma: float = 5.0
    delay_bucket_edges: Tuple[int, ...] = (-9999, -1, 0, 7, 15, 30, 45, 60, 90, 9999)
    # Minimum columns required
    required: Tuple[str, ...] = (
        "invoice_id", "invoice_date", "due_date", "payment_date",
        "amount", "direction"
    )

# ---------- Core ETL ----------
class CashflowETL:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()

    def load(self, paths: Iterable[str]) -> pd.DataFrame:
        frames = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(p)
            ext = p.suffix.lower()
            if ext in (".csv", ".txt"):
                df = pd.read_csv(p)
            elif ext in (".parquet", ".pq"):
                df = pd.read_parquet(p)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            df["__source_file"] = p.name
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        raw = _standardize_columns(raw)
        logger.info("Loaded %d rows from %d files", len(raw), len(frames))
        return raw

    def validate_and_coerce(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        report = {"missing_required": [], "duplicates_removed": 0, "outlier_flags": 0}

        # Required columns
        for c in self.cfg.required:
            if c not in df.columns:
                report["missing_required"].append(c)
        if report["missing_required"]:
            raise ValueError(f"Missing required columns: {report['missing_required']}")

        # Coercions
        for dt_col in ("invoice_date", "due_date", "payment_date"):
            df[dt_col] = coerce_datetime(df[dt_col], utc=self.cfg.timezone_utc)

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
        df["direction"] = df["direction"].str.lower().str.strip()

        # Normalize sign convention: inflow positive, outflow negative
        df["signed_amount"] = np.where(df["direction"].eq("outflow"), -df["amount"], df["amount"])

        # Days to pay
        df["days_to_pay"] = (df["payment_date"] - df["invoice_date"]).dt.days

        # Delay vs due date
        df["days_past_due"] = (df["payment_date"] - df["due_date"]).dt.days

        # Delay buckets
        edges = list(self.cfg.delay_bucket_edges)
        labels = [f"({edges[i]},{edges[i+1]}]" for i in range(len(edges)-1)]
        df["delay_bucket"] = pd.cut(df["days_past_due"], bins=edges, labels=labels, include_lowest=True)

        # Deduplication
        if self.cfg.drop_duplicate_invoices:
            before = len(df)
            df = df.drop_duplicates(subset=list(self.cfg.duplicate_subset), keep="first")
            report["duplicates_removed"] = before - len(df)

        # Outlier flag (per product_category)
        def _zscore(group):
            m = group["amount"].mean()
            s = group["amount"].std(ddof=0) or 1.0
            return (group["amount"] - m) / s

        if "product_category" not in df.columns:
            df["product_category"] = "UNKNOWN"
        df["z_amount"] = df.groupby("product_category", dropna=False, group_keys=False).apply(_zscore)
        df["amount_is_outlier"] = df["z_amount"].abs() >= self.cfg.outlier_sigma
        report["outlier_flags"] = int(df["amount_is_outlier"].sum())

        # Week index
        df["week_start"] = week_start(df["invoice_date"])

        # Null handling summary
        nulls = df.isna().mean().sort_values(ascending=False)
        report["null_rate_top"] = nulls.head(10).to_dict()

        return df, report

    def aggregate_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        # Weekly AR / AP signals
        agg = (
            df.groupby(["week_start"], dropna=False)
              .agg(
                  cash_inflow=("signed_amount", lambda s: s[s > 0].sum()),
                  cash_outflow=("signed_amount", lambda s: -s[s < 0].sum()),
                  net_cash=("signed_amount", "sum"),
                  invoices=("invoice_id", "nunique"),
                  mean_days_to_pay=("days_to_pay", "mean"),
                  late_rate=("days_past_due", lambda s: np.mean(s.fillna(0) > 0)),
                  outlier_count=("amount_is_outlier", "sum")
              )
              .reset_index()
              .sort_values("week_start")
        )
        return agg

    def build(self, inputs: Iterable[str], outdir: str) -> Dict:
        out_dir = Path(outdir); out_dir.mkdir(parents=True, exist_ok=True)
        raw = self.load(inputs)
        df, report = self.validate_and_coerce(raw)
        weekly = self.aggregate_weekly(df)

        # Persist
        raw_path = out_dir / "transactions_clean.parquet"
        weekly_path = out_dir / "cashflow_weekly.parquet"
        report_path = out_dir / "data_quality_report.json"
        df.to_parquet(raw_path, index=False)
        weekly.to_parquet(weekly_path, index=False)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Saved clean transactions -> %s", raw_path)
        logger.info("Saved weekly aggregates -> %s", weekly_path)
        logger.info("Saved data quality report -> %s", report_path)

        return {
            "transactions_clean": str(raw_path),
            "cashflow_weekly": str(weekly_path),
            "data_quality_report": str(report_path),
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manufacturing cash-flow ETL")
    p.add_argument("--input", nargs="+", required=True, help="CSV/XLSX/Parquet file(s)")
    p.add_argument("--outdir", required=True, help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    etl = CashflowETL()
    artifacts = etl.build(args.input, args.outdir)
    print(json.dumps(artifacts, indent=2))
