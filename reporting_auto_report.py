"""
reporting_auto_report.py
------------------------
Generates a static, publication-ready report (Markdown + images) that:
  - Describes data quality & weekly cash-flow dynamics
  - Summarizes forecasting backtest metrics
  - Visualizes anomaly counts and cluster composition
  - Captures EWS（Early Warning Signals）alert timeline

No external templating required; produces Markdown and PNG figures.
Dependencies: pandas, numpy, matplotlib

CLI
---
python reporting_auto_report.py \
  --weekly_parquet out/etl/cashflow_weekly.parquet \
  --forecast_json out/forecast/results.json \
  --anomaly_parquet out/anomaly/annotated.parquet \
  --cluster_parquet out/cluster/assignments.parquet \
  --ews_json out/ews/alerts.json \
  --outdir out/report

Notes
-----
- The script does not set specific colors/styles to keep consistency.
- All figures are saved as separate PNGs and referenced from the Markdown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_weekly_series(df: pd.DataFrame, outdir: Path) -> str:
    p = outdir / "weekly_net_cash.png"
    plt.figure()
    s = df.sort_values("week_start")["net_cash"].astype(float).reset_index(drop=True)
    plt.plot(s.values)
    plt.title("Weekly Net Cash")
    plt.xlabel("Weeks")
    plt.ylabel("Net Cash")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)

def plot_anomaly_counts(df: pd.DataFrame, outdir: Path) -> str:
    p = outdir / "anomaly_counts.png"
    plt.figure()
    cols = [c for c in df.columns if c.endswith("_flag")]
    counts = df[cols].astype(bool).sum().sort_values(ascending=False)
    counts.plot(kind="bar")
    plt.title("Anomaly Flags Count")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)

def plot_cluster_sizes(df: pd.DataFrame, outdir: Path) -> str:
    p = outdir / "cluster_sizes.png"
    plt.figure()
    df["cluster"].value_counts().sort_index().plot(kind="bar")
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)

def write_markdown(outdir: Path, weekly_img: str, anomaly_img: str, cluster_img: str,
                   forecast_json: dict | None, ews_json: dict | None):
    md = ["# Experiment Report",
          "",
          "## Overview",
          "This report aggregates ETL, forecasting, anomaly detection, clustering, and early warning outputs.",
          ""]
    if forecast_json:
        md.append("## Forecasting Backtest")
        for row in forecast_json.get("results", []):
            md.append(f"- Horizon {row['horizon']}: MAE={row['mae']:.2f}, MAPE={row['mape']:.2f}%, RMSE={row['rmse']:.2f}, "
                      f"DirAcc={row['dir_acc']:.1f}%, Bias={row['bias']:.2f}")
        md.append("")
    md.extend(["## Weekly Net Cash", f"![weekly]({Path(weekly_img).name})", ""])
    if anomaly_img:
        md.extend(["## Anomaly Categories", f"![anom]({Path(anomaly_img).name})", ""])
    if cluster_img:
        md.extend(["## Cluster Sizes", f"![cluster]({Path(cluster_img).name})", ""])
    if ews_json:
        md.append("## Early Warning Alerts (excerpt)")
        alerts = ews_json.get("alerts", [])[:10]
        for a in alerts:
            md.append(f"- {a['week']} | risk={a['risk_prob']:.2f} | label={a['label']}")
    (outdir / "index.md").write_text("\n".join(md), encoding="utf-8")

def _parse_args():
    ap = argparse.ArgumentParser(description="Static report generator (Markdown + PNG)")
    ap.add_argument("--weekly_parquet", required=True)
    ap.add_argument("--forecast_json", default=None)
    ap.add_argument("--anomaly_parquet", default=None)
    ap.add_argument("--cluster_parquet", default=None)
    ap.add_argument("--ews_json", default=None)
    ap.add_argument("--outdir", required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    outdir = Path(args.outdir); _ensure(outdir)

    weekly = pd.read_parquet(args.weekly_parquet)
    weekly_img = plot_weekly_series(weekly, outdir)

    anomaly_img = None
    if args.anomaly_parquet and Path(args.anomaly_parquet).exists():
        an = pd.read_parquet(args.anomaly_parquet)
        anomaly_img = plot_anomaly_counts(an, outdir)

    cluster_img = None
    if args.cluster_parquet and Path(args.cluster_parquet).exists():
        clu = pd.read_parquet(args.cluster_parquet)
        if "cluster" in clu.columns:
            cluster_img = plot_cluster_sizes(clu, outdir)

    forecast_json = None
    if args.forecast_json and Path(args.forecast_json).exists():
        forecast_json = json.loads(Path(args.forecast_json).read_text(encoding="utf-8"))

    ews_json = None
    if args.ews_json and Path(args.ews_json).exists():
        ews_json = json.loads(Path(args.ews_json).read_text(encoding="utf-8"))

    write_markdown(outdir, weekly_img, anomaly_img, cluster_img, forecast_json, ews_json)
    print(f"Report written to {outdir / 'index.md'}")
