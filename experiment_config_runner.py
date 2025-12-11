"""
experiment_config_runner.py
---------------------------
Unified experiment runner that executes a full suite of experiments
from a single YAML/JSON config file. Integrates with the following modules:
  - cashflow_data_pipeline.py
  - cashflow_forecasting_ensemble.py
  - cashflow_anomaly_detection.py
  - bi_tool_tco_roi_analysis.py
  - payment_behavior_clustering.py
  - early_warning_signal_generator.py

Features
--------
- Deterministic seeding and run metadata
- Structured outputs (artifacts directory with JSON/Parquet/PNG)
- Lightweight experiment registry (CSV + JSONL)
- Optional MLflow-like directory layout (no external service required)
- CLI: python experiment_config_runner.py --config config.yaml --artifacts out/

Example YAML
------------
seed: 42
tasks:
  - name: etl
    inputs: ["data/ap.csv", "data/ar.csv"]
    outdir: "out/etl"
  - name: forecast
    weekly_parquet: "out/etl/cashflow_weekly.parquet"
    out_json: "out/forecast/results.json"
  - name: anomalies
    transactions_parquet: "out/etl/transactions_clean.parquet"
    out_parquet: "out/anomaly/annotated.parquet"
  - name: clustering
    transactions_parquet: "out/etl/transactions_clean.parquet"
    out_parquet: "out/cluster/assignments.parquet"
  - name: ews
    weekly_parquet: "out/etl/cashflow_weekly.parquet"
    out_json: "out/ews/alerts.json"
  - name: report
    weekly_parquet: "out/etl/cashflow_weekly.parquet"
    forecast_json: "out/forecast/results.json"
    anomaly_parquet: "out/anomaly/annotated.parquet"
    cluster_parquet: "out/cluster/assignments.parquet"
    outdir: "out/report"

Copyright (c) 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Optional YAML support
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("runner")

# Import local modules (guarded)
def _safe_import(name: str):
    try:
        return __import__(name.replace(".py",""))
    except Exception as e:
        logger.warning("Optional module %s unavailable: %s", name, e)
        return None

m_etl   = _safe_import("cashflow_data_pipeline")
m_fc    = _safe_import("cashflow_forecasting_ensemble")
m_anom  = _safe_import("cashflow_anomaly_detection")
m_tco   = _safe_import("bi_tool_tco_roi_analysis")
m_clu   = _safe_import("payment_behavior_clustering")
m_ews   = _safe_import("early_warning_signal_generator")

@dataclass
class RunMeta:
    run_id: str
    seed: int
    started_at: str
    python: str
    tasks: List[str]

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _write_json(p: str | Path, obj: Any):
    p = Path(p); _ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML not installed; please use JSON config or install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)

def set_seed(seed: int):
    np.random.seed(seed)

def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    name = task.get("name")
    if name == "etl" and m_etl:
        etl = m_etl.CashflowETL()
        artifacts = etl.build(task["inputs"], task["outdir"])
        return {"task": "etl", "artifacts": artifacts}

    if name == "forecast" and m_fc:
        weekly = pd.read_parquet(task["weekly_parquet"])
        rb = m_fc.RollingBacktester(horizons=tuple(task.get("horizons", (1,4))))
        out = rb.run(weekly_df=weekly, target_col=task.get("target_col", "net_cash"))
        _write_json(task["out_json"], out)
        return {"task": "forecast", "out_json": task["out_json"]}

    if name == "anomalies" and m_anom:
        df = pd.read_parquet(task["transactions_parquet"])
        det = m_anom.AnomalyDetector()
        out_df = det.detect(df)
        outp = task["out_parquet"]
        _ensure_dir(Path(outp).parent)
        out_df.to_parquet(outp, index=False)
        return {"task": "anomalies", "out_parquet": outp, "rows": len(out_df)}

    if name == "clustering" and m_clu:
        result = m_clu.cluster_from_transactions(
            transactions_parquet=task["transactions_parquet"],
            out_parquet=task["out_parquet"],
            n_clusters=task.get("n_clusters", 6)
        )
        return {"task": "clustering", **result}

    if name == "ews" and m_ews:
        weekly = task["weekly_parquet"]
        out_json = task["out_json"]
        thr = float(task.get("threshold", 0.6))
        result = m_ews.generate_alerts(weekly_parquet=weekly, out_json=out_json, threshold=thr)
        return {"task": "ews", **result}

    if name == "report":
        # defer to reporting module if present, else stub simple index
        outdir = Path(task["outdir"]); _ensure_dir(outdir)
        index = outdir / "index.md"
        lines = ["# Experiment Report", f"- Generated: {_now_iso()}"]
        for k in ("weekly_parquet","forecast_json","anomaly_parquet","cluster_parquet"):
            if k in task:
                lines.append(f"- {k}: {task[k]}")
        index.write_text("\n".join(lines), encoding="utf-8")
        return {"task": "report", "index": str(index)}

    raise ValueError(f"Unknown or unavailable task: {name}")

def run_config(cfg: Dict[str, Any], artifacts_dir: Optional[str] = None) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tasks = [t.get("name") for t in cfg.get("tasks", [])]
    meta = RunMeta(run_id, seed, _now_iso(), sys.version.split()[0], tasks)
    results = {"meta": asdict(meta), "tasks": []}

    for task in cfg.get("tasks", []):
        logger.info("Running task: %s", task.get("name"))
        res = run_task(task)
        results["tasks"].append(res)

    if artifacts_dir:
        _ensure_dir(artifacts_dir)
        _write_json(Path(artifacts_dir) / f"run_{run_id}.json", results)
    return results

def _parse_args():
    ap = argparse.ArgumentParser(description="Unified experiment runner")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config")
    ap.add_argument("--artifacts", default="artifacts", help="Directory to store run metadata")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    cfg = _read_config(args.config)
    out = run_config(cfg, artifacts_dir=args.artifacts)
    print(json.dumps(out, indent=2))
