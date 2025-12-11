"""
bi_tool_tco_roi_analysis.py
---------------------------
Three-year Total Cost of Ownership (TCO), ROI, and payback calculator
for Excel vs. Power BI vs. Tableau scenarios, with sensitivity and
scale scenarios (weekly cash-flow volume) matching the EI paper's design.

Produces: per-tool TCO table, benefits table, ROI %, payback months,
and optional Monte Carlo risk-adjusted analysis.

Dependencies: pandas, numpy

Copyright (c) 2025
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

@dataclass
class CostInputs:
    # Initial (year 1) costs
    licensing_y1: float
    impl_consulting: float
    it_setup: float
    dashboard_dev: float
    training: float
    # Ongoing (years 2-3) / per-3yr totals for stacked items
    licensing_y2y3: float
    weekly_ops_3y: float
    maintenance_3y: float

@dataclass
class BenefitInputs:
    cash_buffer_reduction: float
    time_savings_value: float
    discount_capture: float
    decision_quality: float

@dataclass
class Scenario:
    name: str
    excel_costs: CostInputs
    pbi_costs: CostInputs
    tbl_costs: CostInputs
    pbi_benefits: BenefitInputs
    tbl_benefits: BenefitInputs

def tco(c: CostInputs) -> float:
    return float(c.licensing_y1 + c.impl_consulting + c.it_setup + c.dashboard_dev + c.training
                 + c.licensing_y2y3 + c.weekly_ops_3y + c.maintenance_3y)

def roi(total_benefits: float, total_costs: float) -> float:
    return float((total_benefits - total_costs) / max(total_costs, 1e-8) * 100.0)

def payback_months(total_costs: float, annual_benefit: float) -> float:
    monthly = annual_benefit / 12.0
    return float(total_costs / max(monthly, 1e-8))

def scenario_default() -> Scenario:
    # Defaults drawn to mirror the paper's Table 5 & 6 magnitudes for a mid-sized enterprise
    excel = CostInputs(
        licensing_y1=0, impl_consulting=0, it_setup=0, dashboard_dev=0, training=0,
        licensing_y2y3=0, weekly_ops_3y=40950, maintenance_3y=10800
    )
    pbi = CostInputs(
        licensing_y1=360, impl_consulting=6000, it_setup=2400, dashboard_dev=3000, training=4500,
        licensing_y2y3=720, weekly_ops_3y=10470, maintenance_3y=2250
    )
    tbl = CostInputs(
        licensing_y1=2520, impl_consulting=5250, it_setup=2100, dashboard_dev=2850, training=3750,
        licensing_y2y3=5040, weekly_ops_3y=13185, maintenance_3y=2700
    )
    pbi_b = BenefitInputs(65520, 27000, 14352, 21000)
    tbl_b = BenefitInputs(58968, 23400, 12636, 18000)
    return Scenario("MidSized_Default", excel, pbi, tbl, pbi_b, tbl_b)

def scenario_table(sc: Scenario) -> pd.DataFrame:
    rows = []
    for tool, costs in (("Excel", sc.excel_costs), ("Power BI", sc.pbi_costs), ("Tableau", sc.tbl_costs)):
        rows.append({
            "Tool": tool,
            "Total 3Y TCO": tco(costs),
            "Initial Y1": costs.licensing_y1 + costs.impl_consulting + costs.it_setup + costs.dashboard_dev + costs.training,
            "Ongoing Y2-3": costs.licensing_y2y3 + costs.weekly_ops_3y + costs.maintenance_3y
        })
    return pd.DataFrame(rows)

def benefit_table(sc: Scenario) -> pd.DataFrame:
    rows = []
    rows.append({
        "Tool": "Power BI",
        "Total Benefits (3Y)": sc.pbi_benefits.cash_buffer_reduction + sc.pbi_benefits.time_savings_value +
                               sc.pbi_benefits.discount_capture + sc.pbi_benefits.decision_quality
    })
    rows.append({
        "Tool": "Tableau",
        "Total Benefits (3Y)": sc.tbl_benefits.cash_buffer_reduction + sc.tbl_benefits.time_savings_value +
                               sc.tbl_benefits.discount_capture + sc.tbl_benefits.decision_quality
    })
    return pd.DataFrame(rows)

def summarize(sc: Scenario) -> pd.DataFrame:
    tco_df = scenario_table(sc)
    ben_df = benefit_table(sc).set_index("Tool")
    tco_df = tco_df.set_index("Tool")
    out = tco_df.copy()
    for tool in ["Power BI","Tableau"]:
        tb = float(ben_df.loc[tool, "Total Benefits (3Y)"])
        out.loc[tool, "Total Benefits (3Y)"] = tb
        out.loc[tool, "Net Value (3Y)"] = tb - out.loc[tool, "Total 3Y TCO"]
        out.loc[tool, "ROI %"] = roi(tb, out.loc[tool, "Total 3Y TCO"])
        out.loc[tool, "Payback (months)"] = payback_months(out.loc[tool, "Total 3Y TCO"], tb/3.0)
    out.loc["Excel", "Note"] = "Baseline (no BI adoption)"
    return out.reset_index()

def risk_adjusted(tb: float, tcost: float, annual_disc: float = 0.15, years: int = 3, trials: int = 10000, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    # Model uncertainty by perturbing benefits +-20% (normal approx) each year then discount
    annual_base = tb / years
    draws = []
    for _ in range(trials):
        total = 0.0
        for y in range(1, years+1):
            a = annual_base * (1 + rng.normal(0, 0.2))
            total += a / ((1 + annual_disc) ** y)
        draws.append(total)
    draws = np.array(draws)
    mean = float(draws.mean())
    ra_roi = roi(mean, tcost)
    return {"risk_adjusted_BENEFITS": mean, "risk_adjusted_ROI%": ra_roi}

def _parse_args():
    ap = argparse.ArgumentParser(description="3Y TCO/ROI/payback calculator for BI tools")
    ap.add_argument("--risk_adjusted", action="store_true", help="Compute risk-adjusted ROI via Monte Carlo")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    sc = scenario_default()
    summary = summarize(sc)
    print(summary.to_string(index=False))
    if args.risk_adjusted:
        # Example for Power BI only
        tb = (sc.pbi_benefits.cash_buffer_reduction + sc.pbi_benefits.time_savings_value +
              sc.pbi_benefits.discount_capture + sc.pbi_benefits.decision_quality)
        tcost = tco(sc.pbi_costs)
        ra = risk_adjusted(tb, tcost)
        print("\nRisk-adjusted (Power BI):", ra)
