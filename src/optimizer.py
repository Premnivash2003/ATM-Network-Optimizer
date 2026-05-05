"""
ATM-Net Optimizer — Module 5: Optimization Engine
==================================================
Rule-based ATM recommendations using percentile thresholds.

Classification rules:
  OVERLOADED  → PageRank > P75  AND  Degree > P75  AND  tx_count > P75
  UNDERUSED   → PageRank < P25  AND  tx_count < P25
  NEW ATM     → Midpoint of the 3 ATMs with lowest closeness centrality
                (worst-served areas; a new ATM there would improve coverage)

Output: data/recommendations.txt  +  data/atm_classified.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from network_metrics import load_metrics
from network_builder  import load_graph


# ─── Classification ──────────────────────────────────────────────────────────

def classify_atms(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply percentile-based rules to classify every ATM.

    Returns a copy of metrics_df with three new columns:
      is_overloaded  (bool)
      is_underused   (bool)
      status         (str) : 'Overloaded' | 'Underused' | 'Normal'
    """
    df = metrics_df.copy()

    # ── Thresholds ─────────────────────────────────────────────────────────
    p75_pagerank = np.percentile(df["pagerank"],          75)
    p75_degree   = np.percentile(df["degree"],            75)
    p75_tx       = np.percentile(df["transaction_count"], 75)
    p25_pagerank = np.percentile(df["pagerank"],          25)
    p25_tx       = np.percentile(df["transaction_count"], 25)

    # ── Classification masks ───────────────────────────────────────────────
    overloaded_mask = (
        (df["pagerank"]          > p75_pagerank) &
        (df["degree"]            > p75_degree)   &
        (df["transaction_count"] > p75_tx)
    )
    underused_mask = (
        (df["pagerank"]          < p25_pagerank) &
        (df["transaction_count"] < p25_tx)
    )

    df["is_overloaded"] = overloaded_mask
    df["is_underused"]  = underused_mask

    # Edge case: can't be both (overloaded wins if triggered simultaneously)
    df.loc[df["is_overloaded"], "is_underused"] = False

    df["status"] = "Normal"
    df.loc[df["is_overloaded"], "status"] = "Overloaded"
    df.loc[df["is_underused"],  "status"] = "Underused"

    # Store thresholds as metadata columns (useful for dashboard display)
    df["_p75_pagerank"] = round(p75_pagerank, 6)
    df["_p75_degree"]   = round(p75_degree,   6)
    df["_p75_tx"]       = int(p75_tx)
    df["_p25_pagerank"] = round(p25_pagerank, 6)
    df["_p25_tx"]       = int(p25_tx)

    return df


# ─── New ATM Recommendation ──────────────────────────────────────────────────

def recommend_new_atm(
    classified_df: pd.DataFrame,
    n_worst: int = 3,
) -> dict:
    """
    Find the n_worst ATMs by closeness centrality (= least accessible nodes).
    Recommend placing a new ATM at their geographic centroid.

    Returns a dict with keys: latitude, longitude, rationale, source_atms
    """
    worst = classified_df.nsmallest(n_worst, "closeness")

    new_lat = worst["latitude"].mean()
    new_lon = worst["longitude"].mean()
    source_ids = worst["atm_id"].tolist()

    rationale = (
        f"These {n_worst} ATMs have the lowest closeness centrality "
        f"({', '.join(source_ids)}), meaning they are the most isolated "
        f"in the network. Placing a new ATM at their geographic midpoint "
        f"({new_lat:.6f}, {new_lon:.6f}) would improve network connectivity "
        f"and reduce travel distance for underserved customers."
    )

    return {
        "latitude"   : round(new_lat, 6),
        "longitude"  : round(new_lon, 6),
        "source_atms": source_ids,
        "rationale"  : rationale,
    }


# ─── Report Writer ───────────────────────────────────────────────────────────

def build_recommendations_report(
    classified_df: pd.DataFrame,
    new_atm: dict,
) -> str:
    """
    Build a structured plain-text report string.
    """
    overloaded = classified_df[classified_df["status"] == "Overloaded"]
    underused  = classified_df[classified_df["status"] == "Underused"]

    lines = []
    lines.append("=" * 65)
    lines.append("  ATM-NET OPTIMIZER — RECOMMENDATION REPORT")
    lines.append("=" * 65)
    lines.append(f"  Total ATMs analysed  : {len(classified_df)}")
    lines.append(f"  Overloaded ATMs      : {len(overloaded)}")
    lines.append(f"  Underused ATMs       : {len(underused)}")
    lines.append(f"  Normal ATMs          : {len(classified_df) - len(overloaded) - len(underused)}")
    lines.append("")

    # ── Thresholds used ────────────────────────────────────────────────────
    r = classified_df.iloc[0]
    lines.append("  CLASSIFICATION THRESHOLDS")
    lines.append("  " + "-" * 63)
    lines.append(f"  Overloaded: PageRank > {r['_p75_pagerank']}  |  "
                 f"Degree > {r['_p75_degree']}  |  Tx > {r['_p75_tx']}")
    lines.append(f"  Underused : PageRank < {r['_p25_pagerank']}  |  "
                 f"Tx < {r['_p25_tx']}")
    lines.append("")

    # ── Overloaded ─────────────────────────────────────────────────────────
    lines.append("  OVERLOADED ATMs (ACTION: Add nearby ATM or increase cash capacity)")
    lines.append("  " + "-" * 63)
    if overloaded.empty:
        lines.append("  None identified.")
    else:
        for _, row in overloaded.iterrows():
            lines.append(
                f"  {row['atm_id']:<10}  Zone: {row['zone']:<22}  "
                f"PageRank: {row['pagerank']:.6f}  "
                f"Tx: {row['transaction_count']:>4}"
            )
    lines.append("")

    # ── Underused ──────────────────────────────────────────────────────────
    lines.append("  UNDERUSED ATMs (ACTION: Consider removal or relocation)")
    lines.append("  " + "-" * 63)
    if underused.empty:
        lines.append("  None identified.")
    else:
        for _, row in underused.iterrows():
            lines.append(
                f"  {row['atm_id']:<10}  Zone: {row['zone']:<22}  "
                f"PageRank: {row['pagerank']:.6f}  "
                f"Tx: {row['transaction_count']:>4}"
            )
    lines.append("")

    # ── New ATM Placement ──────────────────────────────────────────────────
    lines.append("  NEW ATM PLACEMENT RECOMMENDATION")
    lines.append("  " + "-" * 63)
    lines.append(f"  Suggested location   : ({new_atm['latitude']}, {new_atm['longitude']})")
    lines.append(f"  Based on ATMs        : {', '.join(new_atm['source_atms'])}")
    lines.append(f"  Rationale:")
    # Word-wrap the rationale at 65 chars
    words = new_atm["rationale"].split()
    line, wrapped = "    ", []
    for w in words:
        if len(line) + len(w) + 1 > 65:
            wrapped.append(line.rstrip())
            line = "    "
        line += w + " "
    wrapped.append(line.rstrip())
    lines.extend(wrapped)
    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)


def print_optimization_summary(classified_df: pd.DataFrame, new_atm: dict) -> None:
    """Console summary."""
    overloaded = classified_df[classified_df["status"] == "Overloaded"]
    underused  = classified_df[classified_df["status"] == "Underused"]

    print("\n" + "=" * 55)
    print("  OPTIMIZATION RESULTS")
    print("=" * 55)
    print(f"  Overloaded ATMs : {len(overloaded)}")
    if not overloaded.empty:
        for aid in overloaded["atm_id"].tolist():
            print(f"    → {aid}")

    print(f"\n  Underused ATMs  : {len(underused)}")
    if not underused.empty:
        for aid in underused["atm_id"].tolist():
            print(f"    → {aid}")

    print(f"\n  New ATM suggestion:")
    print(f"    Lat: {new_atm['latitude']}  Lon: {new_atm['longitude']}")
    print(f"    Based on: {', '.join(new_atm['source_atms'])}")
    print("=" * 55)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(data_dir: str = "data") -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline: load metrics → classify → recommend → save.
    Returns (classified_df, new_atm_dict).
    """
    print("\n[Module 5] Running optimization engine...")

    metrics_df    = load_metrics(data_dir)
    classified_df = classify_atms(metrics_df)

    new_atm = recommend_new_atm(classified_df)

    # Save classified ATMs
    classified_path = os.path.join(data_dir, "atm_classified.csv")
    save_cols = [c for c in classified_df.columns if not c.startswith("_")]
    classified_df[save_cols].to_csv(classified_path, index=False)
    print(f"  Classified ATMs saved → {classified_path}")

    # Save recommendations report
    report = build_recommendations_report(classified_df, new_atm)
    report_path = os.path.join(data_dir, "recommendations.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Recommendations saved → {report_path}")

    print_optimization_summary(classified_df, new_atm)
    return classified_df, new_atm


def load_classified(data_dir: str = "data") -> pd.DataFrame:
    """Convenience loader used by downstream modules."""
    path = os.path.join(data_dir, "atm_classified.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Classified ATMs not found at {path}\n"
            "Run Module 5 first:  python src/optimizer.py"
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    run(data_dir="data")
