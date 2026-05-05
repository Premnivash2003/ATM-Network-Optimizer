"""
ATM-Net Optimizer — Module 3: Network Metrics
==============================================
Loads the ATM graph and computes four centrality measures plus
raw transaction volume per ATM.

Metrics computed:
  degree_centrality      → how many other ATMs a node is connected to (0–1)
  betweenness_centrality → how often a node sits on shortest paths (bridge ATMs)
  closeness_centrality   → how close a node is to all others (accessibility)
  pagerank               → overall importance accounting for neighbour quality
  transaction_count      → raw number of transactions at each ATM

Output: data/atm_metrics.csv
"""

import os
import pandas as pd
import networkx as nx

# Local module — allows running standalone or imported
import sys
sys.path.insert(0, os.path.dirname(__file__))
from network_builder import load_graph


# ─── Metric Computation ──────────────────────────────────────────────────────

def compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute all four centrality measures and return as a DataFrame.

    NetworkX returns dicts keyed by node id — we merge them into one table.
    """
    print("  Computing degree centrality...")
    degree_cent = nx.degree_centrality(G)

    print("  Computing betweenness centrality...")
    # weight=None → treats all edges equally for path counting;
    # using weight='weight' would favour high-traffic bridges
    betweenness_cent = nx.betweenness_centrality(G, weight="weight", normalized=True)

    print("  Computing closeness centrality...")
    closeness_cent = nx.closeness_centrality(G, distance="weight")

    print("  Computing PageRank...")
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=1000)

    # Combine into single DataFrame
    df = pd.DataFrame({
        "atm_id"      : list(degree_cent.keys()),
        "degree"      : list(degree_cent.values()),
        "betweenness" : [betweenness_cent[n] for n in degree_cent],
        "closeness"   : [closeness_cent[n]   for n in degree_cent],
        "pagerank"    : [pagerank[n]          for n in degree_cent],
    })

    # Attach zone info from node attributes
    df["zone"] = df["atm_id"].map(
        {n: d.get("zone_label", "unknown") for n, d in G.nodes(data=True)}
    )
    df["latitude"] = df["atm_id"].map(
        {n: d.get("latitude", 0.0) for n, d in G.nodes(data=True)}
    )
    df["longitude"] = df["atm_id"].map(
        {n: d.get("longitude", 0.0) for n, d in G.nodes(data=True)}
    )

    return df


def compute_transaction_counts(tx_path: str) -> pd.Series:
    """
    Count total transactions per ATM from the raw CSV.
    Returns a Series indexed by atm_id.
    """
    tx_df = pd.read_csv(tx_path)
    return tx_df.groupby("atm_id").size().rename("transaction_count")


def compute_avg_transaction_amount(tx_path: str) -> pd.Series:
    """Average withdrawal amount per ATM — useful context for the dashboard."""
    tx_df = pd.read_csv(tx_path)
    return tx_df.groupby("atm_id")["amount"].mean().round(2).rename("avg_amount")


def merge_metrics(
    centrality_df: pd.DataFrame,
    tx_counts: pd.Series,
    avg_amounts: pd.Series,
) -> pd.DataFrame:
    """Join centrality metrics with transaction statistics."""
    df = centrality_df.copy()
    df = df.merge(tx_counts.reset_index(),  on="atm_id", how="left")
    df = df.merge(avg_amounts.reset_index(), on="atm_id", how="left")
    df["transaction_count"] = df["transaction_count"].fillna(0).astype(int)
    df["avg_amount"]        = df["avg_amount"].fillna(0.0)

    # Round centrality scores for readability
    for col in ("degree", "betweenness", "closeness", "pagerank"):
        df[col] = df[col].round(6)

    return df.sort_values("pagerank", ascending=False).reset_index(drop=True)


def print_metrics_summary(df: pd.DataFrame) -> None:
    """Print top-5 ATMs by PageRank plus a per-zone breakdown."""
    print("\n" + "=" * 65)
    print("  TOP 5 ATMs BY PAGERANK")
    print("=" * 65)
    print(f"  {'ATM':<10} {'Zone':<22} {'PageRank':>10} "
          f"{'Degree':>8} {'Tx Count':>10}")
    print("  " + "-" * 63)
    for _, row in df.head(5).iterrows():
        print(f"  {row['atm_id']:<10} {row['zone']:<22} "
              f"{row['pagerank']:>10.6f} {row['degree']:>8.4f} "
              f"{row['transaction_count']:>10}")
    print()

    print("  METRIC RANGES")
    print("  " + "-" * 63)
    for col in ("degree", "betweenness", "closeness", "pagerank", "transaction_count"):
        print(f"  {col:<22}: min={df[col].min():.4f}  "
              f"max={df[col].max():.4f}  mean={df[col].mean():.4f}")
    print("=" * 65)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(data_dir: str = "data") -> pd.DataFrame:
    """
    Full pipeline: load graph → compute metrics → merge tx data → save CSV.
    Returns the metrics DataFrame.
    """
    print("\n[Module 3] Computing network metrics...")

    # Load graph
    G = load_graph(data_dir)
    print(f"  Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Centrality measures
    centrality_df = compute_centrality_metrics(G)

    # Transaction statistics
    tx_path   = os.path.join(data_dir, "transactions.csv")
    tx_counts = compute_transaction_counts(tx_path)
    avg_amts  = compute_avg_transaction_amount(tx_path)

    # Merge everything
    metrics_df = merge_metrics(centrality_df, tx_counts, avg_amts)

    # Save
    out_path = os.path.join(data_dir, "atm_metrics.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"  Metrics saved → {out_path}")

    print_metrics_summary(metrics_df)
    return metrics_df


def load_metrics(data_dir: str = "data") -> pd.DataFrame:
    """Convenience loader used by downstream modules."""
    path = os.path.join(data_dir, "atm_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metrics not found at {path}\n"
            "Run Module 3 first:  python src/network_metrics.py"
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    run(data_dir="data")
