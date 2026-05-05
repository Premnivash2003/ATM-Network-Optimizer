"""
ATM-Net Optimizer — Module 4: Community Detection
==================================================
Detects ATM clusters (city zones) using the Louvain algorithm.

If python-louvain is not installed, falls back to the Girvan-Newman
algorithm (built into NetworkX) so the module always works.

Output: data/atm_communities.csv
"""

import os
import sys
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from network_builder import load_graph

# ─── Algorithm Selection ─────────────────────────────────────────────────────

def _detect_louvain(G: nx.Graph) -> dict[str, int]:
    """
    Run Louvain community detection (requires python-louvain / community pkg).
    Returns dict: node_id → community_id
    """
    import community as community_louvain          # python-louvain package
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    return partition


def _detect_girvan_newman(G: nx.Graph, n_communities: int = 3) -> dict[str, int]:
    """
    Fallback: Girvan-Newman algorithm (edge-betweenness based).
    Iteratively removes high-betweenness edges until n_communities groups form.
    Returns dict: node_id → community_id
    """
    from networkx.algorithms.community import girvan_newman
    import itertools

    comp = girvan_newman(G)
    # Advance the generator until we reach the desired number of communities
    for communities in itertools.islice(comp, n_communities - 1):
        pass

    partition = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            partition[node] = community_id
    return partition


def detect_communities(G: nx.Graph) -> tuple[dict[str, int], str]:
    """
    Auto-select best available algorithm.
    Returns (partition_dict, algorithm_name_used).
    """
    try:
        partition = _detect_louvain(G)
        algo = "Louvain"
    except ImportError:
        print("  python-louvain not found — using Girvan-Newman fallback")
        partition = _detect_girvan_newman(G, n_communities=3)
        algo = "Girvan-Newman"
    return partition, algo


# ─── Post-Processing ─────────────────────────────────────────────────────────

def build_community_dataframe(
    G: nx.Graph,
    partition: dict[str, int],
    metrics_path: str,
) -> pd.DataFrame:
    """
    Build a rich community DataFrame by joining:
      - Community assignments
      - ATM node attributes (zone, lat, lon)
      - Metrics (pagerank, transaction_count) if available
    """
    records = []
    for node, comm_id in partition.items():
        node_data = G.nodes[node]
        records.append({
            "atm_id"      : node,
            "community_id": comm_id,
            "zone"        : node_data.get("zone_label", "unknown"),
            "latitude"    : node_data.get("latitude",   0.0),
            "longitude"   : node_data.get("longitude",  0.0),
        })
    df = pd.DataFrame(records).sort_values(["community_id", "atm_id"]).reset_index(drop=True)

    # Merge metrics if available
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)[["atm_id", "pagerank", "transaction_count"]]
        df = df.merge(metrics, on="atm_id", how="left")

    return df


def compute_community_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-community aggregate statistics."""
    agg: dict = {"atm_id": "count"}
    if "transaction_count" in df.columns:
        agg["transaction_count"] = "sum"
    if "pagerank" in df.columns:
        agg["pagerank"] = "mean"

    stats = (
        df.groupby("community_id")
        .agg(agg)
        .rename(columns={"atm_id": "n_atms"})
        .reset_index()
    )
    # Dominant zone per community
    zone_mode = (
        df.groupby("community_id")["zone"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("dominant_zone")
    )
    stats = stats.merge(zone_mode, on="community_id")
    return stats


def assign_community_labels(
    df: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """Give each community a human-readable label based on its dominant zone."""
    label_map = {
        row["community_id"]: f"Cluster {row['community_id']} – {row['dominant_zone']}"
        for _, row in stats.iterrows()
    }
    df["community_label"] = df["community_id"].map(label_map)
    return df


def print_community_summary(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    algo: str,
) -> None:
    """Print community breakdown to console."""
    n = df["community_id"].nunique()
    print("\n" + "=" * 60)
    print(f"  COMMUNITY DETECTION ({algo})")
    print("=" * 60)
    print(f"  Communities found  : {n}")
    print()

    for _, row in stats.iterrows():
        print(f"  Cluster {row['community_id']} — {row['dominant_zone']}")
        print(f"    ATMs             : {row['n_atms']}")
        if "transaction_count" in row:
            print(f"    Total transactions: {int(row['transaction_count']):,}")
        if "pagerank" in row:
            print(f"    Avg PageRank     : {row['pagerank']:.6f}")
        members = df[df["community_id"] == row["community_id"]]["atm_id"].tolist()
        print(f"    Members          : {', '.join(members)}")
        print()
    print("=" * 60)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(data_dir: str = "data") -> pd.DataFrame:
    """
    Full pipeline: load graph → detect communities → enrich → save CSV.
    Returns the community DataFrame.
    """
    print("\n[Module 4] Detecting ATM communities...")

    G = load_graph(data_dir)
    print(f"  Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition, algo = detect_communities(G)
    print(f"  Algorithm used: {algo}")

    metrics_path = os.path.join(data_dir, "atm_metrics.csv")
    df = build_community_dataframe(G, partition, metrics_path)

    stats = compute_community_stats(df)
    df    = assign_community_labels(df, stats)

    # Add community_id back to the NetworkX graph as a node attribute
    # (used by visualizer later)
    for node, comm_id in partition.items():
        G.nodes[node]["community"] = comm_id

    out_path = os.path.join(data_dir, "atm_communities.csv")
    df.to_csv(out_path, index=False)
    print(f"  Communities saved → {out_path}")

    print_community_summary(df, stats, algo)
    return df


def load_communities(data_dir: str = "data") -> pd.DataFrame:
    """Convenience loader used by downstream modules."""
    path = os.path.join(data_dir, "atm_communities.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Communities not found at {path}\n"
            "Run Module 4 first:  python src/community_detection.py"
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    run(data_dir="data")
