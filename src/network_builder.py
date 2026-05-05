"""
ATM-Net Optimizer — Module 2: Network Builder
==============================================
Builds a weighted undirected NetworkX graph from transaction data.

Graph model:
  Nodes  → ATMs (with attributes: latitude, longitude, zone)
  Edges  → Pair of ATMs that share at least one common customer
  Weight → Number of UNIQUE customers who used BOTH ATMs

Output: data/atm_network.gpickle
"""

import os
import pickle
import itertools

import pandas as pd
import networkx as nx


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions and ATM location CSVs."""
    tx_path  = os.path.join(data_dir, "transactions.csv")
    atm_path = os.path.join(data_dir, "atm_locations.csv")

    for p in (tx_path, atm_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run Module 1 first:  python src/data_generator.py"
            )

    tx_df  = pd.read_csv(tx_path,  parse_dates=["timestamp"])
    atm_df = pd.read_csv(atm_path)
    return tx_df, atm_df


def build_customer_atm_map(tx_df: pd.DataFrame) -> dict[str, set]:
    """
    Return a dict: customer_id → set of ATM IDs they used.
    Using sets removes duplicate visits — we care about co-usage, not frequency.
    """
    return (
        tx_df.groupby("customer_id")["atm_id"]
        .apply(set)
        .to_dict()
    )


def build_edge_weights(customer_atm_map: dict[str, set]) -> dict[tuple, int]:
    """
    For every customer who used ≥2 ATMs, add 1 to the weight of every
    ATM-pair edge.  Uses combinations so (A,B) and (B,A) are the same edge.

    Returns: dict  {(atm_a, atm_b): shared_customer_count}
    """
    edge_weights: dict[tuple, int] = {}

    for customer_id, atms in customer_atm_map.items():
        if len(atms) < 2:
            continue  # single-ATM users don't create edges

        for atm_a, atm_b in itertools.combinations(sorted(atms), 2):
            key = (atm_a, atm_b)
            edge_weights[key] = edge_weights.get(key, 0) + 1

    return edge_weights


def build_graph(
    atm_df: pd.DataFrame,
    edge_weights: dict[tuple, int],
) -> nx.Graph:
    """
    Construct a NetworkX Graph.
      - One node per ATM, with attributes from atm_locations.csv
      - One edge per ATM pair, weighted by shared customer count
    """
    G = nx.Graph()

    # ── Add nodes ──────────────────────────────────────────────────────────
    for _, row in atm_df.iterrows():
        G.add_node(
            row["atm_id"],
            latitude   = float(row["latitude"]),
            longitude  = float(row["longitude"]),
            zone       = str(row["zone"]),
            zone_label = str(row["zone_label"]),
        )

    # ── Add edges ──────────────────────────────────────────────────────────
    for (atm_a, atm_b), weight in edge_weights.items():
        # Only add edge if both nodes exist (guard against stale data)
        if G.has_node(atm_a) and G.has_node(atm_b):
            G.add_edge(atm_a, atm_b, weight=weight)

    return G


def save_graph(G: nx.Graph, data_dir: str = "data") -> str:
    """Persist graph using pickle (gpickle format)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "atm_network.gpickle")

    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def print_graph_summary(G: nx.Graph) -> None:
    """Print key graph statistics to console."""
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    density = nx.density(G)

    print("\n" + "=" * 50)
    print("  GRAPH SUMMARY")
    print("=" * 50)
    print(f"  Nodes (ATMs)       : {G.number_of_nodes()}")
    print(f"  Edges              : {G.number_of_edges()}")
    print(f"  Avg edge weight    : {sum(weights)/len(weights):.2f}" if weights else "  No edges")
    print(f"  Max edge weight    : {max(weights)}" if weights else "")
    print(f"  Min edge weight    : {min(weights)}" if weights else "")
    print(f"  Graph density      : {density:.4f}")
    print(f"  Is connected       : {nx.is_connected(G)}")
    print("=" * 50)

    # Per-zone node count
    zone_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        z = data.get("zone_label", "unknown")
        zone_counts[z] = zone_counts.get(z, 0) + 1

    print("\n  Nodes per zone:")
    for zone, count in sorted(zone_counts.items()):
        print(f"    {zone:<25}: {count}")
    print()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(data_dir: str = "data") -> nx.Graph:
    """
    Full pipeline: load → map → edges → graph → save.
    Returns the NetworkX Graph so other modules can import and call this.
    """
    print("\n[Module 2] Building ATM network graph...")

    tx_df, atm_df = load_data(data_dir)

    print(f"  Loaded {len(tx_df):,} transactions for "
          f"{tx_df['customer_id'].nunique()} customers")

    customer_atm_map = build_customer_atm_map(tx_df)
    multi_atm_users  = sum(1 for s in customer_atm_map.values() if len(s) >= 2)
    print(f"  Customers using ≥2 ATMs (edge contributors): {multi_atm_users}")

    edge_weights = build_edge_weights(customer_atm_map)
    print(f"  Unique ATM-pair edges created: {len(edge_weights)}")

    G = build_graph(atm_df, edge_weights)

    path = save_graph(G, data_dir)
    print(f"  Graph saved → {path}")

    print_graph_summary(G)
    return G


def load_graph(data_dir: str = "data") -> nx.Graph:
    """
    Convenience loader used by downstream modules.
    Raises a clear error if network_builder hasn't been run yet.
    """
    path = os.path.join(data_dir, "atm_network.gpickle")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found at {path}\n"
            "Run Module 2 first:  python src/network_builder.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    run(data_dir="data")
