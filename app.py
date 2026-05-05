"""
ATM-Net Optimizer — Streamlit Dashboard (app.py)
=================================================
Professional multi-page dashboard wiring all six modules together.

Run from the project root:
    streamlit run app.py

Sidebar navigation:
  1. Generate Data       → data_generator.py
  2. Build Network       → network_builder.py
  3. Compute Metrics     → network_metrics.py
  4. Detect Communities  → community_detection.py
  5. Run Optimization    → optimizer.py
  6. Visualizations      → visualizer.py (all 4 plots)
"""

import os
import sys
import io
import pickle

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Ensure src/ is on the path regardless of working directory ────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
sys.path.insert(0, SRC_DIR)


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATM-Net Optimizer",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1D3557 0%, #457B9D 100%);
    }
    section[data-testid="stSidebar"] * { color: #F1FAEE !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #F0F4F8;
        border: 1px solid #D1DCE8;
        border-radius: 10px;
        padding: 12px 18px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1D3557, #457B9D);
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        margin-bottom: 18px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Status badges */
    .badge-overloaded { background:#E63946; color:white; padding:3px 10px;
                        border-radius:12px; font-size:12px; font-weight:600; }
    .badge-underused  { background:#A8DADC; color:#1D3557; padding:3px 10px;
                        border-radius:12px; font-size:12px; font-weight:600; }
    .badge-normal     { background:#6C757D; color:white; padding:3px 10px;
                        border-radius:12px; font-size:12px; font-weight:600; }

    /* Table alternating rows */
    .dataframe tbody tr:nth-child(even) { background-color: #F7F9FB; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def file_exists(filename: str) -> bool:
    return os.path.exists(data_path(filename))


def load_graph_cached():
    path = data_path("atm_network.gpickle")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def status_badge(status: str) -> str:
    cls = {"Overloaded": "badge-overloaded",
           "Underused":  "badge-underused",
           "Normal":     "badge-normal"}.get(status, "badge-normal")
    return f'<span class="{cls}">{status}</span>'


def render_matplotlib_fig(fig) -> None:
    """Render a Matplotlib figure in Streamlit without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">🔷 {text}</div>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏧 ATM-Net Optimizer")
    st.markdown("**Graph-Based ATM Placement & Usage Analysis**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Home",
            "1️⃣  Generate Data",
            "2️⃣  Build Network",
            "3️⃣  Compute Metrics",
            "4️⃣  Detect Communities",
            "5️⃣  Run Optimization",
            "6️⃣  Visualizations",
        ],
    )

    st.markdown("---")
    st.markdown("**Pipeline Status**")

    checks = [
        ("transactions.csv",   "Data Generated"),
        ("atm_network.gpickle","Network Built"),
        ("atm_metrics.csv",    "Metrics Computed"),
        ("atm_communities.csv","Communities Detected"),
        ("atm_classified.csv", "Optimization Done"),
    ]
    for fname, label in checks:
        icon = "✅" if file_exists(fname) else "⬜"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.caption("Social Network Analysis · Banking Analytics")


# ─── Pages ───────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🏧 ATM-Net Optimizer")
    st.subheader("Graph-Based ATM Placement & Usage Optimization System")
    st.markdown("""
    This dashboard applies **Social Network Analysis (SNA)** to a city-wide ATM
    transaction dataset. Navigate through the pipeline steps using the sidebar.

    ---
    ### 📋 What this system does

    | Step | Module | Purpose |
    |------|--------|---------|
    | 1 | Data Generator | Synthetic ATM transactions for 500 customers across 3 city zones |
    | 2 | Network Builder | Builds a weighted co-usage graph (nodes = ATMs, edges = shared customers) |
    | 3 | Network Metrics | Computes Degree, Betweenness, Closeness Centrality and PageRank |
    | 4 | Community Detection | Identifies ATM clusters using Girvan-Newman / Louvain algorithm |
    | 5 | Optimizer | Flags overloaded / underused ATMs and recommends new placement |
    | 6 | Visualizations | Four interactive network graphs for insight presentation |

    ---
    ### 🚀 Quick Start
    Run all steps in order using the **sidebar navigation**.
    Each step shows live output, tables, and charts.
    """)

    if file_exists("atm_metrics.csv"):
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        metrics = pd.read_csv(data_path("atm_metrics.csv"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total ATMs",          len(metrics))
        c2.metric("Avg PageRank",         f"{metrics['pagerank'].mean():.4f}")
        c3.metric("Max Transactions",     int(metrics["transaction_count"].max()))
        c4.metric("Avg Degree Centrality",f"{metrics['degree'].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Generate Data
# ══════════════════════════════════════════════════════════════════════════════
elif page == "1️⃣  Generate Data":
    st.title("1️⃣ Generate Synthetic Data")
    st.markdown("Creates realistic ATM transaction data for a simulated city with 3 zones.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Dataset Specification**
        - 🏙 20 ATMs across 3 zones (Downtown, Suburbs, Shopping District)
        - 👤 500 customers, each using 2–3 preferred ATMs
        - 💳 ~8,500 transactions (Jan – Jun 2024)
        - 💰 Amounts in INR, rounded to ₹500 denominations
        - 🕐 Realistic time distribution (daytime-heavy)
        """)
    with col2:
        st.info("**Output files**\n\n`data/transactions.csv`\n\n`data/atm_locations.csv`")

    st.markdown("---")

    if st.button("▶ Generate Data", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic ATM data..."):
            try:
                import data_generator
                with st.expander("📋 Generation Log", expanded=True):
                    # Capture stdout
                    import contextlib
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        atm_df, tx_df = data_generator.run(data_dir=DATA_DIR)
                    st.code(log_buf.getvalue())

                st.success("✅ Data generated successfully!")

                tab1, tab2 = st.tabs(["💳 Transactions (sample)", "📍 ATM Locations"])
                with tab1:
                    st.dataframe(tx_df.head(20), use_container_width=True)
                    st.caption(f"Total rows: {len(tx_df):,}  |  "
                               f"Unique customers: {tx_df['customer_id'].nunique()}  |  "
                               f"Unique ATMs: {tx_df['atm_id'].nunique()}")
                with tab2:
                    st.dataframe(atm_df, use_container_width=True)

                # Summary metrics
                st.markdown("---")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Transactions", f"{len(tx_df):,}")
                c2.metric("Unique Customers",   tx_df["customer_id"].nunique())
                c3.metric("ATMs",               len(atm_df))
                c4.metric("Avg Amount (INR)",   f"₹{tx_df['amount'].mean():,.0f}")

            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif file_exists("transactions.csv"):
        st.info("✅ Data already exists. Click the button above to regenerate.")
        tx_df  = pd.read_csv(data_path("transactions.csv"))
        atm_df = pd.read_csv(data_path("atm_locations.csv"))
        tab1, tab2 = st.tabs(["💳 Transactions (sample)", "📍 ATM Locations"])
        with tab1:
            st.dataframe(tx_df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(atm_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Build Network
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2️⃣  Build Network":
    st.title("2️⃣ Build ATM Network Graph")
    st.markdown("""
    Constructs a **weighted undirected graph** from transactions.
    - **Nodes** = ATMs
    - **Edges** = Two ATMs connected if at least one customer used both
    - **Edge weight** = Number of unique shared customers
    """)
    st.info("**Output:** `data/atm_network.gpickle`")
    st.markdown("---")

    if not file_exists("transactions.csv"):
        st.warning("⚠️ Please run Step 1 (Generate Data) first.")
    else:
        if st.button("▶ Build Network", type="primary", use_container_width=True):
            with st.spinner("Building ATM co-usage graph..."):
                try:
                    import network_builder
                    import contextlib
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        G = network_builder.run(data_dir=DATA_DIR)
                    with st.expander("📋 Build Log", expanded=True):
                        st.code(log_buf.getvalue())

                    st.success("✅ Network graph built successfully!")

                    st.markdown("---")
                    c1,c2,c3,c4 = st.columns(4)
                    import networkx as nx
                    c1.metric("Nodes (ATMs)", G.number_of_nodes())
                    c2.metric("Edges",        G.number_of_edges())
                    weights = [d["weight"] for _,_,d in G.edges(data=True)]
                    c3.metric("Avg Edge Weight", f"{sum(weights)/len(weights):.2f}")
                    c4.metric("Graph Density",   f"{nx.density(G):.4f}")

                    # Top edges table
                    st.markdown("### 🔗 Strongest Connections (Top 10 edges by weight)")
                    edge_rows = sorted(
                        [(u, v, d["weight"]) for u,v,d in G.edges(data=True)],
                        key=lambda x: x[2], reverse=True
                    )[:10]
                    st.dataframe(
                        pd.DataFrame(edge_rows, columns=["ATM A","ATM B","Shared Customers"]),
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        elif file_exists("atm_network.gpickle"):
            st.info("✅ Network already built. Click above to rebuild.")
            G = load_graph_cached()
            if G:
                import networkx as nx
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Nodes (ATMs)", G.number_of_nodes())
                c2.metric("Edges",        G.number_of_edges())
                weights = [d["weight"] for _,_,d in G.edges(data=True)]
                c3.metric("Avg Edge Weight", f"{sum(weights)/len(weights):.2f}")
                c4.metric("Graph Density",   f"{nx.density(G):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Compute Metrics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "3️⃣  Compute Metrics":
    st.title("3️⃣ Network Metrics")
    st.markdown("""
    Computes four centrality measures for every ATM node plus raw transaction volume.

    | Metric | Meaning |
    |--------|---------|
    | **Degree Centrality** | Fraction of other ATMs this node is connected to |
    | **Betweenness Centrality** | How often this ATM sits on shortest paths (bridge role) |
    | **Closeness Centrality** | How accessible this ATM is to the whole network |
    | **PageRank** | Overall importance, accounting for quality of connections |
    """)
    st.info("**Output:** `data/atm_metrics.csv`")
    st.markdown("---")

    if not file_exists("atm_network.gpickle"):
        st.warning("⚠️ Please run Step 2 (Build Network) first.")
    else:
        if st.button("▶ Compute Metrics", type="primary", use_container_width=True):
            with st.spinner("Computing centrality metrics (this may take a moment)..."):
                try:
                    import network_metrics
                    import contextlib
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        metrics_df = network_metrics.run(data_dir=DATA_DIR)
                    with st.expander("📋 Computation Log", expanded=False):
                        st.code(log_buf.getvalue())
                    st.success("✅ Metrics computed successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if file_exists("atm_metrics.csv"):
            metrics_df = pd.read_csv(data_path("atm_metrics.csv"))

            st.markdown("---")
            # Summary stats
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Highest PageRank ATM",
                      metrics_df.loc[metrics_df["pagerank"].idxmax(), "atm_id"])
            c2.metric("Most Transactions",
                      metrics_df.loc[metrics_df["transaction_count"].idxmax(), "atm_id"])
            c3.metric("Best Closeness ATM",
                      metrics_df.loc[metrics_df["closeness"].idxmax(), "atm_id"])
            c4.metric("Highest Betweenness ATM",
                      metrics_df.loc[metrics_df["betweenness"].idxmax(), "atm_id"])

            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(
                ["📊 Full Metrics Table", "🥇 Top 5 by PageRank", "📈 Distribution Charts"]
            )

            with tab1:
                display_cols = ["atm_id","zone","degree","betweenness",
                                "closeness","pagerank","transaction_count","avg_amount"]
                st.dataframe(
                    metrics_df[display_cols].style.background_gradient(
                        subset=["pagerank","transaction_count"], cmap="Blues"
                    ),
                    use_container_width=True,
                )

            with tab2:
                top5 = metrics_df.head(5)[["atm_id","zone","pagerank",
                                           "degree","betweenness","closeness",
                                           "transaction_count"]]
                st.dataframe(top5, use_container_width=True, hide_index=True)
                for _, row in top5.iterrows():
                    with st.expander(f"🏧 {row['atm_id']} — {row['zone']}"):
                        cc1,cc2,cc3,cc4 = st.columns(4)
                        cc1.metric("PageRank",     f"{row['pagerank']:.6f}")
                        cc2.metric("Degree",       f"{row['degree']:.4f}")
                        cc3.metric("Closeness",    f"{row['closeness']:.4f}")
                        cc4.metric("Transactions", int(row["transaction_count"]))

            with tab3:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.patch.set_facecolor("#F8F9FA")
                plot_configs = [
                    ("pagerank",          "PageRank",          "#4C9BE8", axes[0,0]),
                    ("degree",            "Degree Centrality", "#56C47B", axes[0,1]),
                    ("betweenness",       "Betweenness",       "#F4A261", axes[1,0]),
                    ("transaction_count", "Transaction Count", "#E63946", axes[1,1]),
                ]
                for col, label, color, ax in plot_configs:
                    vals = metrics_df.sort_values(col, ascending=False)
                    bars = ax.bar(vals["atm_id"], vals[col], color=color, alpha=0.85,
                                  edgecolor="white", linewidth=0.5)
                    ax.set_title(label, fontsize=11, fontweight="bold", color="#1D3557")
                    ax.set_facecolor("#F0F2F5")
                    ax.tick_params(axis="x", rotation=45, labelsize=6)
                    ax.tick_params(axis="y", labelsize=7)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                plt.suptitle("ATM Metric Distributions", fontsize=13,
                             fontweight="bold", color="#1D3557", y=1.01)
                plt.tight_layout()
                render_matplotlib_fig(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Community Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "4️⃣  Detect Communities":
    st.title("4️⃣ Community Detection")
    st.markdown("""
    Detects **ATM clusters** representing distinct city zones using graph partitioning.

    - **Louvain algorithm** (if `python-louvain` is installed) — fast, high quality
    - **Girvan-Newman** fallback — edge-betweenness iterative removal

    Communities reveal which ATMs serve the same customer population.
    """)
    st.info("**Output:** `data/atm_communities.csv`")
    st.markdown("---")

    if not file_exists("atm_network.gpickle"):
        st.warning("⚠️ Please run Step 2 (Build Network) first.")
    else:
        if st.button("▶ Detect Communities", type="primary", use_container_width=True):
            with st.spinner("Running community detection algorithm..."):
                try:
                    import community_detection
                    import contextlib
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        comm_df = community_detection.run(data_dir=DATA_DIR)
                    with st.expander("📋 Detection Log", expanded=False):
                        st.code(log_buf.getvalue())
                    st.success("✅ Communities detected successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if file_exists("atm_communities.csv"):
            comm_df = pd.read_csv(data_path("atm_communities.csv"))
            n_comms = comm_df["community_id"].nunique()

            st.markdown("---")
            st.metric("Communities Found", n_comms)
            st.markdown("---")

            tab1, tab2 = st.tabs(["📋 Community Table", "📊 Community Summary"])
            with tab1:
                st.dataframe(comm_df, use_container_width=True, hide_index=True)

            with tab2:
                for comm_id in sorted(comm_df["community_id"].unique()):
                    subset = comm_df[comm_df["community_id"] == comm_id]
                    label  = subset["community_label"].iloc[0]
                    with st.expander(f"🔵 {label}  ({len(subset)} ATMs)", expanded=True):
                        c1,c2,c3 = st.columns(3)
                        c1.metric("ATMs in Cluster", len(subset))
                        if "transaction_count" in subset.columns:
                            c2.metric("Total Transactions", f"{subset['transaction_count'].sum():,}")
                        if "pagerank" in subset.columns:
                            c3.metric("Avg PageRank", f"{subset['pagerank'].mean():.6f}")
                        st.write("**Members:** " + ", ".join(subset["atm_id"].tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Optimization
# ══════════════════════════════════════════════════════════════════════════════
elif page == "5️⃣  Run Optimization":
    st.title("5️⃣ Optimization Engine")
    st.markdown("""
    Applies **percentile-based rules** to classify every ATM and recommends
    where to place a new ATM for maximum network coverage improvement.

    | Classification | Rule |
    |----------------|------|
    | 🔴 **Overloaded** | PageRank > P75 **AND** Degree > P75 **AND** Tx > P75 |
    | 🔵 **Underused**  | PageRank < P25 **AND** Tx < P25 |
    | ⚫ **Normal**     | Everything else |
    | ⭐ **New ATM**    | Midpoint of 3 lowest-closeness ATMs |
    """)
    st.info("**Output:** `data/recommendations.txt`  +  `data/atm_classified.csv`")
    st.markdown("---")

    if not file_exists("atm_metrics.csv"):
        st.warning("⚠️ Please run Step 3 (Compute Metrics) first.")
    else:
        if st.button("▶ Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Classifying ATMs and generating recommendations..."):
                try:
                    import optimizer
                    import contextlib
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        classified_df, new_atm = optimizer.run(data_dir=DATA_DIR)
                    with st.expander("📋 Optimizer Log", expanded=False):
                        st.code(log_buf.getvalue())
                    st.success("✅ Optimization complete!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if file_exists("atm_classified.csv"):
            classified_df = pd.read_csv(data_path("atm_classified.csv"))
            worst = classified_df.nsmallest(3, "closeness")
            new_atm = {
                "latitude" : round(float(worst["latitude"].mean()), 6),
                "longitude": round(float(worst["longitude"].mean()), 6),
                "source_atms": worst["atm_id"].tolist(),
            }

            overloaded = classified_df[classified_df["status"] == "Overloaded"]
            underused  = classified_df[classified_df["status"] == "Underused"]
            normal     = classified_df[classified_df["status"] == "Normal"]

            # KPI row
            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            c1.metric("🔴 Overloaded ATMs", len(overloaded))
            c2.metric("🔵 Underused ATMs",  len(underused))
            c3.metric("⚫ Normal ATMs",     len(normal))

            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔴 Overloaded", "🔵 Underused", "⭐ New ATM", "📋 Full Classification"
            ])

            with tab1:
                if overloaded.empty:
                    st.success("No overloaded ATMs found.")
                else:
                    st.error(f"**{len(overloaded)} overloaded ATM(s) detected — action required!**")
                    for _, row in overloaded.iterrows():
                        with st.container():
                            cc1,cc2,cc3,cc4 = st.columns(4)
                            cc1.markdown(f"### {row['atm_id']}")
                            cc2.metric("Zone", row["zone"])
                            cc3.metric("PageRank", f"{row['pagerank']:.5f}")
                            cc4.metric("Transactions", int(row["transaction_count"]))
                    st.markdown("""
                    **Recommended Actions:**
                    - 💰 Increase cash replenishment frequency
                    - 🏧 Install an additional ATM nearby
                    - 📣 Redirect customers to nearby underused ATMs
                    """)

            with tab2:
                if underused.empty:
                    st.success("No underused ATMs found.")
                else:
                    st.warning(f"**{len(underused)} underused ATM(s) detected.**")
                    for _, row in underused.iterrows():
                        with st.container():
                            cc1,cc2,cc3,cc4 = st.columns(4)
                            cc1.markdown(f"### {row['atm_id']}")
                            cc2.metric("Zone", row["zone"])
                            cc3.metric("PageRank", f"{row['pagerank']:.5f}")
                            cc4.metric("Transactions", int(row["transaction_count"]))
                    st.markdown("""
                    **Recommended Actions:**
                    - 🔄 Relocate to a higher-demand area
                    - 🗑 Consider decommissioning
                    - 📋 Review placement vs customer density
                    """)

            with tab3:
                st.markdown("### ⭐ New ATM Placement Recommendation")
                st.info(f"""
                **Suggested coordinates:**
                - Latitude:  `{new_atm['latitude']}`
                - Longitude: `{new_atm['longitude']}`

                **Based on lowest-closeness ATMs:** {', '.join(new_atm['source_atms'])}

                **Rationale:** These ATMs have the poorest accessibility in the network.
                Placing a new ATM at their geographic midpoint improves overall network
                closeness and reduces underserved zones.
                """)

                # Show recommendations.txt if available
                rec_path = data_path("recommendations.txt")
                if os.path.exists(rec_path):
                    with open(rec_path) as f:
                        st.text(f.read())

            with tab4:
                # Color-coded full table
                def highlight_status(row):
                    colors = {"Overloaded": "#FFD5D5", "Underused": "#D5F0F3", "Normal": ""}
                    return [f"background-color: {colors.get(row['status'], '')}" for _ in row]

                display = classified_df[["atm_id","zone","status","pagerank",
                                         "degree","closeness","transaction_count"]]
                st.dataframe(
                    display.style.apply(highlight_status, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "6️⃣  Visualizations":
    st.title("6️⃣ Network Visualizations")
    st.markdown("""
    Four graph visualizations generated from the complete ATM network analysis.
    Each plot uses real geographic coordinates (lat/lon) as node positions.
    """)
    st.markdown("---")

    if not file_exists("atm_network.gpickle"):
        st.warning("⚠️ Please complete Steps 2–5 before viewing visualizations.")
    else:
        # Load all data needed
        G = load_graph_cached()

        viz_choice = st.selectbox(
            "Select visualization",
            options=[
                "1 — ATM Co-usage Network (by zone)",
                "2 — Top 5 ATMs by PageRank",
                "3 — Community Clusters",
                "4 — Overloaded / Underused ATMs",
            ],
        )

        if st.button("▶ Generate Plot", type="primary", use_container_width=True):
            with st.spinner("Rendering visualization..."):
                try:
                    import visualizer

                    if "1" in viz_choice:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        plt.close(fig)
                        visualizer.plot_network(G, save_path=None)
                        buf = io.BytesIO()
                        # Re-generate and capture
                        import matplotlib
                        matplotlib.use("Agg")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#F0F2F5")
                        ax.set_title("ATM Network — Co-usage Graph by Zone",
                                     fontsize=15, fontweight="bold", color="#1D3557", pad=18)
                        ax.set_xlabel("Longitude", fontsize=10)
                        ax.set_ylabel("Latitude",  fontsize=10)
                        # Call the actual function with a tempfile
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                            tmp = tf.name
                        visualizer.plot_network(G, save_path=tmp)
                        st.image(tmp, use_column_width=True)
                        os.unlink(tmp)

                    elif "2" in viz_choice:
                        if not file_exists("atm_metrics.csv"):
                            st.warning("Run Step 3 first.")
                        else:
                            metrics_df = pd.read_csv(data_path("atm_metrics.csv"))
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                                tmp = tf.name
                            visualizer.plot_top_atms(G, metrics_df, save_path=tmp)
                            st.image(tmp, use_column_width=True)
                            os.unlink(tmp)

                    elif "3" in viz_choice:
                        if not file_exists("atm_communities.csv"):
                            st.warning("Run Step 4 first.")
                        else:
                            comm_df = pd.read_csv(data_path("atm_communities.csv"))
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                                tmp = tf.name
                            visualizer.plot_communities(G, comm_df, save_path=tmp)
                            st.image(tmp, use_column_width=True)
                            os.unlink(tmp)

                    elif "4" in viz_choice:
                        if not file_exists("atm_classified.csv"):
                            st.warning("Run Step 5 first.")
                        else:
                            classified_df = pd.read_csv(data_path("atm_classified.csv"))
                            worst = classified_df.nsmallest(3, "closeness")
                            new_atm = {
                                "latitude" : round(float(worst["latitude"].mean()), 6),
                                "longitude": round(float(worst["longitude"].mean()), 6),
                                "source_atms": worst["atm_id"].tolist(),
                            }
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                                tmp = tf.name
                            visualizer.plot_overload_status(G, classified_df, new_atm, save_path=tmp)
                            st.image(tmp, use_column_width=True)
                            os.unlink(tmp)

                except Exception as e:
                    st.error(f"❌ Error rendering plot: {e}")
                    st.exception(e)

        st.markdown("---")
        st.markdown("### 📥 Download All Plots")
        if st.button("Generate & Save All 4 Plots to data/plots/"):
            with st.spinner("Generating all plots..."):
                try:
                    import visualizer
                    import contextlib
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        visualizer.generate_all_plots(
                            data_dir=DATA_DIR,
                            output_dir=os.path.join(DATA_DIR, "plots"),
                        )
                    st.success("✅ All plots saved to data/plots/")
                    st.code(log_buf.getvalue())

                    # Show all 4
                    plot_dir = os.path.join(DATA_DIR, "plots")
                    cols = st.columns(2)
                    for i, fname in enumerate(
                        ["01_network.png","02_top_atms.png","03_communities.png","04_overload.png"]
                    ):
                        fpath = os.path.join(plot_dir, fname)
                        if os.path.exists(fpath):
                            with cols[i % 2]:
                                st.image(fpath, use_column_width=True)
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.exception(e)
