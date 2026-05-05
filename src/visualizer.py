"""
ATM-Net Optimizer — Module 6: Visualizer
Compatibility: NetworkX 3.x
"""
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from network_builder import load_graph

COLORS = {
    "downtown":"#4C9BE8","suburbs":"#56C47B","shopping":"#F4A261",
    "overloaded":"#E63946","underused":"#A8DADC","normal":"#6C757D",
    "highlight":"#FFD60A","new_atm":"#7B2FBE","edge":"#CCCCCC",
}
COMMUNITY_PALETTE = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261","#264653"]

def _get_pos(G):
    return {n:(d["longitude"],d["latitude"]) for n,d in G.nodes(data=True)}

def _edge_widths(G, scale=0.4):
    w=[d.get("weight",1) for _,_,d in G.edges(data=True)]
    mx=max(w) if w else 1
    return [scale+(v/mx)*2.5 for v in w]

def _edge_alphas(G):
    w=[d.get("weight",1) for _,_,d in G.edges(data=True)]
    mx=max(w) if w else 1
    return [0.2+0.6*(v/mx) for v in w]

def _base_figure(title, figsize=(12,8)):
    fig,ax=plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F0F2F5")
    ax.set_title(title,fontsize=15,fontweight="bold",pad=18,color="#1D3557")
    ax.set_xlabel("Longitude",fontsize=10,color="#555555")
    ax.set_ylabel("Latitude",fontsize=10,color="#555555")
    ax.tick_params(colors="#777777",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#CCCCCC")
    return fig,ax

def _draw_edges(ax,G,pos,color="#CCCCCC",scale=0.4,vary_alpha=True):
    widths=_edge_widths(G,scale)
    alphas=_edge_alphas(G) if vary_alpha else [0.35]*G.number_of_edges()
    for i,(u,v) in enumerate(G.edges()):
        ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],
                color=color,linewidth=widths[i],alpha=alphas[i],zorder=1)

def _draw_nodes(ax,G,pos,node_colors,node_sizes=500):
    xs=[pos[n][0] for n in G.nodes()]
    ys=[pos[n][1] for n in G.nodes()]
    sz=node_sizes if isinstance(node_sizes,list) else [node_sizes]*len(xs)
    ax.scatter(xs,ys,c=node_colors,s=sz,edgecolors="#FFFFFF",linewidths=1.5,zorder=3)

def _draw_labels(ax,G,pos,font_color="#1D3557",font_size=7):
    for n in G.nodes():
        x,y=pos[n]
        ax.text(x,y,n,fontsize=font_size,fontweight="bold",
                color=font_color,ha="center",va="center",zorder=4)

def _watermark(ax):
    ax.text(0.99,0.01,"ATM-Net Optimizer",transform=ax.transAxes,
            fontsize=8,color="#BBBBBB",ha="right",va="bottom",style="italic")

def _finish(fig,save_path):
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",exist_ok=True)
        fig.savefig(save_path,dpi=150,bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()

# ── Plot 1: Base network ──────────────────────────────────────────────────────
def plot_network(G, save_path=None):
    pos=_get_pos(G)
    zone_c={n:COLORS.get(d.get("zone","normal"),COLORS["normal"]) for n,d in G.nodes(data=True)}
    node_colors=[zone_c[n] for n in G.nodes()]
    fig,ax=_base_figure("ATM Network — Co-usage Graph by Zone")
    _draw_edges(ax,G,pos,color=COLORS["edge"],vary_alpha=True)
    _draw_nodes(ax,G,pos,node_colors,500)
    _draw_labels(ax,G,pos,"#1D3557")
    for u,v,d in sorted(G.edges(data=True),key=lambda e:e[2].get("weight",0),reverse=True)[:3]:
        mx=(pos[u][0]+pos[v][0])/2; my=(pos[u][1]+pos[v][1])/2
        ax.annotate(f"w={d['weight']}",(mx,my),fontsize=6.5,color="#333333",ha="center",
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",alpha=0.75),zorder=5)
    ax.legend(handles=[
        mpatches.Patch(facecolor=COLORS["downtown"],label="Downtown"),
        mpatches.Patch(facecolor=COLORS["suburbs"],label="Suburbs"),
        mpatches.Patch(facecolor=COLORS["shopping"],label="Shopping District"),
        Line2D([0],[0],color=COLORS["edge"],linewidth=2,label="Co-usage edge"),
    ],loc="upper left",fontsize=9,framealpha=0.9,facecolor="white")
    _watermark(ax); _finish(fig,save_path)

# ── Plot 2: Top ATMs ──────────────────────────────────────────────────────────
def plot_top_atms(G, metrics_df, top_n=5, save_path=None):
    pos=_get_pos(G)
    top_ids=metrics_df.nlargest(top_n,"pagerank")["atm_id"].tolist()
    node_colors=[COLORS["highlight"] if n in top_ids else "#CCCCCC" for n in G.nodes()]
    node_sizes=[900 if n in top_ids else 300 for n in G.nodes()]
    fig,ax=_base_figure(f"Top {top_n} ATMs by PageRank  (Gold = Top Ranked)")
    _draw_edges(ax,G,pos,color="#DDDDDD",scale=0.2,vary_alpha=False)
    _draw_nodes(ax,G,pos,node_colors,node_sizes)
    _draw_labels(ax,G,pos,"#333333",6.5)
    for rank,atm_id in enumerate(top_ids,1):
        if atm_id not in pos: continue
        x,y=pos[atm_id]
        row=metrics_df[metrics_df["atm_id"]==atm_id].iloc[0]
        ax.annotate(f"#{rank}  PR={row['pagerank']:.4f}\nTx={row['transaction_count']}",
                    xy=(x,y),xytext=(x+0.003,y+0.004),fontsize=7.5,
                    color="#1D3557",fontweight="bold",zorder=6,
                    bbox=dict(boxstyle="round,pad=0.4",fc=COLORS["highlight"],alpha=0.9,edgecolor="#AAAAAA"),
                    arrowprops=dict(arrowstyle="->",color="#555555",lw=0.8))
    ax.legend(handles=[
        mpatches.Patch(facecolor=COLORS["highlight"],label=f"Top {top_n} (PageRank)"),
        mpatches.Patch(facecolor="#CCCCCC",label="Other ATMs"),
    ],loc="upper left",fontsize=9,framealpha=0.9,facecolor="white")
    _watermark(ax); _finish(fig,save_path)

# ── Plot 3: Communities ───────────────────────────────────────────────────────
def plot_communities(G, community_df, save_path=None):
    pos=_get_pos(G)
    comm_map=community_df.set_index("atm_id")["community_id"].to_dict()
    n_comms=community_df["community_id"].nunique()
    palette=COMMUNITY_PALETTE[:n_comms]
    node_colors=[palette[comm_map.get(n,0)%len(palette)] for n in G.nodes()]
    fig,ax=_base_figure("ATM Clusters — Community Detection")
    for comm_id,color in enumerate(palette):
        members=community_df[community_df["community_id"]==comm_id]["atm_id"].tolist()
        coords=[pos[m] for m in members if m in pos]
        if len(coords)<2: continue
        xs=[c[0] for c in coords]; ys=[c[1] for c in coords]
        cx,cy=np.mean(xs),np.mean(ys)
        rx=(max(xs)-min(xs))/2+0.005; ry=(max(ys)-min(ys))/2+0.005
        ax.add_patch(mpatches.Ellipse((cx,cy),width=rx*2,height=ry*2,
            color=color,alpha=0.10,zorder=0,linewidth=1.5,linestyle="--",edgecolor=color))
    _draw_edges(ax,G,pos,color="#CCCCCC",scale=0.3,vary_alpha=False)
    _draw_nodes(ax,G,pos,node_colors,500)
    _draw_labels(ax,G,pos,"#111111")
    label_lookup=(community_df.drop_duplicates("community_id")
                  .set_index("community_id")["community_label"].to_dict())
    ax.legend(handles=[
        mpatches.Patch(facecolor=palette[i],alpha=0.85,label=label_lookup.get(i,f"Cluster {i}"))
        for i in range(n_comms)
    ],loc="upper left",fontsize=9,framealpha=0.9,facecolor="white")
    _watermark(ax); _finish(fig,save_path)

# ── Plot 4: Overload status ───────────────────────────────────────────────────
def plot_overload_status(G, classified_df, new_atm=None, save_path=None):
    pos=_get_pos(G)
    status_map=classified_df.set_index("atm_id")["status"].to_dict()
    tx_map=classified_df.set_index("atm_id")["transaction_count"].to_dict()
    sc={"Overloaded":COLORS["overloaded"],"Underused":COLORS["underused"],"Normal":COLORS["normal"]}
    node_colors=[sc.get(status_map.get(n,"Normal"),COLORS["normal"]) for n in G.nodes()]
    node_sizes=[800 if status_map.get(n)=="Overloaded" else 350 if status_map.get(n)=="Underused" else 450
                for n in G.nodes()]
    fig,ax=_base_figure("ATM Load Status — Overloaded / Underused / Normal")
    _draw_edges(ax,G,pos,color="#DDDDDD",scale=0.2,vary_alpha=False)
    _draw_nodes(ax,G,pos,node_colors,node_sizes)
    _draw_labels(ax,G,pos,"#FFFFFF",7)
    for node in G.nodes():
        s=status_map.get(node,"Normal")
        if s not in ("Overloaded","Underused"): continue
        x,y=pos[node]
        ax.annotate(f"{s}\nTx={tx_map.get(node,0)}",xy=(x,y),xytext=(x+0.004,y+0.004),
                    fontsize=7.5,color="#1D3557",zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35",fc="white",alpha=0.87,edgecolor="#AAAAAA"),
                    arrowprops=dict(arrowstyle="->",color="#888888",lw=0.8))
    if new_atm:
        ax.scatter([new_atm["longitude"]],[new_atm["latitude"]],
                   marker="*",s=700,c=COLORS["new_atm"],zorder=7,
                   edgecolors="white",linewidths=0.8)
        ax.annotate("NEW ATM\n(Suggested)",
                    xy=(new_atm["longitude"],new_atm["latitude"]),
                    xytext=(new_atm["longitude"]+0.005,new_atm["latitude"]-0.005),
                    fontsize=8,color=COLORS["new_atm"],fontweight="bold",zorder=7,
                    bbox=dict(boxstyle="round,pad=0.35",fc="white",alpha=0.87),
                    arrowprops=dict(arrowstyle="->",color=COLORS["new_atm"],lw=1.0))
    legend_elements=[
        mpatches.Patch(facecolor=COLORS["overloaded"],label="Overloaded"),
        mpatches.Patch(facecolor=COLORS["underused"],label="Underused"),
        mpatches.Patch(facecolor=COLORS["normal"],label="Normal"),
    ]
    if new_atm:
        legend_elements.append(Line2D([0],[0],marker="*",color="w",
            markerfacecolor=COLORS["new_atm"],markersize=12,label="Suggested New ATM"))
    ax.legend(handles=legend_elements,loc="upper left",fontsize=9,framealpha=0.9,facecolor="white")
    _watermark(ax); _finish(fig,save_path)

# ── Convenience runner ────────────────────────────────────────────────────────
def generate_all_plots(data_dir="data", output_dir="data/plots"):
    from network_metrics     import load_metrics
    from community_detection import load_communities
    from optimizer           import load_classified
    os.makedirs(output_dir,exist_ok=True)
    G=load_graph(data_dir)
    metrics_df=load_metrics(data_dir)
    community_df=load_communities(data_dir)
    classified_df=load_classified(data_dir)
    worst=classified_df.nsmallest(3,"closeness")
    new_atm={"latitude":round(float(worst["latitude"].mean()),6),
             "longitude":round(float(worst["longitude"].mean()),6),
             "source_atms":worst["atm_id"].tolist()}
    print("\n[Module 6] Generating visualizations...")
    plot_network(G,save_path=os.path.join(output_dir,"01_network.png"))
    plot_top_atms(G,metrics_df,save_path=os.path.join(output_dir,"02_top_atms.png"))
    plot_communities(G,community_df,save_path=os.path.join(output_dir,"03_communities.png"))
    plot_overload_status(G,classified_df,new_atm,save_path=os.path.join(output_dir,"04_overload.png"))
    print(f"  All plots saved to {output_dir}/")

if __name__=="__main__":
    generate_all_plots(data_dir="data",output_dir="data/plots")
