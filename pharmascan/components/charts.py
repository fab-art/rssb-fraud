"""
Chart and visualization helpers for PharmaScan.

This module contains functions for creating matplotlib charts:
- Horizontal bar charts
- Time series charts
- Histograms
- Network graph data preparation
"""

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from pharmascan.config.settings import (
    ACCENT,
    ACCENT2,
    BORDER,
    CARD,
    DARK,
    DANGER,
    MUTED,
    TEXT,
    WARN,
)


def setup_plot_style():
    """Configure matplotlib plot style for dark theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": CARD,
            "axes.facecolor": DARK,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": BORDER,
            "grid.linewidth": 0.5,
            "font.family": "monospace",
            "font.size": 9,
        }
    )


def hbar_chart(labels: list, values: list, color, title: str, xlabel: str):
    """
    Create a horizontal bar chart.
    
    Args:
        labels: List of label strings
        values: List of numeric values
        color: Color string or list of colors
        title: Chart title
        xlabel: X-axis label
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(labels) * 0.42)))
    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=color if isinstance(color, list) else [color] * len(labels),
        height=0.65,
    )
    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            color=TEXT,
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlim(0, max(values) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def time_series_chart(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a time series chart showing monthly visit volume.
    
    Args:
        df: DataFrame with 'visit_date' column
        
    Returns:
        Matplotlib figure object or None if insufficient data
    """
    if "visit_date" not in df.columns:
        return None
    s = df["visit_date"].dropna()
    if len(s) < 2:
        return None
    monthly = s.dt.to_period("M").value_counts().sort_index()
    dates, vals = [str(p) for p in monthly.index], monthly.values
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(vals)), vals, alpha=0.2, color=ACCENT)
    ax.plot(range(len(vals)), vals, color=ACCENT, linewidth=2, marker="o", markersize=4)
    step = max(1, len(dates) // 12)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Visits")
    ax.set_title("Monthly Visit Volume", fontsize=11, fontweight="bold", color=TEXT, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def rapid_histogram(rapid: list[dict]) -> Optional[plt.Figure]:
    """
    Create a histogram showing distribution of rapid revisits.
    
    Args:
        rapid: List of rapid revisit dicts with 'days_apart' key
        
    Returns:
        Matplotlib figure object or None if no data
    """
    if not rapid:
        return None
    days = [r["days_apart"] for r in rapid]
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = list(range(1, max(days) + 2))
    _, bins_out, patches = ax.hist(days, bins=bins, color=WARN, edgecolor=CARD, rwidth=0.8)
    for patch, left in zip(patches, bins_out):
        if left <= 2:
            patch.set_facecolor(DANGER)
    ax.set_xlabel("Days Between Visits")
    ax.set_ylabel("Cases")
    ax.set_title("Rapid Revisit Distribution", fontsize=11, fontweight="bold", color=TEXT, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            mpatches.Patch(color=DANGER, label="≤2 days"),
            mpatches.Patch(color=WARN, label="3+ days"),
        ],
        fontsize=8,
        facecolor=CARD,
        edgecolor=BORDER,
        labelcolor=TEXT,
    )
    fig.tight_layout()
    return fig


def build_network_data(
    df: pd.DataFrame, col_a: str, col_b: str, max_nodes: int, min_edge_weight: int
):
    """
    Build graph data for vis.js interactive rendering.
    
    Args:
        df: Input DataFrame
        col_a: First column name for nodes
        col_b: Second column name for nodes
        max_nodes: Maximum number of nodes to include
        min_edge_weight: Minimum edge weight to include
        
    Returns:
        Tuple of (vis_nodes, vis_edges, stats) or (None, None, {})
    """
    sub = df[[col_a, col_b]].dropna()
    if len(sub) == 0:
        return None, None, {}

    G = nx.Graph()

    # Optimized: use groupby and size instead of iterrows loop
    edge_counts = (
        sub.groupby([col_a, col_b], dropna=False).size().reset_index(name="weight")
    )
    edge_counts = edge_counts[edge_counts["weight"] >= min_edge_weight]

    # Add edges with weights directly
    for _, row in edge_counts.iterrows():
        a, b, w = str(row[col_a]), str(row[col_b]), int(row["weight"])
        G.add_node(a, side="A")
        G.add_node(b, side="B")
        G.add_edge(a, b, weight=w)

    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes) == 0:
        return None, None, {}

    # Prune to top nodes by degree (optimized)
    if len(G.nodes) > max_nodes:
        degree_list = list(G.degree())
        degree_list.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, _ in degree_list[:max_nodes]]
        G = G.subgraph(top_nodes).copy()

    nodes_a = [n for n, d in G.nodes(data=True) if d.get("side") == "A"]
    nodes_b = [n for n, d in G.nodes(data=True) if d.get("side") == "B"]
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    col_a_lbl = col_a.replace("_", " ").title()
    col_b_lbl = col_b.replace("_", " ").title()

    # Build vis.js nodes list
    vis_nodes = []
    for n, data in G.nodes(data=True):
        deg = degrees.get(n, 1)
        is_a = data.get("side") == "A"
        size = (
            max(14, min(50, 10 + deg * 4))
            if is_a
            else max(8, min(30, 6 + deg * 2))
        )
        vis_nodes.append(
            {
                "id": n,
                "label": str(n)[:20] + ("…" if len(str(n)) > 20 else ""),
                "title": f"<b>{n}</b><br>Type: {col_a_lbl if is_a else col_b_lbl}<br>Connections: {deg}",
                "color": {
                    "background": "#00e5a0" if is_a else "#0ea5e9",
                    "border": "#00b87a" if is_a else "#0284c7",
                    "highlight": {
                        "background": "#ffffff",
                        "border": "#00e5a0" if is_a else "#0ea5e9",
                    },
                    "hover": {
                        "background": "#f0fdf4" if is_a else "#e0f2fe",
                        "border": "#00e5a0" if is_a else "#0ea5e9",
                    },
                },
                "shape": "diamond" if is_a else "dot",
                "size": size,
                "font": {"color": "#e2e8f0", "size": 11, "face": "DM Mono, monospace"},
                "group": "A" if is_a else "B",
                "degree": deg,
            }
        )

    # Build vis.js edges list
    max_w = max((G[u][v].get("weight", 1) for u, v in G.edges()), default=1)
    vis_edges = []
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        w = data.get("weight", 1)
        vis_edges.append(
            {
                "id": i,
                "from": u,
                "to": v,
                "weight": w,
                "width": max(0.5, min(6, 0.5 + 4 * (w / max_w))),
                "color": {
                    "color": "rgba(100,116,139,0.35)",
                    "highlight": "#00e5a0",
                    "hover": "#f59e0b",
                },
                "title": f"Co-occurrences: {w}",
                "smooth": {"type": "dynamic"},
            }
        )

    stats = {
        "nodes_a": len(nodes_a),
        "nodes_b": len(nodes_b),
        "edges": len(vis_edges),
        "density": round(nx.density(G), 4),
        "avg_degree": round(sum(degrees.values()) / max(1, len(degrees)), 2),
        "top_a": sorted([(n, degrees[n]) for n in nodes_a], key=lambda x: -x[1])[:10],
        "top_b": sorted([(n, degrees[n]) for n in nodes_b], key=lambda x: -x[1])[:10],
        "col_a_lbl": col_a_lbl,
        "col_b_lbl": col_b_lbl,
    }
    return vis_nodes, vis_edges, stats
