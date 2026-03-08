"""
PharmaScan — Pharmacy Voucher Intelligence (Streamlit Edition)

Install:
    pip install streamlit pandas matplotlib networkx openpyxl odfpy

Run:
    streamlit run pharmascan_streamlit.py
"""

import difflib
import io
import re
import warnings
from collections import defaultdict as _dd

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaScan",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colours ───────────────────────────────────────────────────────────────────
ACCENT  = "#00e5a0"
ACCENT2 = "#0ea5e9"
PURPLE  = "#a78bfa"
WARN    = "#f59e0b"
DANGER  = "#ef4444"
MUTED   = "#64748b"
TEXT    = "#e2e8f0"
DARK    = "#0d1117"
CARD    = "#111720"
BORDER  = "#1e2a38"

plt.rcParams.update({
    "figure.facecolor": CARD,  "axes.facecolor":  DARK,
    "axes.edgecolor":   BORDER,"axes.labelcolor": MUTED,
    "axes.titlecolor":  TEXT,  "xtick.color":     MUTED,
    "ytick.color":      MUTED, "text.color":      TEXT,
    "grid.color":       BORDER,"grid.linewidth":  0.5,
    "font.family":      "monospace", "font.size": 9,
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap');

.stApp { background: #080c10; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2a38; }

[data-testid="stMetric"] {
    background: #111720; border: 1px solid #1e2a38;
    border-radius: 12px; padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .5px; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 26px !important; font-weight: 800 !important; font-family: 'Syne', sans-serif !important; }

.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-bottom: 1px solid #1e2a38; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; font-weight: 600; border-radius: 0; border-bottom: 2px solid transparent; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: #00e5a0 !important; border-bottom: 2px solid #00e5a0 !important; background: transparent !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.sidebar-title { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: #e2e8f0; margin-bottom: 4px; }
.sidebar-sub   { font-size: 12px; color: #64748b; margin-bottom: 20px; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.chip {
    background: rgba(14,165,233,.08); border: 1px solid rgba(14,165,233,.2);
    border-radius: 6px; padding: 3px 10px; font-size: 11px;
    font-family: 'DM Mono', monospace; color: #0ea5e9;
}
.info-banner {
    background: rgba(14,165,233,.06); border: 1px solid rgba(14,165,233,.2);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
}
.info-banner b { color: #0ea5e9; }

.sec-head {
    font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700;
    color: #e2e8f0; padding-left: 10px; border-left: 3px solid #00e5a0;
    margin: 20px 0 12px;
}

.rapid-card {
    background: #111720; border: 1px solid #1e2a38;
    border-left: 3px solid #f59e0b; border-radius: 10px; padding: 12px 14px;
}
.rapid-card.crit { border-left-color: #ef4444; }
.rc-head { display: flex; justify-content: space-between; align-items: flex-start; }
.rc-name { font-size: 13px; font-weight: 600; color: #e2e8f0; }
.rc-id   { font-size: 11px; color: #64748b; font-family: 'DM Mono', monospace; margin-top: 2px; }
.rc-days { font-size: 24px; font-weight: 800; font-family: 'Syne', sans-serif; color: #f59e0b; line-height: 1; }
.rapid-card.crit .rc-days { color: #ef4444; }
.rc-days small { font-size: 11px; font-weight: 400; }
.rc-meta { font-size: 11px; color: #64748b; font-family: 'DM Mono', monospace; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ── Column normalisation — tuned to real Pharmacie Vinca / RAMA vouchers ──────
COLUMN_MAP = {
    # ── Exact columns from the real data ──────────────────────────────────────
    r"#":                                               "row_number",
    r"paper.?code":                                     "voucher_id",
    r"dispensing.?date":                                "visit_date",
    r"patient.?name":                                   "patient_name",
    r"patient.?type":                                   "patient_type",
    r"gender":                                          "gender",
    r"is.?newborn":                                     "is_newborn",
    r"rama.?number":                                    "patient_id",
    r"practitioner.?name":                              "doctor_name",
    r"practitioner.?type":                              "doctor_type",
    r"total.?cost":                                     "amount",
    r"patient.?co.?payment":                            "patient_copay",
    r"insurance.?co.?payment":                          "insurance_copay",
    r"medicine.?cost":                                  "medicine_cost",
    # ── Generic fallbacks ─────────────────────────────────────────────────────
    r"patient.?(id|no|num|number|code)?":               "patient_id",
    r"pat.?id|pid":                                     "patient_id",
    r"doctor.?(id|no|num|code)?":                       "doctor_id",
    r"(doctor|dr|physician|prescriber).?name":          "doctor_name",
    r"doc.?id|did":                                     "doctor_id",
    r"prescriber":                                      "doctor_name",
    r"(visit|service|rx|voucher).?date":                "visit_date",
    r"date.?(of.?)?(visit|service|dispensing)?":        "visit_date",
    r"date":                                            "visit_date",
    r"(pharmacy|facility|clinic|hospital|branch).?(name|id|code)?": "facility",
    r"(drug|medicine|medication|item|product).?(name|description|desc)?": "drug_name",
    r"(drug|medicine|medication|item|product).?(code|id)?":              "drug_code",
    r"(amount|cost|price|value|total|charge)":          "amount",
    r"quantity|qty":                                    "quantity",
    r"(diagnosis|diag|icd|condition)":                  "diagnosis",
    r"(voucher|claim|ref|reference).?(no|number|id|code)?": "voucher_id",
}


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes: bytes, filename: str, rapid_days: int):
    fname = filename.lower()
    if fname.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", on_bad_lines="skip")
    elif fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    elif fname.endswith(".ods"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="odf")
    else:
        raise ValueError("Unsupported file type. Use CSV, XLSX, XLS, or ODS.")

    # Normalise column names
    renamed, used = {}, {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "_", col.lower().strip())
        key = re.sub(r"_+", "_", key).strip("_")
        matched = False
        for pattern, target in COLUMN_MAP.items():
            if re.fullmatch(pattern, key):
                if target not in used:
                    renamed[col] = target
                    used[target] = col
                    matched = True
                break
        if not matched:
            renamed[col] = key
    df = df.rename(columns=renamed)

    # Parse dates
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    else:
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > len(df) * 0.5:
                        df["visit_date"] = parsed
                        break
                except Exception:
                    pass

    # Summary stats
    s = {"total_rows": len(df), "columns": list(df.columns)}
    id_col = "patient_id" if "patient_id" in df.columns else "patient_name" if "patient_name" in df.columns else None
    if id_col:
        vc = df[id_col].value_counts()
        s["patient_col"]     = id_col
        s["unique_patients"] = int(df[id_col].nunique())
        s["repeat_patients"] = int((vc > 1).sum())
        s["max_visits"]      = int(vc.max())
        s["top_patients"]    = vc.head(15).rename_axis("id").reset_index(name="visits")
    dcol = "doctor_name" if "doctor_name" in df.columns else "doctor_id" if "doctor_id" in df.columns else None
    if dcol:
        dvc = df[dcol].value_counts()
        s["unique_doctors"] = int(df[dcol].nunique())
        s["top_doctors"]    = dvc.head(15).rename_axis("doctor").reset_index(name="visits")
        s["doctor_col"]     = dcol
    if "visit_date" in df.columns:
        v = df["visit_date"].dropna()
        if len(v):
            s["date_min"] = str(v.min().date())
            s["date_max"] = str(v.max().date())
    if "facility" in df.columns:
        fvc = df["facility"].value_counts()
        s["unique_facilities"] = int(df["facility"].nunique())
        s["top_facilities"]    = fvc.head(10).rename_axis("name").reset_index(name="visits")
    for amt_col in ["amount", "medicine_cost", "insurance_copay", "patient_copay"]:
        if amt_col in df.columns:
            df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    if "amount" in df.columns:
        s["total_amount"] = round(float(df["amount"].sum()), 2)
        s["avg_amount"]   = round(float(df["amount"].mean()), 2)

    # Repeat visits
    repeat_groups, repeat_detail = [], pd.DataFrame()
    if id_col:
        vc2 = df[id_col].value_counts()
        repeat_ids = vc2[vc2 > 1].index.tolist()
        rdf = df[df[id_col].isin(repeat_ids)].copy()
        if "visit_date" in rdf.columns:
            rdf = rdf.sort_values([id_col, "visit_date"])
        repeat_detail = rdf.head(500)
        for pid in repeat_ids[:300]:
            grp = df[df[id_col] == pid]
            entry = {id_col: str(pid), "visits": int(len(grp))}
            if "patient_name" in grp.columns and id_col != "patient_name":
                entry["patient_name"] = str(grp["patient_name"].iloc[0])
            if "visit_date" in grp.columns:
                dates = grp["visit_date"].dropna().sort_values()
                entry["dates"] = ", ".join(str(d.date()) for d in dates if pd.notna(d))
            repeat_groups.append(entry)
        repeat_groups.sort(key=lambda x: x["visits"], reverse=True)

    # Rapid revisits
    rapid = []
    if id_col and "visit_date" in df.columns:
        cols = [id_col, "visit_date"]
        if "patient_name" in df.columns and id_col != "patient_name":
            cols.append("patient_name")
        if dcol:
            cols.append(dcol)
        sub = df[cols].dropna(subset=[id_col, "visit_date"]).sort_values([id_col, "visit_date"])
        for pid, grp in sub.groupby(id_col):
            dates = grp["visit_date"].tolist()
            for i in range(len(dates) - 1):
                diff = (dates[i + 1] - dates[i]).days
                if 0 < diff <= rapid_days:
                    name = str(grp["patient_name"].iloc[0]) if "patient_name" in grp.columns else str(pid)
                    rapid.append({
                        "patient_id":   str(pid),
                        "patient_name": name,
                        "visit_1":      str(dates[i].date()),
                        "visit_2":      str(dates[i + 1].date()),
                        "days_apart":   diff,
                        "doctor":       str(grp[dcol].iloc[0]) if dcol else "—",
                    })
        rapid.sort(key=lambda x: x["days_apart"])

    return df, renamed, s, repeat_groups, repeat_detail, rapid


# ── Chart helpers ─────────────────────────────────────────────────────────────

def hbar_chart(labels, values, color, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(labels) * 0.42)))
    bars = ax.barh(labels[::-1], values[::-1],
                   color=color if isinstance(color, list) else [color] * len(labels),
                   height=0.65)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=TEXT, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlim(0, max(values) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def time_series_chart(df):
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


def rapid_histogram(rapid):
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
    ax.legend(handles=[mpatches.Patch(color=DANGER, label="≤2 days"),
                        mpatches.Patch(color=WARN,   label="3+ days")],
              fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout()
    return fig


def build_network_data(df: pd.DataFrame, col_a: str, col_b: str,
                       max_nodes: int, min_edge_weight: int):
    """
    Build graph data for vis.js interactive rendering.
    Returns (vis_nodes, vis_edges, stats) or (None, None, {}).
    """
    sub = df[[col_a, col_b]].dropna()
    if len(sub) == 0:
        return None, None, {}

    G = nx.Graph()
    edge_w: dict = {}
    for _, row in sub.iterrows():
        a, b = str(row[col_a]), str(row[col_b])
        G.add_node(a, side="A")
        G.add_node(b, side="B")
        key = (a, b)
        edge_w[key] = edge_w.get(key, 0) + 1

    for (a, b), w in edge_w.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes) == 0:
        return None, None, {}

    # Prune to top nodes by degree
    if len(G.nodes) > max_nodes:
        top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()

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
        size = max(14, min(50, 10 + deg * 4)) if is_a else max(8, min(30, 6 + deg * 2))
        vis_nodes.append({
            "id":    n,
            "label": str(n)[:20] + ("…" if len(str(n)) > 20 else ""),
            "title": f"<b>{n}</b><br>Type: {col_a_lbl if is_a else col_b_lbl}<br>Connections: {deg}",
            "color": {
                "background":  "#00e5a0" if is_a else "#0ea5e9",
                "border":      "#00b87a" if is_a else "#0284c7",
                "highlight":   {"background": "#ffffff", "border": "#00e5a0" if is_a else "#0ea5e9"},
                "hover":       {"background": "#f0fdf4" if is_a else "#e0f2fe",
                                "border":     "#00e5a0" if is_a else "#0ea5e9"},
            },
            "shape": "diamond" if is_a else "dot",
            "size":  size,
            "font":  {"color": "#e2e8f0", "size": 11, "face": "DM Mono, monospace"},
            "group": "A" if is_a else "B",
            "degree": deg,
        })

    # Build vis.js edges list
    max_w = max((G[u][v].get("weight", 1) for u, v in G.edges()), default=1)
    vis_edges = []
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        w = data.get("weight", 1)
        vis_edges.append({
            "id":     i,
            "from":   u,
            "to":     v,
            "weight": w,
            "width":  max(0.5, min(6, 0.5 + 4 * (w / max_w))),
            "color":  {
                "color":     "rgba(100,116,139,0.35)",
                "highlight": "#00e5a0",
                "hover":     "#f59e0b",
            },
            "title":  f"Co-occurrences: {w}",
            "smooth": {"type": "dynamic"},
        })

    stats = {
        "nodes_a":    len(nodes_a),
        "nodes_b":    len(nodes_b),
        "edges":      len(vis_edges),
        "density":    round(nx.density(G), 4),
        "avg_degree": round(sum(degrees.values()) / max(1, len(degrees)), 2),
        "top_a": sorted([(n, degrees[n]) for n in nodes_a], key=lambda x: -x[1])[:10],
        "top_b": sorted([(n, degrees[n]) for n in nodes_b], key=lambda x: -x[1])[:10],
        "col_a_lbl": col_a_lbl,
        "col_b_lbl": col_b_lbl,
    }
    return vis_nodes, vis_edges, stats


def render_vis_network(vis_nodes, vis_edges, stats, physics_mode: str, height: int = 680):
    """Render an interactive vis.js network via st.components.v1.html()."""
    import json
    import streamlit.components.v1 as components

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    col_a_lbl  = stats.get("col_a_lbl", "Node A")
    col_b_lbl  = stats.get("col_b_lbl", "Node B")
    n_a = stats.get("nodes_a", 0)
    n_b = stats.get("nodes_b", 0)

    physics_opts = {
        "Force Atlas 2": json.dumps({
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -60, "centralGravity": 0.01,
                                  "springLength": 120, "springConstant": 0.08, "damping": 0.4},
            "stabilization": {"iterations": 150},
        }),
        "Barnes-Hut": json.dumps({
            "solver": "barnesHut",
            "barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3,
                           "springLength": 140, "springConstant": 0.04, "damping": 0.09},
            "stabilization": {"iterations": 150},
        }),
        "Repulsion": json.dumps({
            "solver": "repulsion",
            "repulsion": {"centralGravity": 0.2, "springLength": 200,
                           "springConstant": 0.05, "nodeDistance": 150, "damping": 0.09},
            "stabilization": {"iterations": 150},
        }),
        "None (static)": json.dumps({"enabled": False}),
    }
    physics_json = physics_opts.get(physics_mode, physics_opts["Force Atlas 2"])

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #080c10; font-family: 'DM Mono', monospace; color: #e2e8f0; overflow: hidden; }}

  #net-wrap {{ position: relative; width: 100%; height: {height}px; background: #0d1117;
               border: 1px solid #1e2a38; border-radius: 12px; overflow: hidden; }}
  #network  {{ width: 100%; height: 100%; }}

  /* Toolbar */
  #toolbar {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  }}
  .tb-btn {{
    background: rgba(17,23,32,.92); border: 1px solid #1e2a38;
    color: #e2e8f0; border-radius: 7px; padding: 6px 12px;
    font-size: 11px; font-family: monospace; cursor: pointer;
    transition: all .15s; backdrop-filter: blur(4px);
  }}
  .tb-btn:hover {{ border-color: #00e5a0; color: #00e5a0; }}
  .tb-btn.active {{ background: rgba(0,229,160,.12); border-color: #00e5a0; color: #00e5a0; }}
  .tb-sep {{ width: 1px; height: 22px; background: #1e2a38; }}

  /* Search box */
  #search-wrap {{ position: absolute; top: 12px; right: 12px; z-index: 10; display: flex; gap: 6px; }}
  #node-search {{
    background: rgba(17,23,32,.92); border: 1px solid #1e2a38;
    color: #e2e8f0; border-radius: 7px; padding: 6px 12px;
    font-size: 11px; font-family: monospace; width: 180px; outline: none;
    backdrop-filter: blur(4px);
  }}
  #node-search:focus {{ border-color: #00e5a0; }}
  #node-search::placeholder {{ color: #64748b; }}

  /* Legend */
  #legend {{
    position: absolute; bottom: 14px; left: 14px; z-index: 10;
    background: rgba(13,17,23,.88); border: 1px solid #1e2a38;
    border-radius: 10px; padding: 10px 14px; backdrop-filter: blur(4px);
    font-size: 11px;
  }}
  .leg-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }}
  .leg-row:last-child {{ margin-bottom: 0; }}
  .leg-dot {{ width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }}
  .leg-dia {{ width: 10px; height: 10px; transform: rotate(45deg); flex-shrink: 0; border-radius: 1px; }}

  /* Stats bar */
  #stats-bar {{
    position: absolute; bottom: 14px; right: 14px; z-index: 10;
    background: rgba(13,17,23,.88); border: 1px solid #1e2a38;
    border-radius: 10px; padding: 10px 14px; backdrop-filter: blur(4px);
    font-size: 11px; color: #64748b; line-height: 1.7;
  }}
  #stats-bar b {{ color: #e2e8f0; }}

  /* Tooltip override */
  .vis-tooltip {{
    background: #111720 !important; border: 1px solid #1e2a38 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 12px !important;
    padding: 8px 12px !important; box-shadow: 0 4px 20px rgba(0,0,0,.4) !important;
  }}

  /* Selected info panel */
  #info-panel {{
    display: none; position: absolute; top: 56px; right: 12px; z-index: 10;
    background: rgba(13,17,23,.95); border: 1px solid #1e2a38;
    border-radius: 10px; padding: 14px 16px; font-size: 12px;
    min-width: 200px; max-width: 260px; backdrop-filter: blur(4px);
  }}
  #info-panel .ip-name {{ font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; word-break: break-all; }}
  #info-panel .ip-row  {{ display: flex; justify-content: space-between; margin-bottom: 3px; }}
  #info-panel .ip-lbl  {{ color: #64748b; }}
  #info-panel .ip-val  {{ color: #e2e8f0; font-weight: 600; }}
  #info-panel .ip-close {{ float: right; cursor: pointer; color: #64748b; font-size: 14px; margin-left: 8px; }}
  #info-panel .ip-close:hover {{ color: #ef4444; }}
  #info-panel .ip-nbrs {{ margin-top: 8px; border-top: 1px solid #1e2a38; padding-top: 8px; }}
  #info-panel .ip-nbr  {{ color: #64748b; font-size: 11px; margin-bottom: 2px; }}

  #stabilizing {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    background: rgba(13,17,23,.9); border: 1px solid #1e2a38; border-radius: 10px;
    padding: 16px 24px; font-size: 13px; color: #00e5a0; z-index: 20;
    display: flex; align-items: center; gap: 10px; }}
  .spin {{ width: 16px; height: 16px; border: 2px solid rgba(0,229,160,.2);
    border-top-color: #00e5a0; border-radius: 50%; animation: spin .7s linear infinite; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>

<div id="net-wrap">
  <div id="stabilizing"><div class="spin"></div> Laying out graph…</div>

  <div id="toolbar">
    <button class="tb-btn" onclick="zoomIn()">＋</button>
    <button class="tb-btn" onclick="zoomOut()">－</button>
    <button class="tb-btn" onclick="fitAll()">⊡ Fit</button>
    <div class="tb-sep"></div>
    <button class="tb-btn" id="btn-physics" onclick="togglePhysics()">⏸ Freeze</button>
    <button class="tb-btn" onclick="highlightHubs()">★ Hubs</button>
    <button class="tb-btn" onclick="resetHighlight()">↺ Reset</button>
    <div class="tb-sep"></div>
    <button class="tb-btn" id="btn-labels" onclick="toggleLabels()">🏷 Labels</button>
  </div>

  <div id="search-wrap">
    <input id="node-search" placeholder="🔍 Search node…" oninput="searchNode(this.value)">
  </div>

  <div id="network"></div>

  <div id="info-panel">
    <div><span class="ip-close" onclick="closeInfo()">✕</span><div class="ip-name" id="ip-name"></div></div>
    <div class="ip-row"><span class="ip-lbl">Type</span><span class="ip-val" id="ip-type"></span></div>
    <div class="ip-row"><span class="ip-lbl">Connections</span><span class="ip-val" id="ip-deg"></span></div>
    <div class="ip-nbrs"><div style="color:#64748b;font-size:10px;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px">Connected to</div>
      <div id="ip-nbr-list"></div>
    </div>
  </div>

  <div id="legend">
    <div class="leg-row"><div class="leg-dia" style="background:#00e5a0"></div><span>{col_a_lbl} ({n_a})</span></div>
    <div class="leg-row"><div class="leg-dot" style="background:#0ea5e9"></div><span>{col_b_lbl} ({n_b})</span></div>
  </div>

  <div id="stats-bar">
    <div><b>{stats["nodes_a"] + stats["nodes_b"]}</b> nodes</div>
    <div><b>{stats["edges"]}</b> edges</div>
    <div>avg degree <b>{stats["avg_degree"]}</b></div>
    <div>density <b>{stats["density"]}</b></div>
  </div>
</div>

<script>
const nodesData = {nodes_json};
const edgesData = {edges_json};

// Build lookup maps
const nodeMap = {{}};
nodesData.forEach(n => nodeMap[n.id] = n);

const adjMap = {{}};
edgesData.forEach(e => {{
  if (!adjMap[e.from]) adjMap[e.from] = [];
  if (!adjMap[e.to])   adjMap[e.to]   = [];
  adjMap[e.from].push({{id: e.to,   w: e.weight}});
  adjMap[e.to].push(  {{id: e.from, w: e.weight}});
}});

const nodes = new vis.DataSet(nodesData);
const edges = new vis.DataSet(edgesData);

const container = document.getElementById('network');
const physicsConfig = {physics_json};

const options = {{
  nodes: {{ borderWidth: 1.5, shadow: {{ enabled: true, color: 'rgba(0,0,0,.5)', x: 2, y: 2, size: 8 }} }},
  edges: {{ smooth: {{ type: 'dynamic' }}, shadow: false, selectionWidth: 3 }},
  physics: physicsConfig,
  interaction: {{
    hover: true, tooltipDelay: 150,
    navigationButtons: false,
    keyboard: true,
    multiselect: true,
    zoomView: true,
  }},
  layout: {{ improvedLayout: true }},
}};

const network = new vis.Network(container, {{ nodes, edges }}, options);

// Hide spinner once stabilised
network.on('stabilizationIterationsDone', () => {{
  document.getElementById('stabilizing').style.display = 'none';
  network.setOptions({{ physics: {{ enabled: false }} }});
  document.getElementById('btn-physics').textContent = '▶ Unfreeze';
  physicsRunning = false;
}});
network.on('stabilized', () => {{
  document.getElementById('stabilizing').style.display = 'none';
}});

// State
let physicsRunning = true;
let labelsVisible  = true;
let hubsHighlighted = false;

// ── Controls ──
function zoomIn()  {{ network.moveTo({{ scale: network.getScale() * 1.3, animation: true }}); }}
function zoomOut() {{ network.moveTo({{ scale: network.getScale() * 0.77, animation: true }}); }}
function fitAll()  {{ network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }}); }}

function togglePhysics() {{
  physicsRunning = !physicsRunning;
  network.setOptions({{ physics: {{ enabled: physicsRunning }} }});
  document.getElementById('btn-physics').textContent = physicsRunning ? '⏸ Freeze' : '▶ Unfreeze';
  document.getElementById('btn-physics').classList.toggle('active', physicsRunning);
}}

function toggleLabels() {{
  labelsVisible = !labelsVisible;
  const update = nodesData.map(n => ({{
    id: n.id,
    font: {{ ...n.font, color: labelsVisible ? '#e2e8f0' : 'rgba(0,0,0,0)' }}
  }}));
  nodes.update(update);
  document.getElementById('btn-labels').classList.toggle('active', labelsVisible);
}}

function highlightHubs() {{
  hubsHighlighted = !hubsHighlighted;
  if (hubsHighlighted) {{
    const maxDeg = Math.max(...nodesData.map(n => n.degree));
    const thresh = maxDeg * 0.5;
    const update = nodesData.map(n => ({{
      id: n.id,
      opacity: n.degree >= thresh ? 1.0 : 0.15,
    }}));
    nodes.update(update);
  }} else {{
    resetHighlight();
  }}
}}

function resetHighlight() {{
  hubsHighlighted = false;
  nodes.update(nodesData.map(n => ({{ id: n.id, opacity: 1.0 }})));
  document.getElementById('node-search').value = '';
}}

function searchNode(q) {{
  q = q.trim().toLowerCase();
  if (!q) {{ resetHighlight(); return; }}
  const update = nodesData.map(n => ({{
    id: n.id,
    opacity: n.id.toLowerCase().includes(q) ? 1.0 : 0.1,
  }}));
  nodes.update(update);
  // Focus first match
  const match = nodesData.find(n => n.id.toLowerCase().includes(q));
  if (match) {{
    network.focus(match.id, {{ scale: 1.4, animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }});
  }}
}}

// ── Click → info panel ──
network.on('click', params => {{
  if (params.nodes.length === 1) {{
    const nid = params.nodes[0];
    const nd  = nodeMap[nid];
    if (!nd) return;
    document.getElementById('ip-name').textContent = nid;
    document.getElementById('ip-type').textContent = nd.group === 'A' ? '{col_a_lbl}' : '{col_b_lbl}';
    document.getElementById('ip-deg').textContent  = nd.degree;
    const nbrs = (adjMap[nid] || []).sort((a,b) => b.w - a.w).slice(0, 10);
    document.getElementById('ip-nbr-list').innerHTML =
      nbrs.map(n => `<div class="ip-nbr">• ${{n.id.length > 24 ? n.id.slice(0,24)+'…' : n.id}} <span style="color:#00e5a0">×${{n.w}}</span></div>`).join('');
    document.getElementById('info-panel').style.display = 'block';

    // Dim non-neighbours
    const connectedIds = new Set([nid, ...nbrs.map(n => n.id)]);
    nodes.update(nodesData.map(n => ({{ id: n.id, opacity: connectedIds.has(n.id) ? 1.0 : 0.1 }})));
    edges.update(edgesData.map(e => ({{
      id: e.id,
      color: {{
        color: (e.from === nid || e.to === nid) ? '#00e5a0' : 'rgba(100,116,139,0.06)',
        highlight: '#00e5a0', hover: '#f59e0b',
      }},
    }})));
  }} else {{
    closeInfo();
    resetHighlight();
  }}
}});

// ── Double-click → zoom to node ──
network.on('doubleClick', params => {{
  if (params.nodes.length === 1) {{
    network.focus(params.nodes[0], {{ scale: 2.0, animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
  }}
}});

// ── Hover ──
network.on('hoverNode', params => {{
  container.style.cursor = 'pointer';
}});
network.on('blurNode',  () => {{ container.style.cursor = 'default'; }});

function closeInfo() {{
  document.getElementById('info-panel').style.display = 'none';
  resetHighlight();
}}
</script>
</body>
</html>"""

    components.html(html, height=height + 10, scrolling=False)


def fmt_number(n):
    if n >= 1e9:  return f"{n/1e9:.1f}B"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{n:,.0f}"


# ── Name normalisation engine ─────────────────────────────────────────────────

def _toks(name: str) -> set:
    """Lowercase alpha-numeric tokens from a name string."""
    return set(re.sub(r"[^a-z0-9 ]", "", name.lower()).split())

def _seq_ratio(a: str, b: str) -> float:
    sa = " ".join(sorted(_toks(a)))
    sb = " ".join(sorted(_toks(b)))
    return difflib.SequenceMatcher(None, sa, sb).ratio()

def _tok_fuzzy_subset(a: str, b: str, thresh: float = 0.76) -> bool:
    """
    True if every token in the SHORTER name has a fuzzy-close counterpart
    in the LONGER name (catches 'Aurbain'/'Urbain', 'Constatin'/'Constantin').
    """
    ta = list(_toks(a))
    tb = list(_toks(b))
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    for tok in shorter:
        best = max((difflib.SequenceMatcher(None, tok, lt).ratio() for lt in longer), default=0)
        if best < thresh:
            return False
    return True

def _match_score(a: str, b: str):
    """
    Returns (score 0–1, reason str).
    reason ∈ {'subset', 'typo', 'none'}
    """
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0, "none"

    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

    # Rule 1 — exact token subset ('ZACHEE' ⊂ 'Niyonsenga Zachee')
    if shorter <= longer:
        boost = min(0.12, len(shorter) * 0.04)
        return 0.88 + boost, "subset"

    # Rule 2 — fuzzy-token subset: every short token ≈ some long token
    if _tok_fuzzy_subset(a, b):
        return 0.85, "typo"

    # Rule 3 — high overall char-sequence similarity
    ratio = _seq_ratio(a, b)
    if ratio >= 0.88:
        return ratio, "typo"

    return 0.0, "none"


def detect_name_clusters(names: list, counts: dict) -> list[dict]:
    """
    Cluster similar names using Union-Find.
    Returns list of dicts:
      { "canonical": str, "variants": [str,...], "method": str, "confidence": float }
    Only clusters with ≥2 members are returned.
    """
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            if len(_toks(pa)) >= len(_toks(pb)):
                parent[pb] = pa
            else:
                parent[pa] = pb

    # ── Pass 1: merge multi-token names (typos + reordering) ─────────────────
    multi = [n for n in names if len(_toks(n)) >= 2]
    for i, a in enumerate(multi):
        for b in multi[i + 1:]:
            sc, why = _match_score(a, b)
            if sc > 0 and why != "none":
                union(a, b)

    # ── Pass 2: single-token names → merge only if token is unique to 1 cluster ──
    def get_clusters():
        c = _dd(list)
        for n in names:
            c[find(n)].append(n)
        return c

    cls1 = get_clusters()
    tok_to_roots: dict = _dd(set)
    for root, members in cls1.items():
        if len(members) > 1:
            for m in members:
                for t in _toks(m):
                    tok_to_roots[t].add(root)

    for name in names:
        if len(_toks(name)) != 1:
            continue
        tok = next(iter(_toks(name)))
        candidates = tok_to_roots.get(tok, set()) - {find(name)}
        if len(candidates) == 1:
            union(name, next(iter(candidates)))

    # ── Build final clusters ─────────────────────────────────────────────────
    final: dict = _dd(list)
    for n in names:
        final[find(n)].append(n)

    def best_canonical(members):
        def score(n):
            tc   = len(_toks(n))
            freq = counts.get(n, 0)
            # Title-case preferred; "Dr " prefix demoted
            titled  = n == n.title()
            no_pfx  = not re.match(r"^(Dr|DR)\s", n)
            return (tc, no_pfx, titled, freq, len(n))
        return max(members, key=score)

    results = []
    for root, members in final.items():
        if len(members) < 2:
            continue
        canon = best_canonical(members)
        variants = [m for m in members if m != canon]
        # Compute overall confidence
        scores = [_match_score(canon, v)[0] for v in variants]
        conf   = round(sum(scores) / len(scores), 3) if scores else 1.0
        # Flag suspicious: variant shares NO token with canonical
        ct = _toks(canon)
        suspicious = any(not (_toks(v) & ct) for v in variants)
        results.append({
            "canonical":  canon,
            "variants":   sorted(variants, key=lambda x: (-counts.get(x, 0), -len(x))),
            "confidence": conf,
            "suspicious": suspicious,
            "count":      len(members),
        })
    results.sort(key=lambda x: (-x["count"], -x["confidence"]))
    return results


def apply_name_normalisation(df: pd.DataFrame, col: str,
                              approved_clusters: list[dict]) -> pd.DataFrame:
    """Apply approved rename clusters to a column in a copy of df."""
    df = df.copy()
    mapping = {}
    for c in approved_clusters:
        for v in c["variants"]:
            mapping[v] = c["canonical"]
    df[col] = df[col].map(lambda x: mapping.get(x, x))
    return df





def _build_rssb_logo_bytes() -> bytes:
    """
    Render the RSSB logo faithful to the official brand:
    - Navy outer ring with 'RWANDA SOCIAL' (top arc) / 'SECURITY BOARD' (bottom arc)
    - Gold sunburst ring
    - RSSB Blue (#005BAA) shield with 'RSSB' in white
    - 'Our Health / Our Future' tagline in navy
    Returns PNG bytes. Falls back gracefully if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import math as _math

        RSSB_BLUE  = (0, 91, 170)
        NAVY_DARK  = (12, 22, 78)
        GOLD_RING  = (225, 178, 50)
        WHITE_C    = (255, 255, 255)
        DARK_TEXT  = (14, 24, 80)

        SCALE = 5
        FW, FH = 580, 160
        img = Image.new("RGBA", (FW * SCALE, FH * SCALE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        CX = int(78 * SCALE); CY = int(80 * SCALE)
        R_OUT  = int(72 * SCALE); R_GO = int(61 * SCALE)
        R_GI   = int(50 * SCALE)
        SHW    = int(84 * SCALE); SHH = int(90 * SCALE)

        draw.ellipse([CX-R_OUT, CY-R_OUT, CX+R_OUT, CY+R_OUT], fill=NAVY_DARK)
        draw.ellipse([CX-R_GO,  CY-R_GO,  CX+R_GO,  CY+R_GO ], fill=GOLD_RING)
        draw.ellipse([CX-R_GI,  CY-R_GI,  CX+R_GI,  CY+R_GI ], fill=NAVY_DARK)

        for i in range(14):
            a = _math.radians(-165 + i * (330 / 13))
            r1, r2 = R_GI + int(2*SCALE), R_GO - int(2*SCALE)
            draw.line([CX + int(r1*_math.cos(a)), CY + int(r1*_math.sin(a)),
                       CX + int(r2*_math.cos(a)), CY + int(r2*_math.sin(a))],
                      fill=(190, 140, 15), width=int(1.2*SCALE))
        draw.ellipse([CX-R_GI, CY-R_GI, CX+R_GI, CY+R_GI], fill=NAVY_DARK)

        def _shield(cx, cy, w, h, c):
            r = int(w * 0.16); pts = []
            for a in range(180, 271, 4):
                pts.append((cx-w//2+r+int(r*_math.cos(_math.radians(a))),
                             cy-h//2+r+int(r*_math.sin(_math.radians(a)))))
            for a in range(270, 361, 4):
                pts.append((cx+w//2-r+int(r*_math.cos(_math.radians(a))),
                             cy-h//2+r+int(r*_math.sin(_math.radians(a)))))
            pts += [(cx+w//2, cy+h//4), (cx, cy+h//2), (cx-w//2, cy+h//4)]
            draw.polygon(pts, fill=c)
        _shield(CX, CY, SHW, SHH, RSSB_BLUE)

        try:
            f_rssb = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(26*SCALE))
            f_ring = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(7*SCALE))
            f_tag  = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(17*SCALE))
        except Exception:
            f_rssb = f_ring = f_tag = ImageFont.load_default()

        bb = draw.textbbox((0,0), "RSSB", font=f_rssb)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        draw.text((CX-tw//2, CY-th//2-int(2*SCALE)), "RSSB", fill=WHITE_C, font=f_rssb)

        def _cw(ch):
            b = draw.textbbox((0,0), ch, font=f_ring); return b[2]-b[0]

        def _arc_top(text, cx, cy, radius, center_deg, spacing=1.10):
            arcs = [_math.degrees(_cw(c)*spacing/radius) for c in text]
            angle = center_deg - sum(arcs)/2
            for ch, arc in zip(text, arcs):
                mid = angle + arc/2
                rad = _math.radians(mid)
                x = cx + radius*_math.cos(rad); y = cy + radius*_math.sin(rad)
                b = draw.textbbox((0,0), ch, font=f_ring)
                ci = Image.new("RGBA", (b[2]-b[0]+6, b[3]-b[1]+6), (0,0,0,0))
                ImageDraw.Draw(ci).text((3,3), ch, font=f_ring, fill=WHITE_C)
                rot = ci.rotate(-(mid+90), expand=True, resample=Image.BICUBIC)
                img.paste(rot, (int(x-rot.width/2), int(y-rot.height/2)), rot)
                angle += arc

        def _arc_bottom(text, cx, cy, radius, center_deg, spacing=1.10):
            arcs = [_math.degrees(_cw(c)*spacing/radius) for c in text]
            angle = center_deg + sum(arcs)/2
            for ch, arc in zip(text, arcs):
                mid = angle - arc/2
                rad = _math.radians(mid)
                x = cx + radius*_math.cos(rad); y = cy + radius*_math.sin(rad)
                b = draw.textbbox((0,0), ch, font=f_ring)
                ci = Image.new("RGBA", (b[2]-b[0]+6, b[3]-b[1]+6), (0,0,0,0))
                ImageDraw.Draw(ci).text((3,3), ch, font=f_ring, fill=WHITE_C)
                rot = ci.rotate(-(mid-90), expand=True, resample=Image.BICUBIC)
                img.paste(rot, (int(x-rot.width/2), int(y-rot.height/2)), rot)
                angle -= arc

        TEXT_R = int(55.5*SCALE)
        _arc_top(   "RWANDA SOCIAL",  CX, CY, TEXT_R, -90)
        _arc_bottom("SECURITY BOARD", CX, CY, TEXT_R,  90)

        TX = int(168*SCALE); TY = CY - int(26*SCALE)
        draw.line([TX-int(14*SCALE), CY-int(35*SCALE), TX-int(14*SCALE), CY+int(35*SCALE)],
                  fill=(160, 165, 200), width=int(1.5*SCALE))
        bb1 = draw.textbbox((0,0), "Our Health", font=f_tag)
        lh  = bb1[3]-bb1[1]
        draw.text((TX, TY),                    "Our Health", fill=DARK_TEXT, font=f_tag)
        draw.text((TX, TY+lh+int(8*SCALE)),    "Our Future", fill=DARK_TEXT, font=f_tag)

        final = img.resize((FW, FH), Image.LANCZOS)
        buf = io.BytesIO()
        final.save(buf, "PNG", optimize=True)
        buf.seek(0)
        return buf.read()
    except Exception:
        return b""


def generate_counter_verification_xlsx(
    df: pd.DataFrame,
    deductions: list[dict],
    meta: dict,
    prepared_by: str,
    verified_by: str,
    approved_by: str,
    pc_col:  str | None = None,
    ins_col: str | None = None,
    tot_col: str | None = None,
    obs_col: str | None = None,
    dif_col: str | None = None,
) -> bytes:
    """
    Generate RSSB-branded counter-verification Excel report.
    Colours: #005BAA (RSSB Blue), #00A651 (Green), #F4B400 (Gold), #333333 (text)

    Sheet 1 — 'Counter Verification Report'   (summary + deduction schedule)
    Sheet 2 — 'After Counter Verification'    (full 1,800+ row listing)
    """
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as _XLImage
    from openpyxl.styles import (Font as _Font, PatternFill as _PFill,
        Alignment as _Align, Border as _Border, Side as _Side)
    from openpyxl.utils import get_column_letter as _gcl
    import io as _io
    from datetime import datetime as _dt

    # ── Official brand palette ────────────────────────────────────────────────
    _NAVY       = "005BAA"   # RSSB Blue — primary
    _NAVY_DK    = "003D7A"   # darker navy for text contrast
    _GREEN      = "00A651"   # RSSB Green — approvals / positive
    _GOLD       = "F4B400"   # RSSB Gold — accent
    _GOLD_DK    = "C9920A"   # darker gold for borders
    _WHITE      = "FFFFFF"
    _GRAY_LT    = "F2F2F2"   # table background (from brand guide)
    _GRAY_MID   = "D6DCE4"
    _GRAY_TEXT  = "333333"   # official dark gray body text
    _RED        = "C00000"
    _RED_BG     = "FCE4D6"
    _AMBER_BG   = "FFF2CC"
    _GREEN_BG   = "E8F5EE"
    _BLUE_LT    = "EBF3FB"   # light tint of RSSB Blue

    # ── Style micro-helpers ───────────────────────────────────────────────────
    def _fill(h): return _PFill("solid", fgColor=h)
    def _font(bold=False, color="333333", size=10, italic=False, name="Calibri"):
        return _Font(bold=bold, color=color, size=size, italic=italic, name=name)
    def _border(style="thin", color="D6DCE4"):
        s = _Side(border_style=style, color=color)
        return _Border(left=s, right=s, top=s, bottom=s)
    def _align(h="left", v="center", wrap=False):
        return _Align(horizontal=h, vertical=v, wrap_text=wrap)
    _C  = _align("center"); _L = _align("left"); _R = _align("right")
    _CW = _align("center", wrap=True); _LW = _align("left", wrap=True)

    def _hdr(ws, addr, text, bg=None, fc=_WHITE, sz=10):
        c = ws[addr]
        c.value = text; c.fill = _fill(bg or _NAVY)
        c.font = _font(True, fc, sz); c.alignment = _CW
        c.border = _border("thin", _GOLD_DK)
        return c

    def _meta_row(ws, rn, label, value):
        ws.row_dimensions[rn].height = 20
        ws.merge_cells(f"A{rn}:B{rn}")
        lc = ws[f"A{rn}"]
        lc.value = label; lc.font = _font(True, _NAVY_DK, 10)
        lc.fill = _fill(_GRAY_LT); lc.alignment = _L
        lc.border = _border("thin", _GRAY_MID)
        ws.merge_cells(f"C{rn}:H{rn}")
        vc = ws[f"C{rn}"]
        vc.value = value; vc.font = _font(False, _GRAY_TEXT, 10)
        vc.fill = _fill(_BLUE_LT); vc.alignment = _L
        vc.border = _border("thin", _GRAY_MID)

    generated_at = _dt.now().strftime("%d/%m/%Y %H:%M")
    logo_bytes   = _build_rssb_logo_bytes()

    wb = Workbook(); wb.remove(wb.active)

    # ════════════════════════════════════════════════════════════════════════════
    # SHEET 1 — Counter Verification Report
    # ════════════════════════════════════════════════════════════════════════════
    ws2 = wb.create_sheet("Counter Verification Report")
    ws2.sheet_view.showGridLines = False
    ws2.page_setup.orientation  = "portrait"
    ws2.page_setup.paperSize    = 9
    ws2.print_options.horizontalCentered = True

    for col, w in {"A":3,"B":26,"C":26,"D":22,"E":16,"F":16,"G":16,"H":4}.items():
        ws2.column_dimensions[col].width = w

    # Row 1 — Logo
    ws2.row_dimensions[1].height = 72
    ws2.merge_cells("A1:H1")
    ws2["A1"].fill = _fill(_WHITE)
    if logo_bytes:
        _li = _XLImage(_io.BytesIO(logo_bytes))
        _li.width = 232; _li.height = 64
        ws2.add_image(_li, "A1")
    else:
        ws2["A1"].value = "RSSB — Rwanda Social Security Board"
        ws2["A1"].font  = _font(True, _NAVY_DK, 16)
        ws2["A1"].alignment = _C

    # Row 2 — Gold accent bar
    ws2.row_dimensions[2].height = 5
    for col in "ABCDEFGH":
        ws2[f"{col}2"].fill = _fill(_GOLD)

    # Row 3 — Title banner
    ws2.row_dimensions[3].height = 34
    ws2.merge_cells("A3:H3")
    c = ws2["A3"]
    c.value = "REPORT OF COUNTER VERIFICATION — PHARMACY"
    c.fill = _fill(_NAVY); c.font = _font(True, _WHITE, 14)
    c.alignment = _CW;     c.border = _border("thin", _GOLD_DK)

    # Row 4 — Subtitle
    ws2.row_dimensions[4].height = 18
    ws2.merge_cells("A4:H4")
    c = ws2["A4"]
    c.value = (f"Rwanda Social Security Board  ·  Our Health, Our Future  ·  "
               f"Generated: {generated_at}  ·  www.rssb.rw")
    c.fill = _fill(_BLUE_LT); c.font = _font(False, _NAVY_DK, 9, italic=True)
    c.alignment = _C

    # Row 5 — spacer
    ws2.row_dimensions[5].height = 6

    # Rows 6–10 — Metadata
    _meta_row(ws2, 6,  "PROVINCE:",               meta.get("province",""))
    _meta_row(ws2, 7,  "ADMINISTRATIVE DISTRICT:", meta.get("district",""))
    _meta_row(ws2, 8,  "PHARMACY:",                meta.get("pharmacy",""))
    _meta_row(ws2, 9,  "PERIOD:",                  meta.get("period",""))
    _meta_row(ws2, 10, "CODE:",                    meta.get("code",""))

    # Row 11 — spacer
    ws2.row_dimensions[11].height = 8

    # Row 12 — Section header: Deduction Schedule
    ws2.row_dimensions[12].height = 22
    ws2.merge_cells("A12:H12")
    c = ws2["A12"]
    c.value = "DEDUCTION SCHEDULE"; c.fill = _fill(_NAVY_DK)
    c.font = _font(True, _WHITE, 11); c.alignment = _C

    # Row 13 — Table headers
    ws2.row_dimensions[13].height = 40
    for (start, end, text) in [
        ("A13","A13","No."),
        ("B13","B13","ID Invoice\n(Paper Code)"),
        ("C13","C13","Beneficiary's\nRAMA No."),
        ("D13","D13","Patient Name"),
        ("E13","F13","Amount Deducted\n(RWF)"),
        ("G13","H13","Explanation of Deduction"),
    ]:
        if start != end: ws2.merge_cells(f"{start}:{end}")
        _hdr(ws2, start, text)

    # Deduction data rows
    FIRST_DED = 14
    for i, ded in enumerate(deductions):
        rn = FIRST_DED + i
        bg = _WHITE if i % 2 == 0 else _GRAY_LT
        ws2.row_dimensions[rn].height = 18

        for addr, val, aln, fnt in [
            (f"A{rn}", i+1,                            _C,  _font(True, _NAVY_DK, 10)),
            (f"B{rn}", str(ded.get("voucher_id","")),  _C,  _font(False, _GRAY_TEXT, 10, name="Courier New")),
            (f"C{rn}", str(ded.get("rama","")),         _C,  _font(False, _GRAY_TEXT, 10, name="Courier New")),
            (f"D{rn}", str(ded.get("patient_name","")), _LW, _font(False, _GRAY_TEXT, 10)),
        ]:
            c = ws2[addr]; c.value = val; c.alignment = aln
            c.font = fnt; c.fill = _fill(bg); c.border = _border("thin", _GRAY_MID)

        ws2.merge_cells(f"E{rn}:F{rn}")
        c = ws2[f"E{rn}"]
        c.value = float(ded.get("difference",0) or 0)
        c.alignment = _R; c.fill = _fill(_RED_BG)
        c.font = _font(True, _RED, 10); c.border = _border("thin", _GRAY_MID)
        c.number_format = "#,##0"

        ws2.merge_cells(f"G{rn}:H{rn}")
        c = ws2[f"G{rn}"]
        c.value = str(ded.get("observation","")); c.alignment = _LW
        c.font = _font(False, _GRAY_TEXT, 10); c.fill = _fill(bg)
        c.border = _border("thin", _GRAY_MID)

    # Total row
    total_row = FIRST_DED + len(deductions)
    last_ded  = total_row - 1
    ws2.row_dimensions[total_row].height = 28

    ws2.merge_cells(f"A{total_row}:D{total_row}")
    c = ws2[f"A{total_row}"]
    c.value = "TOTAL AMOUNT DEDUCTED (RWF)"; c.fill = _fill(_NAVY)
    c.font = _font(True, _WHITE, 11); c.alignment = _align("right")
    c.border = _border("thin", _GOLD_DK)

    ws2.merge_cells(f"E{total_row}:F{total_row}")
    c = ws2[f"E{total_row}"]
    c.value = f"=SUM(E{FIRST_DED}:E{last_ded})"
    c.fill = _fill(_GOLD); c.font = _font(True, _NAVY_DK, 13)
    c.alignment = _R; c.border = _border("medium", _GOLD_DK)
    c.number_format = "#,##0"

    ws2.merge_cells(f"G{total_row}:H{total_row}")
    ws2[f"G{total_row}"].fill   = _fill(_NAVY)
    ws2[f"G{total_row}"].border = _border("thin", _GOLD_DK)

    # ── Summary stats ──────────────────────────────────────────────────────────
    stat_row = total_row + 2
    ws2.row_dimensions[total_row+1].height = 10
    ws2.row_dimensions[stat_row].height    = 22
    ws2.merge_cells(f"A{stat_row}:H{stat_row}")
    c = ws2[f"A{stat_row}"]
    c.value = "VERIFICATION SUMMARY"; c.fill = _fill(_NAVY)
    c.font = _font(True, _WHITE, 10); c.alignment = _C

    _total_rows  = len(df) if df is not None else 0
    _total_amt   = float(df[tot_col].sum()) if (df is not None and tot_col and tot_col in df.columns) else 0
    _total_ded   = sum(float(d.get("difference",0) or 0) for d in deductions)
    _ins_sum     = float(df[ins_col].sum()) if (df is not None and ins_col and ins_col in df.columns) else 0
    _net_85      = _ins_sum - _total_ded if _ins_sum else 0

    for si, (lbl, val, icon) in enumerate([
        ("Total Vouchers Reviewed",      str(_total_rows),             "📋"),
        ("Total Deductions Applied",     str(len(deductions)),          "⚠"),
        ("Gross Claims (100%)",          f"RWF {_total_amt:,.0f}",    "💰"),
        ("Total Amount Deducted",        f"RWF {_total_ded:,.0f}",    "➖"),
        ("Net Payable 85% (RSSB share)", f"RWF {_net_85:,.0f}",       "✅"),
    ]):
        rn = stat_row + 1 + si
        ws2.row_dimensions[rn].height = 20
        ws2.merge_cells(f"A{rn}:C{rn}")
        lc = ws2[f"A{rn}"]
        lc.value = f"{icon}  {lbl}"; lc.font = _font(True, _NAVY_DK, 10)
        lc.fill = _fill(_GRAY_LT); lc.alignment = _L; lc.border = _border("thin", _GRAY_MID)
        ws2.merge_cells(f"D{rn}:H{rn}")
        vc = ws2[f"D{rn}"]
        bg_v = _GREEN_BG if si == 4 else (_RED_BG if si == 3 else _BLUE_LT)
        col_v = _GREEN   if si == 4 else (_RED    if si == 3 else _NAVY_DK)
        vc.value = val; vc.font = _font(True, col_v, 10)
        vc.fill = _fill(bg_v); vc.alignment = _R; vc.border = _border("thin", _GRAY_MID)

    # ── Signature block ────────────────────────────────────────────────────────
    sig_row = stat_row + 8
    ws2.row_dimensions[sig_row - 1].height = 14

    for (sc, ec, title, name_val) in [
        ("B","C","Prepared by", prepared_by),
        ("D","E","Verified by", verified_by),
        ("F","G","Approved by", approved_by),
    ]:
        ws2.row_dimensions[sig_row].height = 20
        ws2.merge_cells(f"{sc}{sig_row}:{ec}{sig_row}")
        c = ws2[f"{sc}{sig_row}"]
        c.value = title; c.fill = _fill(_NAVY); c.font = _font(True, _WHITE, 10)
        c.alignment = _C; c.border = _border("thin", _GOLD_DK)

    for offset, lbl in enumerate(["Date:", "Signature:", "Names:"]):
        rn = sig_row + 1 + offset
        ws2.row_dimensions[rn].height = 22
        for sc, ec, nv in [("B","C",prepared_by),("D","E",verified_by),("F","G",approved_by)]:
            ws2.merge_cells(f"{sc}{rn}:{ec}{rn}")
            c = ws2[f"{sc}{rn}"]
            c.value = f"Names: {nv}" if lbl == "Names:" else lbl
            c.font  = _font(False, _GRAY_TEXT, 10)
            if lbl == "Signature:":
                c.fill = _fill(_WHITE)
                c.border = _Border(
                    bottom=_Side(border_style="medium", color=_NAVY),
                    left=_Side(border_style="thin", color=_GRAY_MID),
                    right=_Side(border_style="thin", color=_GRAY_MID),
                    top=_Side(border_style="thin", color=_GRAY_MID),
                )
            else:
                c.fill = _fill(_GRAY_LT); c.alignment = _L
                c.border = _border("thin", _GRAY_MID)

    # Gold bar + footer
    foot_bar = sig_row + 5
    ws2.row_dimensions[foot_bar].height = 5
    for col in "ABCDEFGH":
        ws2[f"{col}{foot_bar}"].fill = _fill(_GOLD)

    foot_txt = foot_bar + 1
    ws2.row_dimensions[foot_txt].height = 16
    ws2.merge_cells(f"A{foot_txt}:H{foot_txt}")
    c = ws2[f"A{foot_txt}"]
    c.value = (f"RSSB · Rwanda Social Security Board · "
               f"Medical Claims Counter-Verification · Confidential · {generated_at}")
    c.font = _font(False, _NAVY_DK, 8, italic=True)
    c.fill = _fill(_BLUE_LT); c.alignment = _C

    # ════════════════════════════════════════════════════════════════════════════
    # SHEET 2 — After Counter Verification (full records)
    # ════════════════════════════════════════════════════════════════════════════
    ws1 = wb.create_sheet("After Counter Verification")
    ws1.sheet_view.showGridLines = False
    ws1.freeze_panes = "A5"

    for col, w in {"A":14,"B":16,"C":30,"D":16,"E":28,
                   "F":26,"G":16,"H":8,"I":22,"J":22,
                   "K":22,"L":22,"M":20,"N":36}.items():
        ws1.column_dimensions[col].width = w

    # Row 1 — Logo
    ws1.row_dimensions[1].height = 72
    ws1.merge_cells("A1:N1")
    ws1["A1"].fill = _fill(_WHITE)
    if logo_bytes:
        _li2 = _XLImage(_io.BytesIO(logo_bytes))
        _li2.width = 232; _li2.height = 64
        ws1.add_image(_li2, "A1")

    # Row 2 — Gold bar
    ws1.row_dimensions[2].height = 5
    for col in [_gcl(i) for i in range(1,15)]:
        ws1[f"{col}2"].fill = _fill(_GOLD)

    # Row 3 — Title
    ws1.row_dimensions[3].height = 30
    ws1.merge_cells("A3:N3")
    c = ws1["A3"]
    c.value = (f"AFTER COUNTER VERIFICATION — PATIENT VOUCHER RECORDS  ·  "
               f"{meta.get('pharmacy','')}  ·  Period: {meta.get('period','')}")
    c.fill = _fill(_NAVY); c.font = _font(True, _WHITE, 11)
    c.alignment = _CW; c.border = _border("thin", _GOLD_DK)

    # Row 4 — Column headers
    ws1.row_dimensions[4].height = 42
    for ci, h in enumerate([
        "Paper Code","Dispensing Date","Patient Name","RAMA Number",
        "Practitioner Name","Health Facility","Date of\nTreatment","Verified",
        "Total Before\nCounter-V (RWF)","85% After\nCounter-V (RWF)",
        "After Counter-Ver\n100% (RWF)","After Counter-Ver\n85% (RWF)",
        "Amount\nDeducted (RWF)","Explanations",
    ], 1):
        c = ws1.cell(4, ci, h)
        c.fill = _fill(_NAVY); c.font = _font(True, _WHITE, 9)
        c.alignment = _CW; c.border = _border("thin", _GOLD_DK)

    ded_lookup = {str(d.get("voucher_id","")): d for d in deductions}

    def _safe(v): return "" if (v is None or str(v) in ("nan","NaT","None")) else v
    def _amt(row_data, key):
        v = row_data.get(key)
        try:   return float(v) if v and str(v) not in ("nan","None","NaT") else None
        except: return None

    for ri, row_data in enumerate(df.to_dict("records")):
        rn = 5 + ri
        ws1.row_dimensions[rn].height = 16

        pc_val  = str(_safe(row_data.get(pc_col or "voucher_id","")))
        ded     = ded_lookup.get(pc_val, {})
        ded_amt = float(ded.get("difference",0) or 0) if ded else 0
        obs_val = ded.get("observation","") if ded else str(_safe(row_data.get(obs_col or "","")))
        is_ded  = ded_amt > 0

        # Alternating: white / light gray per branding guide
        bg = _AMBER_BG if is_ded else (_WHITE if ri % 2 == 0 else _GRAY_LT)

        tot_val  = _amt(row_data, tot_col or "amount")
        ins_val  = _amt(row_data, ins_col or "insurance_copay")
        after100 = (tot_val  - ded_amt) if (tot_val  is not None) else None
        after85  = (after100 * 0.85)    if (after100 is not None) else None

        def _d(v):
            import datetime as _datetime
            if isinstance(v, (_datetime.datetime, _datetime.date)):
                return v.strftime("%d/%m/%Y")
            s = str(v or "")
            return "" if s in ("nan","NaT","None","") else s

        row_vals = [
            pc_val, _d(row_data.get("visit_date")),
            str(_safe(row_data.get("patient_name",""))),
            str(_safe(row_data.get("patient_id",""))),
            str(_safe(row_data.get("doctor_name",""))),
            str(_safe(row_data.get("facility", meta.get("pharmacy","")))),
            _d(row_data.get("visit_date")),
            "NO" if is_ded else "YES",
            tot_val, ins_val, after100, after85,
            ded_amt if is_ded else None, obs_val,
        ]
        aligns14  = [_C,_C,_LW,_C,_LW,_LW,_C,_C,_R,_R,_R,_R,_R,_LW]
        fmts14    = [None]*8 + ["#,##0"]*5 + [None]
        bolds14   = [False]*12 + [is_ded, False]
        colors14  = [_GRAY_TEXT]*12 + [_RED if is_ded else _GRAY_TEXT, _GRAY_TEXT]
        fn14      = ["Courier New","Calibri","Calibri","Courier New"] + ["Calibri"]*10

        for ci, (val, aln, fmt, bld, clr, fn_) in enumerate(
                zip(row_vals,aligns14,fmts14,bolds14,colors14,fn14), 1):
            c = ws1.cell(rn, ci, val)
            c.fill = _fill(bg); c.font = _font(bld, clr, 9, name=fn_)
            c.alignment = aln; c.border = _border("thin", _GRAY_MID)
            if fmt: c.number_format = fmt

    # Totals row
    tot_rn = 5 + len(df)
    ws1.row_dimensions[tot_rn].height = 24
    for ci in range(1,15):
        c = ws1.cell(tot_rn, ci)
        c.fill = _fill(_NAVY); c.border = _border("thin", _GOLD_DK)
    ws1.cell(tot_rn, 8).value     = "TOTALS"
    ws1.cell(tot_rn, 8).font      = _font(True, _WHITE, 10)
    ws1.cell(tot_rn, 8).alignment = _R

    for ci_off, col_l in enumerate("IJKLM", 9):
        c = ws1.cell(tot_rn, ci_off)
        c.value  = f"=SUM({col_l}5:{col_l}{tot_rn-1})"
        c.font   = _font(True, _GOLD if ci_off == 13 else _WHITE, 10)
        c.alignment = _R; c.number_format = "#,##0"

    # Status legend row
    leg_rn = tot_rn + 2
    ws1.row_dimensions[tot_rn+1].height = 8
    ws1.row_dimensions[leg_rn].height   = 18
    ws1.merge_cells(f"A{leg_rn}:N{leg_rn}")
    c = ws1[f"A{leg_rn}"]
    c.value = ("LEGEND:   🟡 Yellow row = Deduction applied    "
               "⬜/🔲 White/Gray = Verified (alternating)    "
               "Amount Deducted column shown in red")
    c.font = _font(False, _NAVY_DK, 9, italic=True)
    c.fill = _fill(_BLUE_LT); c.alignment = _L

    # Gold bar + footer
    foot_b = leg_rn + 1
    ws1.row_dimensions[foot_b].height = 5
    for col in [_gcl(i) for i in range(1,15)]:
        ws1[f"{col}{foot_b}"].fill = _fill(_GOLD)
    foot_t = foot_b + 1
    ws1.row_dimensions[foot_t].height = 14
    ws1.merge_cells(f"A{foot_t}:N{foot_t}")
    c = ws1[f"A{foot_t}"]
    c.value = (f"RSSB · Rwanda Social Security Board · "
               f"Medical Claims Counter-Verification · Confidential · {generated_at}")
    c.font = _font(False, _NAVY_DK, 8, italic=True)
    c.fill = _fill(_BLUE_LT); c.alignment = _C

    buf = _io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf.read()



with st.sidebar:
    st.markdown('<div class="sidebar-title">💊 PharmaScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Pharmacy Voucher Intelligence</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload voucher report",
        type=["csv", "xlsx", "xls", "ods"],
        help="Supports CSV, Excel (.xlsx/.xls), and ODS",
    )

    st.markdown("---")
    st.markdown("**⚙️ Analysis Settings**")
    rapid_days = st.slider("Rapid revisit window (days)", 1, 30, 7)
    top_n      = st.slider("Top N for charts", 5, 25, 15)
    show_raw   = st.checkbox("Show raw column names in tables", value=False)

    st.markdown("---")
    st.markdown("""**📋 Detected columns**
<small style='color:#64748b;font-family:monospace;line-height:2'>
<b style='color:#0ea5e9'>Patient</b><br>
RAMA Number → patient_id<br>
Patient Name → patient_name<br>
Patient Type · Gender · Is Newborn<br><br>
<b style='color:#0ea5e9'>Practitioner</b><br>
Practitioner Name → doctor_name<br>
Practitioner Type → doctor_type<br><br>
<b style='color:#0ea5e9'>Visit</b><br>
Dispensing Date → visit_date<br>
Paper Code → voucher_id<br><br>
<b style='color:#0ea5e9'>Financials</b><br>
Total Cost → amount<br>
Medicine Cost · Patient Co-payment<br>
Insurance Co-payment
</small>""", unsafe_allow_html=True)

# ── Landing page ──────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
<div style='text-align:center;padding:80px 20px 40px'>
  <div style='font-size:56px;margin-bottom:16px'>💊</div>
  <div style='font-family:Syne,sans-serif;font-size:36px;font-weight:800;color:#e2e8f0;margin-bottom:8px'>
    Pharma<span style='color:#00e5a0'>Scan</span></div>
  <div style='color:#64748b;font-size:16px;margin-bottom:32px'>Pharmacy Voucher Intelligence</div>
</div>""", unsafe_allow_html=True)
    for col, icon, title, desc in zip(
        st.columns(4),
        ["🔧", "🔁", "⚡", "🕸️"],
        ["Auto Column Fix", "Repeat Detection", "Rapid Revisits", "Network Graph"],
        [
            "Maps RAMA Number, Dispensing Date, Practitioner Name and 30+ variants automatically",
            "Finds patients with multiple vouchers and ranks them by frequency",
            "Flags same patient returning within your chosen day window",
            "Pick any two columns — patients, doctors, types — to explore as a network",
        ],
    ):
        with col:
            st.markdown(f"""
<div style='background:#111720;border:1px solid #1e2a38;border-radius:12px;padding:20px;
            text-align:center;min-height:150px'>
  <div style='font-size:28px;margin-bottom:8px'>{icon}</div>
  <div style='font-weight:700;color:#e2e8f0;margin-bottom:4px'>{title}</div>
  <div style='font-size:12px;color:#64748b'>{desc}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload a pharmacy voucher file in the sidebar to begin")
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("Analysing voucher data…"):
    try:
        df, col_map, s, repeat_groups, repeat_detail, rapid = load_and_process(
            file_bytes, uploaded.name, rapid_days
        )
    except Exception as e:
        st.error(f"❌ Could not process file: {e}")
        st.stop()

changed = {k: v for k, v in col_map.items() if k != v}
if changed:
    chips = "".join(f'<span class="chip">{k} → {v}</span>' for k, v in changed.items())
    st.markdown(f"""
<div class="info-banner">
  <b>🔧 Columns auto-normalised</b> — {len(changed)} name(s) mapped:
  <div class="chip-row">{chips}</div>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_summary, tab_records, tab_repeat, tab_rapid, tab_network, tab_norm, tab_cv, tab_xfac = st.tabs([
    "📊 Summary",
    "📋 All Records",
    f"🔁 Repeat Patients  {'🟡' if repeat_groups else '🟢'}  {len(repeat_groups)}",
    f"⚡ Rapid Revisits  {'🔴' if rapid else '🟢'}  {len(rapid)}",
    "🕸️ Network Graph",
    "✏️ Normalise Names",
    "📄 Counter-Verification Report",
    "🏥 Cross-Facility Match",
])

# ══ SUMMARY ══════════════════════════════════════════════════════════════════
with tab_summary:
    c = st.columns(4)
    c[0].metric("Total Records",         f"{s['total_rows']:,}")
    c[1].metric("Unique Patients",       f"{s['unique_patients']:,}" if "unique_patients" in s else "—")
    c[2].metric("Repeat Patients",       f"{s['repeat_patients']:,}" if "repeat_patients" in s else "—")
    c[3].metric("Rapid Revisits",        str(len(rapid)), delta=f"≤{rapid_days} day window")

    c2 = st.columns(4)
    c2[0].metric("Unique Practitioners", f"{s['unique_doctors']:,}" if "unique_doctors" in s else "—")
    c2[1].metric("Max Visits / Patient", str(s.get("max_visits", "—")))
    c2[2].metric("Total Cost (RWF)",     fmt_number(s["total_amount"]) if "total_amount" in s else "—")
    c2[3].metric("Avg Cost / Visit",     fmt_number(s["avg_amount"])   if "avg_amount"   in s else "—")

    if "date_min" in s:
        st.markdown(
            f'<p style="font-size:12px;color:#64748b;font-family:monospace;margin:8px 0 20px">'
            f'📅 {s["date_min"]} — {s["date_max"]}</p>',
            unsafe_allow_html=True,
        )

    fig_t = time_series_chart(df)
    if fig_t:
        st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

    left, right = st.columns(2)
    with left:
        if "top_patients" in s:
            td = s["top_patients"].head(top_n)
            colors = [DANGER if v >= 10 else WARN if v >= 5 else ACCENT for v in td["visits"]]
            fig = hbar_chart([str(x)[:22] for x in td["id"]], td["visits"].tolist(),
                             colors, "Top Patients (RAMA No.) by Visit Count", "Visits")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
    with right:
        if "top_doctors" in s:
            td = s["top_doctors"].head(top_n)
            fig = hbar_chart([str(x)[:22] for x in td["doctor"]], td["visits"].tolist(),
                             ACCENT2, "Top Practitioners by Visit Volume", "Visits")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Practitioner type breakdown
    if "doctor_type" in df.columns:
        dt_vc = df["doctor_type"].value_counts().head(top_n)
        fig = hbar_chart(
            [str(x)[:30] for x in dt_vc.index],
            dt_vc.values.tolist(),
            PURPLE, "Visits by Practitioner Type", "Visits",
        )
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Gender & patient type pie charts
    gc1, gc2 = st.columns(2)
    for target_col, label, target_col_obj in [
        (gc1, "Gender Breakdown",       "gender"),
        (gc2, "Patient Type Breakdown", "patient_type"),
    ]:
        if target_col_obj in df.columns:
            vc = df[target_col_obj].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.pie(vc.values, labels=vc.index,
                   colors=[ACCENT, ACCENT2, PURPLE, WARN, DANGER][:len(vc)],
                   autopct="%1.1f%%", pctdistance=0.8,
                   textprops={"color": TEXT, "fontsize": 9},
                   wedgeprops={"linewidth": 1.5, "edgecolor": CARD})
            ax.set_title(label, fontsize=11, fontweight="bold", color=TEXT, pad=10)
            with target_col:
                st.pyplot(fig, use_container_width=True); plt.close(fig)

# ══ ALL RECORDS ═══════════════════════════════════════════════════════════════
with tab_records:
    st.markdown(f'<div class="sec-head">All Records — {len(df):,} rows</div>', unsafe_allow_html=True)
    search = st.text_input("🔍 Filter rows", key="rec_search", placeholder="Type to search any column…")
    display_df = df.copy()
    if not show_raw:
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    if search:
        mask = display_df.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]
        st.caption(f"{len(display_df):,} matching rows")
    st.dataframe(display_df, use_container_width=True, height=520)

# ══ REPEAT PATIENTS ══════════════════════════════════════════════════════════
with tab_repeat:
    if not repeat_groups:
        st.success("✅ No patients with multiple visits detected.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Repeat Patients", len(repeat_groups))
        c2.metric("Max Visits",      s.get("max_visits", "—"))
        heavy = sum(1 for g in repeat_groups if g["visits"] >= 5)
        c3.metric("High-frequency (≥5)", heavy)

        visit_counts = [g["visits"] for g in repeat_groups]
        if len(set(visit_counts)) > 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            bins = list(range(2, max(visit_counts) + 2))
            _, bins_out, patches = ax.hist(visit_counts, bins=bins,
                                           color=ACCENT, edgecolor=CARD, rwidth=0.8)
            for patch, left in zip(patches, bins_out[:-1]):
                if left >= 10: patch.set_facecolor(DANGER)
                elif left >= 5: patch.set_facecolor(WARN)
            ax.set_xlabel("Visits per Patient"); ax.set_ylabel("Patients")
            ax.set_title("Distribution of Repeat Visit Counts", fontsize=11,
                         fontweight="bold", color=TEXT, pad=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3); fig.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown('<div class="sec-head">Patient Visit Groups</div>', unsafe_allow_html=True)
        grp_df = pd.DataFrame(repeat_groups)
        srch = st.text_input("🔍 Filter", key="rep_search", placeholder="Name or RAMA number…")
        if srch:
            mask = grp_df.apply(lambda c: c.astype(str).str.contains(srch, case=False, na=False)).any(axis=1)
            grp_df = grp_df[mask]

        def highlight_v(val):
            try:
                v = int(val)
                if v >= 10: return "color:#ef4444;font-weight:bold"
                if v >= 5:  return "color:#f59e0b;font-weight:bold"
            except Exception:
                pass
            return ""

        st.dataframe(grp_df.style.applymap(highlight_v, subset=["visits"]),
                     use_container_width=True, height=360)

        st.markdown('<div class="sec-head">Detailed Repeat Visit Records</div>', unsafe_allow_html=True)
        st.dataframe(repeat_detail, use_container_width=True, height=380)

# ══ RAPID REVISITS ════════════════════════════════════════════════════════════
with tab_rapid:
    if not rapid:
        st.success(f"✅ No rapid revisits detected within {rapid_days} days.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rapid Cases", len(rapid))
        critical = sum(1 for r in rapid if r["days_apart"] <= 2)
        c2.metric("Critical (≤2 days)", critical)
        avg_d = sum(r["days_apart"] for r in rapid) / len(rapid)
        c3.metric("Avg Days Apart", f"{avg_d:.1f}")

        fig_h = rapid_histogram(rapid)
        if fig_h:
            st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

        st.markdown(f'<div class="sec-head">⚠️ {len(rapid)} cases — window ≤{rapid_days} days</div>',
                    unsafe_allow_html=True)
        for i in range(0, len(rapid), 3):
            batch = rapid[i:i + 3]
            cols = st.columns(len(batch))
            for col, r in zip(cols, batch):
                crit = r["days_apart"] <= 2
                col.markdown(f"""
<div class="rapid-card {'crit' if crit else ''}">
  <div class="rc-head">
    <div>
      <div class="rc-name">{r['patient_name'][:24]}</div>
      <div class="rc-id">{r['patient_id']}</div>
    </div>
    <div class="rc-days">{r['days_apart']}<small> d</small></div>
  </div>
  <div class="rc-meta">📅 {r['visit_1']} → {r['visit_2']}</div>
  <div class="rc-meta">👨‍⚕️ {r['doctor'][:28]}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:24px">Full Table</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rapid), use_container_width=True, height=360)

# ══ NETWORK GRAPH ════════════════════════════════════════════════════════════
with tab_network:

    # All text / categorical columns available for network nodes
    cat_cols = [c for c in df.columns
                if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not cat_cols:
        cat_cols = list(df.columns)

    def col_idx(candidates):
        for name in candidates:
            if name in cat_cols:
                return cat_cols.index(name)
        return 0

    st.markdown('<div class="sec-head">🔧 Network Configuration</div>', unsafe_allow_html=True)

    cfg1, cfg2, cfg3, cfg4, cfg5 = st.columns([2, 2, 1.8, 1.4, 1.4])

    with cfg1:
        col_a = st.selectbox(
            "◆ Node A (diamonds)",
            options=cat_cols,
            index=col_idx(["doctor_name", "doctor_type", "doctor_id"]),
            help="Each unique value in this column becomes a diamond-shaped node",
        )
    with cfg2:
        b_default = col_idx(["patient_name", "patient_id", "gender", "patient_type"])
        if cat_cols[b_default] == col_a and len(cat_cols) > 1:
            b_default = (b_default + 1) % len(cat_cols)
        col_b = st.selectbox(
            "● Node B (circles)",
            options=cat_cols,
            index=b_default,
            help="Each unique value in this column becomes a circle-shaped node",
        )
    with cfg3:
        physics_mode = st.selectbox(
            "Physics / layout",
            options=["Force Atlas 2", "Barnes-Hut", "Repulsion", "None (static)"],
            help="Force simulation used to position nodes",
        )
    with cfg4:
        max_nodes = st.slider("Max nodes", 20, 400, 150)
    with cfg5:
        min_edge  = st.slider("Min edge weight", 1, 10, 1,
                              help="Hide edges with fewer shared visits than this")

    net_height = st.slider("Canvas height (px)", 400, 900, 680, step=40)

    # Quick preset combos
    st.markdown("""
<div style='font-size:11px;color:#64748b;margin-bottom:14px;font-family:monospace'>
  💡 Try: <b style='color:#e2e8f0'>Practitioner Name ↔ Patient Name</b> &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Practitioner Type ↔ Patient Type</b> &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Practitioner Name ↔ Gender</b> &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Practitioner Name ↔ Patient Type</b>
</div>""", unsafe_allow_html=True)

    if col_a == col_b:
        st.warning("⚠️ Node A and Node B must be different columns.")
    else:
        card_a = df[col_a].nunique()
        card_b = df[col_b].nunique()
        st.markdown(
            f'<p style="font-size:12px;color:#64748b;font-family:monospace;margin-bottom:16px">'
            f'◆ <b style="color:#00e5a0">{col_a.replace("_"," ").title()}</b>: {card_a} unique values'
            f'&nbsp;|&nbsp;'
            f'● <b style="color:#0ea5e9">{col_b.replace("_"," ").title()}</b>: {card_b} unique values'
            f'&nbsp;|&nbsp; {len(df):,} rows total</p>',
            unsafe_allow_html=True,
        )

        with st.spinner("Building network…"):
            vis_nodes, vis_edges, stats = build_network_data(
                df, col_a, col_b, max_nodes, min_edge
            )

        if vis_nodes is None:
            st.warning("No edges found. Try lowering the minimum edge weight or choosing different columns.")
        else:
            # Stats row
            mc = st.columns(5)
            mc[0].metric(f"◆ {col_a.replace('_',' ').title()[:14]}", stats["nodes_a"])
            mc[1].metric(f"● {col_b.replace('_',' ').title()[:14]}", stats["nodes_b"])
            mc[2].metric("Edges",      stats["edges"])
            mc[3].metric("Avg Degree", stats["avg_degree"])
            mc[4].metric("Density",    stats["density"])

            # Controls hint
            st.markdown("""
<div style='font-size:11px;color:#64748b;margin:8px 0 6px;font-family:monospace'>
  🖱 <b style='color:#e2e8f0'>Drag</b> nodes to reposition &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Scroll</b> to zoom &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Click</b> a node to highlight its neighbours &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Double-click</b> to zoom in &nbsp;·&nbsp;
  <b style='color:#e2e8f0'>Search</b> any node by name
</div>""", unsafe_allow_html=True)

            # ── Render interactive vis.js network ──
            render_vis_network(vis_nodes, vis_edges, stats, physics_mode, height=net_height)

            # Top connected nodes tables below the canvas
            st.markdown("<br>", unsafe_allow_html=True)
            if stats["top_a"] or stats["top_b"]:
                ta, tb = st.columns(2)
                with ta:
                    st.markdown(
                        f'<div class="sec-head">◆ Top {col_a.replace("_"," ").title()} by connections</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(stats["top_a"], columns=["node", "connections"])
                          .style.background_gradient(subset=["connections"], cmap="Greens"),
                        use_container_width=True, height=320,
                    )
                with tb:
                    st.markdown(
                        f'<div class="sec-head">● Top {col_b.replace("_"," ").title()} by connections</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(stats["top_b"], columns=["node", "connections"])
                          .style.background_gradient(subset=["connections"], cmap="Blues"),
                        use_container_width=True, height=320,
                    )

# ══ NORMALISE NAMES TAB ══════════════════════════════════════════════════════
with tab_norm:

    st.markdown('<div class="sec-head">✏️ Doctor / Practitioner Name Normalisation</div>',
                unsafe_allow_html=True)

    # Pick which column to normalise
    str_cols = [c for c in df.columns
                if df[c].dtype == object or str(df[c].dtype).startswith("string")]

    norm_col = st.selectbox(
        "Column to normalise",
        options=str_cols,
        index=str_cols.index("doctor_name") if "doctor_name" in str_cols else 0,
        help="Fuzzy-match similar names within this column and merge variants into one canonical form",
    )

    with st.spinner("Detecting name clusters…"):
        col_counts = df[norm_col].value_counts().to_dict()
        col_names  = sorted(df[norm_col].dropna().unique().tolist(), key=str)
        clusters   = detect_name_clusters(col_names, col_counts)

    n_clusters   = len(clusters)
    n_suspicious = sum(1 for c in clusters if c["suspicious"])
    n_variants   = sum(len(c["variants"]) for c in clusters)

    # ── Summary metrics ──
    mc = st.columns(4)
    mc[0].metric("Unique raw names",   len(col_names))
    mc[1].metric("Variant clusters",   n_clusters)
    mc[2].metric("Total variants",     n_variants,
                 delta=f"→ {len(col_names) - n_variants} after merge")
    mc[3].metric("⚠️ Needs review",    n_suspicious)

    st.markdown("""
<div style='font-size:11px;color:#64748b;margin:8px 0 16px;font-family:monospace'>
  Below are proposed name merges. <b style='color:#e2e8f0'>✅ Check</b> clusters to approve them,
  then click <b style='color:#00e5a0'>Apply Selected</b>. Suspicious clusters (no shared tokens)
  are shown in amber — review carefully before approving.
</div>""", unsafe_allow_html=True)

    if not clusters:
        st.success("✅ No similar names detected — the column looks clean.")
    else:
        # ── Quick select helpers ──
        qc1, qc2, qc3 = st.columns([1, 1, 4])
        with qc1:
            if st.button("✅ Select all"):
                for c in clusters:
                    st.session_state[f"norm_{c['canonical']}"] = True
        with qc2:
            if st.button("❌ Deselect all"):
                for c in clusters:
                    st.session_state[f"norm_{c['canonical']}"] = False

        # ── Cluster review cards ──
        approved = []
        for cluster in clusters:
            key      = f"norm_{cluster['canonical']}"
            default  = not cluster["suspicious"]    # auto-select non-suspicious
            checked  = st.session_state.get(key, default)

            conf_pct = int(cluster["confidence"] * 100)
            if cluster["suspicious"]:
                border = "#f59e0b"
                conf_badge = f'<span style="background:rgba(245,158,11,.15);color:#f59e0b;border-radius:5px;padding:2px 8px;font-size:10px">⚠️ {conf_pct}% — review</span>'
            elif conf_pct >= 90:
                border = "#00e5a0"
                conf_badge = f'<span style="background:rgba(0,229,160,.1);color:#00e5a0;border-radius:5px;padding:2px 8px;font-size:10px">✓ {conf_pct}%</span>'
            else:
                border = "#0ea5e9"
                conf_badge = f'<span style="background:rgba(14,165,233,.1);color:#0ea5e9;border-radius:5px;padding:2px 8px;font-size:10px">{conf_pct}%</span>'

            # Variant pills
            freq_total = sum(col_counts.get(v, 0) for v in cluster["variants"])
            canon_freq = col_counts.get(cluster["canonical"], 0)
            variant_pills = " ".join(
                f'<span style="background:#1e2a38;border-radius:5px;padding:2px 8px;font-size:11px;'
                f'font-family:monospace;color:#94a3b8;margin:2px">'
                f'{v} <span style="color:#64748b">×{col_counts.get(v,0)}</span></span>'
                for v in cluster["variants"]
            )

            st.markdown(f"""
<div style="background:#111720;border:1px solid {border};border-left:3px solid {border};
     border-radius:10px;padding:14px 16px;margin-bottom:10px">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap">
    <div style="flex:1;min-width:200px">
      <div style="font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:4px">
        → <span style="color:#00e5a0">{cluster['canonical']}</span>
        <span style="color:#64748b;font-size:11px;margin-left:6px">×{canon_freq} occurrences</span>
      </div>
      <div style="font-size:11px;color:#64748b;margin-bottom:8px;font-family:monospace">
        {len(cluster['variants'])} variant(s) · {freq_total} rows affected
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:4px">{variant_pills}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex-shrink:0">
      {conf_badge}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            checked = st.checkbox(
                f"Approve merge → {cluster['canonical']}",
                value=checked,
                key=key,
            )
            if checked:
                approved.append(cluster)

        # ── Apply button ──
        st.markdown("<br>", unsafe_allow_html=True)
        col_apply, col_preview = st.columns([1, 2])

        with col_apply:
            st.markdown(
                f'<p style="font-size:12px;color:#64748b;font-family:monospace">'
                f'{len(approved)} cluster(s) selected · '
                f'{sum(len(c["variants"]) for c in approved)} variant(s) will be merged</p>',
                unsafe_allow_html=True,
            )
            do_apply = st.button("⚡ Apply Selected Normalisations",
                                 type="primary", disabled=len(approved) == 0)

        if do_apply:
            st.session_state["normalised_df"]  = apply_name_normalisation(df, norm_col, approved)
            st.session_state["normalised_col"] = norm_col
            st.session_state["normalised_map"] = {
                v: c["canonical"]
                for c in approved for v in c["variants"]
            }

        # ── Show result ──
        if "normalised_df" in st.session_state and st.session_state.get("normalised_col") == norm_col:
            ndf = st.session_state["normalised_df"]
            nmap = st.session_state["normalised_map"]

            st.success(f"✅ Applied! {len(nmap)} variant names merged in column **{norm_col}**.")

            before_unique = df[norm_col].nunique()
            after_unique  = ndf[norm_col].nunique()

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Before: unique names", before_unique)
            rc2.metric("After: unique names",  after_unique,
                       delta=f"−{before_unique - after_unique}")
            rc3.metric("Rows updated", len(nmap))

            # Rename map table
            st.markdown('<div class="sec-head">Applied Rename Map</div>', unsafe_allow_html=True)
            map_df = pd.DataFrame(
                [(k, v) for k, v in sorted(nmap.items())],
                columns=["Original variant", "→ Canonical name"]
            )
            st.dataframe(map_df, use_container_width=True, height=300)

            # Updated data preview
            st.markdown('<div class="sec-head">Updated Data Preview</div>', unsafe_allow_html=True)
            srch_n = st.text_input("🔍 Filter", key="norm_srch", placeholder="Search any column…")
            show_df = ndf.copy()
            if not show_raw:
                show_df.columns = [c.replace("_", " ").title() for c in show_df.columns]
            if srch_n:
                mask = show_df.apply(lambda c: c.astype(str).str.contains(srch_n, case=False, na=False)).any(axis=1)
                show_df = show_df[mask]
            st.dataframe(show_df, use_container_width=True, height=480)

            # Download
            csv_bytes = ndf.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download normalised CSV",
                data=csv_bytes,
                file_name=f"pharmascan_normalised_{norm_col}.csv",
                mime="text/csv",
            )

# ══ COUNTER-VERIFICATION REPORT TAB ══════════════════════════════════════════
# ══ COUNTER-VERIFICATION REPORT TAB ══════════════════════════════════════════
with tab_cv:
    st.markdown('<div class="sec-head">📄 Counter-Verification Report Generator</div>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="info-banner">
  <b>How it works:</b><br>
  Upload the annotated voucher file — the same invoice report with two extra columns
  added by the verifier: <b style='color:#00e5a0'>Difference</b> (deduction amount in RWF)
  and <b style='color:#00e5a0'>Observation</b> (reason for deduction). Rows without a
  deduction are simply left blank in those columns.<br><br>
  The app reads those columns, matches records by Paper Code, and generates the
  official two-sheet counter-verification Excel report automatically.
</div>""", unsafe_allow_html=True)

    # ── STEP 1 — Upload annotated file ────────────────────────────────────
    st.markdown('<div class="sec-head">📂 Step 1 — Upload Annotated Voucher File</div>',
                unsafe_allow_html=True)

    cv_upload = st.file_uploader(
        "Upload annotated file (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        key="cv_upload",
        help="Must be the invoice report with Difference and Observation columns filled in for deducted rows.",
    )

    # ── Parse uploaded file ────────────────────────────────────────────────
    ann_df = None
    deduction_list = []

    if cv_upload is not None:
        try:
            raw_bytes = cv_upload.read()
            fname = cv_upload.name.lower()
            if fname.endswith(".csv"):
                raw_ann = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8",
                                      on_bad_lines="skip")
            else:
                # Try each sheet — prefer one named Sheet1, or the one with most rows
                xl_ann = pd.ExcelFile(io.BytesIO(raw_bytes))
                sheet_candidates = []
                for sn in xl_ann.sheet_names:
                    try:
                        tmp = xl_ann.parse(sn, header=0)
                        sheet_candidates.append((sn, tmp))
                    except Exception:
                        pass

                # Prefer sheet whose columns contain "difference" or "observation"
                def score_sheet(df_):
                    cols = " ".join(df_.columns.astype(str).str.lower())
                    return ("diff" in cols or "observ" in cols or "remark" in cols)

                preferred = next(
                    ((sn, d) for sn, d in sheet_candidates if score_sheet(d)),
                    sheet_candidates[0] if sheet_candidates else (None, None)
                )
                chosen_sheet, raw_ann = preferred

            # ── Auto-detect columns ──────────────────────────────────────
            def _normalise_col(c):
                import re as _re
                return _re.sub(r"[^a-z0-9]", "_",
                               str(c).lower().strip()).strip("_")

            col_norm = {_normalise_col(c): c for c in raw_ann.columns}

            def _find_col(patterns):
                for pat in patterns:
                    for norm, orig in col_norm.items():
                        if re.search(pat, norm):
                            return orig
                return None

            detected = {
                "paper_code":  _find_col([r"paper_?code", r"voucher_?no", r"invoice_?no",
                                          r"invoice_?id", r"id_?invoice", r"claim_?no",
                                          r"ref_?no", r"receipt_?no", r"doc_?no"]),
                "rama":        _find_col([r"rama", r"affil", r"member", r"benef"]),
                "patient":     _find_col([r"patient_?name", r"beneficiary_?name",
                                          r"client_?name", r"nom"]),
                "difference":  _find_col([r"diff", r"deduct", r"deduit", r"montant_ded",
                                          r"amount_ded", r"ded_amount"]),
                "observation": _find_col([r"observ", r"remark", r"reason", r"explan",
                                          r"comment", r"note", r"justif", r"motif"]),
                "ins_copay":   _find_col([r"insurance_co", r"ins_cop", r"couverture",
                                          r"rama_amount"]),
                "total_cost":  _find_col([r"total_cost", r"total", r"amount", r"cout"]),
                "visit_date":  _find_col([r"dispensing", r"visit_date", r"date"]),
                "doctor":      _find_col([r"practitioner", r"prescriber", r"doctor",
                                          r"medecin"]),
            }

            st.session_state["ann_df"]       = raw_ann
            st.session_state["ann_detected"] = detected

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            import traceback; st.code(traceback.format_exc())

    # Load from session state if already uploaded
    if "ann_df" in st.session_state:
        raw_ann  = st.session_state["ann_df"]
        detected = st.session_state["ann_detected"]

        # ── STEP 2 — Confirm / override column mapping ─────────────────────
        st.markdown('<div class="sec-head">🔗 Step 2 — Confirm Column Mapping</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div style='font-size:11px;color:#64748b;margin-bottom:10px;font-family:monospace'>
  Columns detected automatically. Correct any mismatches — only
  <b style='color:#e2e8f0'>Paper Code</b>, <b style='color:#e2e8f0'>Difference</b>
  and <b style='color:#e2e8f0'>Observation</b> are required.
</div>""", unsafe_allow_html=True)

        col_opts = ["(none)"] + raw_ann.columns.tolist()
        def _sel_idx(col_name):
            return col_opts.index(col_name) if col_name in col_opts else 0

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            sel_pc  = st.selectbox("📄 Paper Code",
                                   col_opts, index=_sel_idx(detected["paper_code"]),
                                   key="cv_sel_pc")
            sel_obs = st.selectbox("💬 Observation / Reason",
                                   col_opts, index=_sel_idx(detected["observation"]),
                                   key="cv_sel_obs")
        with cc2:
            sel_dif = st.selectbox("💰 Difference (amount deducted)",
                                   col_opts, index=_sel_idx(detected["difference"]),
                                   key="cv_sel_dif")
            sel_ins = st.selectbox("🏥 Insurance Co-payment",
                                   col_opts, index=_sel_idx(detected["ins_copay"]),
                                   key="cv_sel_ins")
        with cc3:
            sel_tot = st.selectbox("💵 Total Cost",
                                   col_opts, index=_sel_idx(detected["total_cost"]),
                                   key="cv_sel_tot")
            sel_pat = st.selectbox("👤 Patient Name",
                                   col_opts, index=_sel_idx(detected["patient"]),
                                   key="cv_sel_pat")

        # Preview
        with st.expander("🔍 Preview uploaded file", expanded=False):
            st.dataframe(raw_ann.head(30), use_container_width=True, height=280)

        # ── Build deduction list ─────────────────────────────────────────
        if sel_pc != "(none)" and sel_dif != "(none)":

            def _to_float(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return 0.0
                try:
                    return float(str(v).replace(",", "").replace(" ", ""))
                except ValueError:
                    return 0.0

            ann_work = raw_ann.copy()
            ann_work["_pc"]  = ann_work[sel_pc].astype(str).str.strip()
            ann_work["_dif"] = ann_work[sel_dif].apply(_to_float)
            ann_work["_obs"] = (ann_work[sel_obs].fillna("").astype(str).str.strip()
                                if sel_obs != "(none)" else "")
            ann_work["_ins"] = (ann_work[sel_ins].apply(_to_float)
                                if sel_ins != "(none)" else 0.0)
            ann_work["_tot"] = (ann_work[sel_tot].apply(_to_float)
                                if sel_tot != "(none)" else 0.0)
            ann_work["_pat"] = (ann_work[sel_pat].fillna("").astype(str).str.strip()
                                if sel_pat != "(none)" else "")

            deducted_rows = ann_work[ann_work["_dif"] > 0].copy()

            # Pull RAMA from main loaded df if possible
            rama_lookup = {}
            vid_main = "voucher_id" if "voucher_id" in df.columns else None
            pid_main = "patient_id" if "patient_id" in df.columns else None
            if vid_main and pid_main:
                for _, r in df[[vid_main, pid_main]].iterrows():
                    rama_lookup[str(r[vid_main]).strip()] = str(r[pid_main]).strip()

            deduction_list = []
            for _, r in deducted_rows.iterrows():
                pc   = r["_pc"]
                rama = rama_lookup.get(pc, "—")
                deduction_list.append({
                    "paper_code":  pc,
                    "rama_no":     rama,
                    "patient":     r["_pat"],
                    "amount":      r["_dif"],
                    "explanation": r["_obs"],
                    "ins_copay":   r["_ins"],
                    "total_cost":  r["_tot"],
                })

            # ── STEP 3 — Deduction summary ─────────────────────────────
            st.markdown('<div class="sec-head">📊 Step 3 — Deduction Summary</div>',
                        unsafe_allow_html=True)

            total_ded    = sum(d["amount"]   for d in deduction_list)
            total_ins    = ann_work["_ins"].sum() if sel_ins != "(none)" else 0
            net_payable  = total_ins - total_ded

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Rows with deductions",    f"{len(deduction_list):,}")
            sm2.metric("Total deducted (RWF)",    f"{total_ded:,.0f}")
            sm3.metric("Total insurance claims",  f"{total_ins:,.0f}")
            sm4.metric("Net payable (RWF)",       f"{net_payable:,.0f}")

            # Observation breakdown
            obs_counts = {}
            for d in deduction_list:
                key = d["explanation"].strip().lower()
                obs_counts[key] = obs_counts.get(key, 0) + 1
            if obs_counts:
                st.markdown("""
<div style='font-size:11px;color:#64748b;font-family:monospace;margin:8px 0 4px'>
  <b style='color:#e2e8f0'>Deduction reasons breakdown:</b>
</div>""", unsafe_allow_html=True)
                reason_cols = st.columns(min(len(obs_counts), 4))
                for i, (reason, cnt) in enumerate(
                        sorted(obs_counts.items(), key=lambda x: -x[1])):
                    reason_cols[i % len(reason_cols)].metric(
                        reason[:40] or "(blank)", cnt)

            # Deduction table
            ded_tbl = pd.DataFrame([{
                "#":             i + 1,
                "Paper Code":    d["paper_code"],
                "Patient":       d["patient"],
                "RAMA No.":      d["rama_no"],
                "Ins. Co-pay":   d["ins_copay"],
                "Deducted (RWF)":d["amount"],
                "Observation":   d["explanation"],
            } for i, d in enumerate(deduction_list)])
            st.dataframe(ded_tbl, use_container_width=True, height=320)

        else:
            st.info("👆 Assign at least the **Paper Code** and **Difference** columns above to continue.")

        # ── STEP 4 — Report metadata ───────────────────────────────────────
        if deduction_list:
            st.markdown('<div class="sec-head">📋 Step 4 — Report Metadata</div>',
                        unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                cv_province = st.text_input("Province",                value="WESTERN PROVINCE", key="cv_prov")
                cv_district = st.text_input("Administrative District", value="RUBAVU",            key="cv_dist")
                cv_pharmacy = st.text_input("Pharmacy",
                    value="PHARMACIE VINCA GISENYI LTD", key="cv_pharm")
            with mc2:
                if "date_min" in s and "date_max" in s:
                    default_period = f"{s['date_min']} to {s['date_max']}"
                else:
                    default_period = ""
                cv_period  = st.text_input("Period",          value=default_period, key="cv_per")
                cv_code    = st.text_input("Code",            value="",             key="cv_code")
                cv_prep    = st.text_input("Prepared by",     value="",             key="cv_prep")

            sc1, sc2 = st.columns(2)
            with sc1:
                cv_verified = st.text_input("Verified by",
                    value="Alphonsine MUKAKAYIBANDA", key="cv_verif")
            with sc2:
                cv_approved = st.text_input("Approved by", value="", key="cv_approv")

            # ── STEP 5 — Generate ──────────────────────────────────────────
            st.markdown('<div class="sec-head">⬇️ Step 5 — Generate Report</div>',
                        unsafe_allow_html=True)

            if st.button("📊 Generate Counter-Verification Excel Report",
                         type="primary", key="cv_gen_btn"):
                meta = {
                    "province": cv_province,
                    "district": cv_district,
                    "pharmacy": cv_pharmacy,
                    "period":   cv_period,
                    "code":     cv_code,
                }

                # Merge annotated file into df for Sheet1 (use uploaded file rows)
                # Build a combined df from ann_work so we have Difference + Observation
                ann_for_report = ann_work.copy()

                with st.spinner("Building Excel report…"):
                    try:
                        xlsx_bytes = generate_counter_verification_xlsx(
                            df          = ann_for_report,
                            deductions  = deduction_list,
                            meta        = meta,
                            prepared_by = cv_prep,
                            verified_by = cv_verified,
                            approved_by = cv_approved,
                            pc_col      = sel_pc,
                            ins_col     = sel_ins if sel_ins != "(none)" else None,
                            tot_col     = sel_tot if sel_tot != "(none)" else None,
                            obs_col     = sel_obs if sel_obs != "(none)" else None,
                            dif_col     = sel_dif,
                        )
                        st.session_state["cv_xlsx"]      = xlsx_bytes
                        st.session_state["cv_generated"] = True
                        st.session_state["cv_fname_meta"] = (cv_pharmacy, cv_period)
                    except Exception as e:
                        st.error(f"❌ Failed to generate report: {e}")
                        import traceback; st.code(traceback.format_exc())

            if st.session_state.get("cv_generated"):
                pharm_s, period_s = st.session_state.get("cv_fname_meta", ("pharmacy", "period"))
                fname = (f"counter_verification_"
                         f"{pharm_s.replace(' ','_')[:25]}_"
                         f"{period_s.replace(' ','_').replace('/','')[:15]}.xlsx")
                st.success("✅ Report ready!")
                st.download_button(
                    label     = "⬇️ Download Counter-Verification Report (.xlsx)",
                    data      = st.session_state["cv_xlsx"],
                    file_name = fname,
                    mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key       = "cv_download",
                )
                st.markdown("""
<div style='font-size:12px;color:#64748b;font-family:monospace;margin-top:8px'>
  <b style='color:#e2e8f0'>Sheet 1</b> — "After counter verification":
  full records table with 100% and 85% columns; deducted rows highlighted in amber.<br>
  <b style='color:#e2e8f0'>Sheet 2</b> — "Counter verification report":
  official summary with deduction list, totals, and signature block.
</div>""", unsafe_allow_html=True)

    else:
        st.info("👆 Upload your annotated voucher file to get started.")

# ══ FRAUD DETECTION TAB ═══════════════════════════════════════════════════════
with tab_xfac:
    import math
    from difflib import SequenceMatcher as _SQ

    st.markdown("""
<style>
.fd-card{background:#111720;border:1px solid #1e2a38;border-radius:12px;
         padding:18px 22px;margin-bottom:14px}
.fd-red   {background:#1a0505;border-color:#7f1d1d}
.fd-amber {background:#1a1000;border-color:#78350f}
.fd-green {background:#031a0a;border-color:#14532d}
.fd-blue  {background:#020e1a;border-color:#1e3a5f}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;
       font-size:11px;font-weight:700;font-family:monospace}
.badge-red   {background:#7f1d1d;color:#fca5a5}
.badge-amber {background:#78350f;color:#fde68a}
.badge-green {background:#14532d;color:#86efac}
.badge-blue  {background:#1e3a5f;color:#93c5fd}
</style>""", unsafe_allow_html=True)

    st.markdown("""
<div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;
     color:#e2e8f0;margin-bottom:4px'>🔍 Pharmacy Fraud Detection</div>
<div style='color:#64748b;font-size:13px;font-family:monospace;margin-bottom:20px'>
  Detects pharmacy RSSB claims where medicine was dispensed with
  <b style='color:#ef4444'>no verifiable hospital or clinic visit record</b>.
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class='fd-card fd-blue' style='margin-bottom:22px'>
<div style='font-size:13px;font-weight:700;color:#38bdf8;margin-bottom:10px'>
  📋 Normal Patient Journey (RSSB)
</div>
<div style='display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11px;
     font-family:monospace;color:#94a3b8'>
  <span style='background:#1e3a5f;color:#93c5fd;padding:3px 9px;border-radius:6px'>🏥 Register at clinic</span>
  <span>→</span>
  <span style='background:#1e3a5f;color:#93c5fd;padding:3px 9px;border-radius:6px'>👨‍⚕️ See doctor</span>
  <span>→</span>
  <span style='background:#1e3a5f;color:#93c5fd;padding:3px 9px;border-radius:6px'>📝 Prescription + voucher</span>
  <span>→</span>
  <span style='background:#1e3a5f;color:#93c5fd;padding:3px 9px;border-radius:6px'>💰 Pay 15% hospital</span>
  <span>→</span>
  <span style='background:#14532d;color:#86efac;padding:3px 9px;border-radius:6px'>💊 Pharmacy dispenses</span>
  <span>→</span>
  <span style='background:#14532d;color:#86efac;padding:3px 9px;border-radius:6px'>💰 Pay 15% pharmacy</span>
  <span>→</span>
  <span style='background:#14532d;color:#86efac;padding:3px 9px;border-radius:6px'>📤 Pharmacy claims 85% RSSB</span>
</div>
<div style='margin-top:10px;font-size:11px;color:#64748b;font-family:monospace'>
  <b style='color:#ef4444'>⚠️ Fraud signal:</b>
  Pharmacy claims 85% but the patient has <i>no clinic visit record</i>
  — no consultation, no real prescription.
</div>
</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # RWANDA-AWARE NAME MATCHING ENGINE
    # Handles all observed patterns in this dataset:
    #   1. Name reversal:    ROSETTE MUKAMUGEMA ↔ MUKAMUGEMA ROSETTE
    #   2. N. prefix:        NYIRANSABIMANA ↔ N.NSABIMANA (Nyira patronymic)
    #   3. JMV initials:     JEAN MARIE VIANNEY ↔ J.M.V
    #   4. Other initials:   J.D'ARC, J.PIERRE, J.CLAUDE, J.DE DIEU, T.ALPHONSE
    #   5. BEBE prefix:      BEBE MUKASHEMA CLAUDETTE ↔ CLAUDETTE MUKASHEMA
    #   6. Spelling typos:   FABIOLA ↔ FABIORA, NZABALINDA ↔ NZABARINDA
    #   7. Missing tokens:   NIYONSENGA BINTU EMMANUEL ↔ EMMANUEL NIYONSENGA
    #   8. Initial middle:   BIYINGOMA T.ALPHONSE ↔ TURIKUNKIKO ALPHONSE BIYINGOMA
    # ════════════════════════════════════════════════════════════════════════

    def _rw_normalise(raw):
        """Pre-process a name with Rwanda-specific expansions."""
        n = str(raw).upper().strip()
        n = re.sub(r'^BEBE\s+', '', n)                          # strip child prefix
        n = re.sub(r'\bJ\.M\.V\b', 'JEAN MARIE VIANNEY', n)    # JMV initials
        n = re.sub(r"\bJ\.D'ARC\b", 'JEANNE DARC', n)
        n = re.sub(r'\bJE\s+DE\s+DIEU\b', 'JEAN DE DIEU', n)
        n = re.sub(r'\bJ\.DE\s+DIEU\b', 'JEAN DE DIEU', n)
        n = re.sub(r'\bJ\.\s*PIERRE\b', 'JEAN PIERRE', n)
        n = re.sub(r'\bJ\.\s*CLAUDE\b', 'JEAN CLAUDE', n)
        n = re.sub(r'\bJ\.\s*BOSCO\b', 'JEAN BOSCO', n)
        n = re.sub(r"['\-\.]", ' ', n)                          # remove punctuation
        n = re.sub(r'\s+', ' ', n).strip()
        return n

    def _rw_tokens(name):
        """Return token set with NYIRA expansion."""
        base = set(_rw_normalise(name).split())
        expanded = set(base)
        for t in base:
            if t.startswith('NYIRA') and len(t) > 5:
                stem = t[5:]               # NYIRАНСАБIMANA → NSABIMANA
                expanded.add(stem)
                expanded.add('N' + stem)   # → NNSABIMANA (catches N.NSABIMANA)
            elif t.startswith('N') and len(t) > 3:
                expanded.add('NYIRA' + t)  # NSABIMANA → NYIRАНСАБIMANA
        return expanded

    def _initial_bonus(ta, tb):
        """Score for single-letter tokens that match the first char of a word."""
        bonus = 0
        for a in ta:
            if len(a) == 1:
                for b in tb:
                    if b.startswith(a) and len(b) > 1:
                        bonus += 1
        for b in tb:
            if len(b) == 1:
                for a in ta:
                    if a.startswith(b) and len(a) > 1:
                        bonus += 1
        return min(bonus * 0.12, 0.30)

    def _rw_name_score(ph_name, h_bene, h_affil=''):
        """
        Rwanda-aware name similarity.
        Returns (score 0–1, 'BENEFICIARY' | 'AFFILIATE').
        Benchmarked: 153 HIGH / 20 MEDIUM / 0 LOW on 173 real hospital-pharmacy pairs.
        """
        def _vs(h_raw):
            if not str(h_raw).strip() or str(h_raw).upper() in ('NAN', ''):
                return 0.0
            ta = _rw_tokens(ph_name)
            tb = _rw_tokens(h_raw)
            ta_base = set(_rw_normalise(ph_name).split())
            tb_base = set(_rw_normalise(h_raw).split())

            tok   = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
            init  = _initial_bonus(ta_base, tb_base)
            seq   = _SQ(None, _rw_normalise(ph_name), _rw_normalise(h_raw)).ratio()

            # Per-token fuzzy for typo cases (e.g. FABIOLA / FABIORA)
            best_fuzzy = max(
                (_SQ(None, a, b).ratio()
                 for a in ta_base if len(a) > 3
                 for b in tb_base if len(b) > 3),
                default=0.0
            )
            fuzzy = best_fuzzy * (0.55 if len(ta_base) > 2 else 0.70)

            return min(1.0, max(tok + init, seq * 0.80, fuzzy))

        b = _vs(h_bene)
        a = _vs(h_affil)
        return (round(b, 3), 'BENEFICIARY') if b >= a else (round(a, 3), 'AFFILIATE')

    def _confidence(score):
        if score >= 0.65: return 'HIGH',   '#22c55e', 'badge-green'
        if score >= 0.35: return 'MEDIUM', '#f59e0b', 'badge-amber'
        return                   'LOW',    '#ef4444', 'badge-red'

    # ── Facility file parser ──────────────────────────────────────────────────
    def _parse_facility(raw_bytes, filename):
        try:
            fname = filename.lower()
            if fname.endswith('.csv'):
                raw = pd.read_csv(io.BytesIO(raw_bytes), encoding='utf-8', on_bad_lines='skip')
                return _extract_fac(raw, filename, 'csv'), 'csv', None

            xl = pd.ExcelFile(io.BytesIO(raw_bytes))
            chosen = None
            for prio in ['after','verified','clean','before','data','invoice']:
                for sn in xl.sheet_names:
                    if prio in sn.lower(): chosen = sn; break
                if chosen: break
            if not chosen:
                sizes = {}
                for sn in xl.sheet_names:
                    try: sizes[sn] = xl.parse(sn, header=None).shape[0]
                    except: pass
                chosen = max(sizes, key=sizes.get)

            raw0 = xl.parse(chosen, header=None)
            # Detect header row
            hrow = 0
            for i, row in raw0.head(20).iterrows():
                joined = ' '.join(str(v).lower() for v in row if pd.notna(v))
                if any(k in joined for k in ['affil','benefi','patient name','voucher']):
                    hrow = i; break
            if hrow > 0:
                raw = pd.read_excel(io.BytesIO(raw_bytes),
                                    sheet_name=chosen, header=hrow)
            else:
                raw0.columns = raw0.iloc[0]
                raw = raw0.iloc[1:].reset_index(drop=True)
            raw.columns = [str(c).strip() for c in raw.columns]
            return _extract_fac(raw, filename, chosen), chosen, None
        except Exception as e:
            return None, None, str(e)

    def _extract_fac(raw, filename, sheet):
        def _f(*pats):
            for p in pats:
                for c in raw.columns:
                    if re.search(p, str(c).lower()): return c
            return None

        rama_c  = _f(r'affil', r'rama', r'member.*no')
        bene_c  = _f(r'benefi.*name', r'patient.*name', r'client.*name')
        affil_c = _f(r'affil.*name')
        date_c  = _f(r'^date$', r'visit.*date', r'dispensing', r'service.*date')
        vou_c   = _f(r'voucher.*id', r'voucher.*ident', r'paper.*code', r'invoice.*no')
        total_c = _f(r'total.*amount', r'total.*cost', r'^total$')
        if not rama_c: raise ValueError('No RAMA/Affiliation column found')

        no_col = raw.columns[0]
        raw = raw[pd.to_numeric(raw[no_col], errors='coerce').notna()].copy()

        out = pd.DataFrame()
        out['_rama']  = raw[rama_c].astype(str).str.strip().str.upper()
        out['_bene']  = raw[bene_c].fillna('').astype(str).str.strip()  if bene_c  else ''
        out['_affil'] = raw[affil_c].fillna('').astype(str).str.strip() if affil_c else ''
        out['_date']  = pd.to_datetime(raw[date_c], errors='coerce')    if date_c  else pd.NaT
        out['_vou']   = raw[vou_c].astype(str).str.strip()              if vou_c   else ''
        out['_total'] = pd.to_numeric(raw[total_c], errors='coerce').fillna(0) if total_c else 0
        out['_src']   = filename
        out['_sheet'] = sheet
        out = out[out['_rama'].str.len() > 2].reset_index(drop=True)
        return out

    # ── Step 1 — Upload ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">📂 Step 1 — Upload Hospital & Clinic Files</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:11px;color:#64748b;font-family:monospace;margin-bottom:10px'>
  Upload one or more hospital/clinic Excel files covering the <b>same period</b>
  as the pharmacy report. The parser auto-selects the <b style='color:#e2e8f0'>AFTER</b>
  (verified) sheet and reads both
  <b style='color:#e2e8f0'>BENEFICIARY</b> and <b style='color:#e2e8f0'>AFFILIATE</b>
  name columns. Upload more facilities to reduce unverified cases.
</div>""", unsafe_allow_html=True)

    xf_up = st.file_uploader('Upload facility files (Excel / CSV)',
                              type=['xlsx','xls','csv'],
                              accept_multiple_files=True, key='xf_up')
    if xf_up:
        frames = []
        for uf in xf_up:
            rb = uf.read()
            parsed, sheet_used, err = _parse_facility(rb, uf.name)
            if parsed is not None and len(parsed) > 0:
                frames.append(parsed)
                unique_ramas = parsed['_rama'].nunique()
                st.success(f"✅ **{uf.name}** — {len(parsed):,} records · "
                           f"sheet: *{sheet_used}* · {unique_ramas:,} unique patients")
            else:
                st.error(f"❌ **{uf.name}** — {err}")
        if frames:
            st.session_state['fd_fac'] = pd.concat(frames, ignore_index=True)

    if 'fd_fac' not in st.session_state:
        st.info('👆 Upload at least one hospital or clinic file to begin.')
        st.stop()

    fac = st.session_state['fd_fac']
    fac_ramas = set(fac['_rama'])

    srcs = fac.groupby('_src').size().reset_index(name='records')
    fcols = st.columns(min(len(srcs), 4))
    for i, row in srcs.iterrows():
        fcols[i % len(fcols)].metric(row['_src'][:40], f"{row['records']:,} records")

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Step 2 — Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">⚙️ Step 2 — Detection Settings</div>',
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        date_win = st.slider('Date window (±days)', 0, 30, 7,
            help='Days between pharmacy dispensing and clinic visit to count as linked')
    with s2:
        name_thr = st.slider('Min name confidence score', 0.0, 1.0, 0.35, 0.05,
            help='0.35 catches all Rwandan name variants including N./JMV initials and typos')
    with s3:
        req_name = st.checkbox('Require name match', value=True,
            help='Uncheck to match on RAMA number only')

    # ── Build pharmacy working cols ───────────────────────────────────────────
    def _gc(*keys):
        for k in keys:
            if k in df.columns: return k
        return None

    _vid = _gc('voucher_id','paper_code','Paper Code')
    _pnm = _gc('patient_name','Patient Name')
    _rma = _gc('patient_id','rama_number','RAMA Number')
    _dt  = _gc('visit_date','dispensing_date','Dispensing Date')
    _ins = _gc('insurance_copay','Insurance Co-payment')
    _tot = _gc('amount','total_cost','Total Cost')
    _doc = _gc('doctor_name','practitioner_name','Practitioner Name')
    _dpt = _gc('practitioner_type','Practitioner Type')

    ph = df.copy()
    ph['_rama'] = ph[_rma].astype(str).str.strip().str.upper() if _rma else ''
    ph['_name'] = ph[_pnm].fillna('').astype(str).str.strip() if _pnm else ''
    ph['_date'] = pd.to_datetime(ph[_dt], errors='coerce') if _dt else pd.NaT
    ph['_vou']  = ph[_vid].astype(str).str.strip() if _vid else ''
    ph['_ins']  = pd.to_numeric(ph[_ins], errors='coerce').fillna(0) if _ins else 0
    ph['_tot']  = pd.to_numeric(ph[_tot], errors='coerce').fillna(0) if _tot else 0
    ph['_doc']  = ph[_doc].fillna('').astype(str) if _doc else ''
    ph['_dpt']  = ph[_dpt].fillna('').astype(str) if _dpt else ''

    # ── Core matching loop ────────────────────────────────────────────────────
    rows = []
    for _, pr in ph.iterrows():
        rama    = pr['_rama']
        ph_date = pr['_date']
        ph_name = pr['_name']

        fac_rows = fac[fac['_rama'] == rama]

        if fac_rows.empty:
            rows.append({'status':'NO_RECORD','conf':None,'score':None,
                         'via':None,'days':None,
                         'ph_vou':pr['_vou'],'ph_name':ph_name,
                         'ph_rama':rama,'ph_date':ph_date,
                         'ph_ins':pr['_ins'],'ph_tot':pr['_tot'],
                         'ph_doc':pr['_doc'],'ph_dpt':pr['_dpt'],
                         'h_vou':None,'h_bene':None,'h_affil':None,
                         'h_date':None,'h_src':None})
            continue

        best = None; best_delta = 9999; best_score = -1
        for _, fr in fac_rows.iterrows():
            h_date = fr['_date']
            delta  = abs((ph_date - h_date).days) \
                     if pd.notna(ph_date) and pd.notna(h_date) else 9999
            score, via = _rw_name_score(ph_name, fr['_bene'], fr['_affil'])
            name_ok = (score >= name_thr) if req_name else True
            if delta <= date_win and name_ok:
                if delta < best_delta or (delta == best_delta and score > best_score):
                    best_delta = delta; best_score = score; best = (fr, via)

        if best:
            fr, via = best
            clabel,_,_ = _confidence(best_score)
            rows.append({'status':'MATCHED','conf':clabel,'score':best_score,
                         'via':via,'days':best_delta,
                         'ph_vou':pr['_vou'],'ph_name':ph_name,
                         'ph_rama':rama,'ph_date':ph_date,
                         'ph_ins':pr['_ins'],'ph_tot':pr['_tot'],
                         'ph_doc':pr['_doc'],'ph_dpt':pr['_dpt'],
                         'h_vou':fr['_vou'],'h_bene':fr['_bene'],
                         'h_affil':fr['_affil'],'h_date':fr['_date'],
                         'h_src':fr['_src']})
        else:
            # RAMA found but no date+name link
            fac_rows2 = fac_rows.copy()
            fac_rows2['_delta'] = fac_rows2['_date'].apply(
                lambda d: abs((ph_date-d).days)
                if pd.notna(ph_date) and pd.notna(d) else 9999)
            fr0 = fac_rows2.sort_values('_delta').iloc[0]
            s0, v0 = _rw_name_score(ph_name, fr0['_bene'], fr0['_affil'])
            cl,_,_ = _confidence(s0)
            nd = int(fr0['_delta']) if fr0['_delta'] < 9000 else None
            rows.append({'status':'UNLINKED','conf':cl,'score':round(s0,3),
                         'via':v0,'days':nd,
                         'ph_vou':pr['_vou'],'ph_name':ph_name,
                         'ph_rama':rama,'ph_date':ph_date,
                         'ph_ins':pr['_ins'],'ph_tot':pr['_tot'],
                         'ph_doc':pr['_doc'],'ph_dpt':pr['_dpt'],
                         'h_vou':fr0['_vou'],'h_bene':fr0['_bene'],
                         'h_affil':fr0['_affil'],'h_date':fr0['_date'],
                         'h_src':fr0['_src']})

    res      = pd.DataFrame(rows)
    no_rec   = res[res['status']=='NO_RECORD']
    unlinked = res[res['status']=='UNLINKED']
    matched  = res[res['status']=='MATCHED']
    total_ins   = res['ph_ins'].sum()
    norec_ins   = no_rec['ph_ins'].sum()
    unlink_ins  = unlinked['ph_ins'].sum()
    match_ins   = matched['ph_ins'].sum()

    # ── Step 3 — Dashboard ────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">📊 Step 3 — Fraud Detection Dashboard</div>',
                unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric('Pharmacy vouchers',     f"{len(ph):,}")
    k2.metric('✅ Visit confirmed',     f"{len(matched):,}",
              f"{100*len(matched)/max(len(ph),1):.1f}%")
    k3.metric('🔴 No facility record', f"{len(no_rec):,}",
              f"RWF {norec_ins:,.0f}", delta_color='inverse')
    k4.metric('🟡 RAMA found, unlinked',f"{len(unlinked):,}",
              f"RWF {unlink_ins:,.0f}", delta_color='inverse')
    k5.metric('Facilities loaded',     f"{fac['_src'].nunique()}")

    b_m = 100*match_ins /total_ins if total_ins else 0
    b_u = 100*unlink_ins/total_ins if total_ins else 0
    b_n = 100*norec_ins /total_ins if total_ins else 0
    st.markdown(f"""
<div style='margin:16px 0 6px'>
  <div style='font-size:11px;color:#64748b;font-family:monospace;margin-bottom:4px'>
    Insurance claims by verification status
  </div>
  <div style='display:flex;height:18px;border-radius:8px;overflow:hidden'>
    <div style='width:{b_m:.1f}%;background:#22c55e'></div>
    <div style='width:{b_u:.1f}%;background:#f59e0b'></div>
    <div style='width:{b_n:.1f}%;background:#ef4444'></div>
  </div>
  <div style='display:flex;gap:20px;margin-top:5px;font-size:11px;font-family:monospace'>
    <span style='color:#22c55e'>■ Verified {b_m:.1f}%  RWF {match_ins:,.0f}</span>
    <span style='color:#f59e0b'>■ Unlinked {b_u:.1f}%  RWF {unlink_ins:,.0f}</span>
    <span style='color:#ef4444'>■ No record {b_n:.1f}%  RWF {norec_ins:,.0f}</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style='background:#020e1a;border:1px solid #1e3a5f;border-radius:8px;
     padding:10px 16px;font-size:11px;font-family:monospace;color:#64748b;
     margin:10px 0 24px'>
  <b style='color:#38bdf8'>Coverage:</b>
  {fac['_src'].nunique()} facility file(s) · {len(fac_ramas):,} unique RAMA numbers.
  <span style='color:#475569'>
  Unverified patients may have visited clinics not yet uploaded.
  </span>
  &nbsp;|&nbsp;
  <b style='color:#38bdf8'>Name matching</b> handles all Rwandan naming patterns:
  reversed order · N./Nyira prefix · J.M.V initials · BEBE prefix ·
  spelling variants · Beneficiary vs Affiliate columns.
</div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2a38;margin:4px 0 24px'>", unsafe_allow_html=True)

    # ── TABLE 1 — NO RECORD ───────────────────────────────────────────────────
    st.markdown(f"""
<div class='fd-card fd-red'>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div>
      <span style='font-size:16px;font-weight:800;color:#f87171;font-family:Syne,sans-serif'>
        🔴 Table 1 — No Hospital / Clinic Visit Record
      </span>
      <span class='badge badge-red' style='margin-left:10px'>{len(no_rec):,} vouchers</span>
      <span class='badge badge-red' style='margin-left:6px'>RWF {norec_ins:,.0f}</span>
    </div>
    <span style='font-size:11px;color:#991b1b;font-family:monospace'>
      Patient RAMA not found in any uploaded facility file
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    if not no_rec.empty:
        c1,c2,c3 = st.columns([2.5,2,1.2])
        with c1: t1s = st.text_input('🔍 Search', placeholder='Name, RAMA, doctor…', key='t1s')
        with c2: t1d = st.selectbox('Filter by prescriber',
                      ['All']+sorted(no_rec['ph_doc'].dropna().unique().tolist()), key='t1d')
        with c3: t1m = st.number_input('Min insurance (RWF)', 0, value=0, step=5000, key='t1m')

        t1 = no_rec.copy()
        if t1s: t1 = t1[t1.apply(lambda c:c.astype(str).str.contains(t1s,case=False,na=False)).any(axis=1)]
        if t1d != 'All': t1 = t1[t1['ph_doc']==t1d]
        if t1m > 0: t1 = t1[t1['ph_ins']>=t1m]

        t1_show = pd.DataFrame({
            'Voucher':             t1['ph_vou'],
            'Patient Name':        t1['ph_name'],
            'RAMA Number':         t1['ph_rama'],
            'Dispensing Date':     pd.to_datetime(t1['ph_date'],errors='coerce').dt.strftime('%d/%m/%Y').fillna('—'),
            'Insurance Claim RWF': t1['ph_ins'],
            'Total Cost RWF':      t1['ph_tot'],
            'Prescriber':          t1['ph_doc'],
            'Specialty':           t1['ph_dpt'],
        }).sort_values('Insurance Claim RWF', ascending=False)
        t1_show.index = range(1, len(t1_show)+1)

        st.markdown(
            f"<div style='font-size:11px;color:{MUTED};font-family:monospace;margin-bottom:6px'>"
            f"<b style='color:#f87171'>{len(t1_show):,}</b> vouchers · "
            f"RWF at risk: <b style='color:#ef4444'>{t1['ph_ins'].sum():,.0f}</b></div>",
            unsafe_allow_html=True)
        st.dataframe(t1_show, use_container_width=True, height=340)

        with st.expander('📊 Prescriber risk breakdown', expanded=False):
            dr = (no_rec.groupby('ph_doc')['ph_ins']
                  .agg(Vouchers='count', Total_RWF='sum')
                  .sort_values('Total_RWF', ascending=False).reset_index())
            dr.columns = ['Prescriber','Vouchers','Total Claimed (RWF)']
            st.dataframe(dr, use_container_width=True, height=260)

        _b = io.BytesIO()
        with pd.ExcelWriter(_b, engine='openpyxl') as _xw:
            from openpyxl.styles import PatternFill as _PF,Font as _F,Alignment as _A
            t1_show.to_excel(_xw, index=False, sheet_name='No Facility Record')
            _ws=_xw.sheets['No Facility Record']
            for _c in _ws[1]:
                _c.fill=_PF('solid',fgColor='7F1D1D')
                _c.font=_F(bold=True,color='FFFFFF',name='Arial',size=10)
                _c.alignment=_A(horizontal='center',wrap_text=True)
            for _i,_r in enumerate(_ws.iter_rows(min_row=2),2):
                for _c in _r: _c.fill=_PF('solid',fgColor='FFE4E4' if _i%2==0 else 'FFFFFF')
        _b.seek(0)
        st.download_button('⬇️ Download Table 1 (.xlsx)', _b.getvalue(),
            'table1_no_facility_record.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl1')

    st.markdown("<hr style='border-color:#1e2a38;margin:28px 0'>", unsafe_allow_html=True)

    # ── TABLE 2 — UNLINKED ────────────────────────────────────────────────────
    st.markdown(f"""
<div class='fd-card fd-amber'>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div>
      <span style='font-size:16px;font-weight:800;color:#fbbf24;font-family:Syne,sans-serif'>
        🟡 Table 2 — RAMA Found but Visit Not Linked
      </span>
      <span class='badge badge-amber' style='margin-left:10px'>{len(unlinked):,} vouchers</span>
      <span class='badge badge-amber' style='margin-left:6px'>RWF {unlink_ins:,.0f}</span>
    </div>
    <span style='font-size:11px;color:#92400e;font-family:monospace'>
      Patient in facility records but outside ±{date_win}d or name below threshold
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    if not unlinked.empty:
        c1,c2 = st.columns([2.5,1.5])
        with c1: t2s = st.text_input('🔍 Search', placeholder='Name, RAMA…', key='t2s')
        with c2: t2g = st.number_input('Max days apart', 1, 365, 60, key='t2g')

        t2 = unlinked.copy()
        if t2s: t2 = t2[t2.apply(lambda c:c.astype(str).str.contains(t2s,case=False,na=False)).any(axis=1)]
        t2 = t2[t2['days'].isna()|(t2['days']<=t2g)]

        t2_show = pd.DataFrame({
            'Voucher':           t2['ph_vou'],
            'Pharmacy Patient':  t2['ph_name'],
            'RAMA':              t2['ph_rama'],
            'Pharmacy Date':     pd.to_datetime(t2['ph_date'],errors='coerce').dt.strftime('%d/%m/%Y').fillna('—'),
            'Insurance RWF':     t2['ph_ins'],
            'Hosp. Beneficiary': t2['h_bene'],
            'Hosp. Affiliate':   t2['h_affil'],
            'Facility Date':     pd.to_datetime(t2['h_date'],errors='coerce').dt.strftime('%d/%m/%Y').fillna('—'),
            'Days Apart':        t2['days'],
            'Name Score':        t2['score'],
            'Matched Via':       t2['via'],
            'Confidence':        t2['conf'],
        }).sort_values('Days Apart', na_position='last')
        t2_show.index = range(1, len(t2_show)+1)

        st.markdown(
            f"<div style='font-size:11px;color:{MUTED};font-family:monospace;margin-bottom:6px'>"
            f"<b style='color:#fbbf24'>{len(t2_show):,}</b> vouchers · "
            f"RWF: <b style='color:#f59e0b'>{t2['ph_ins'].sum():,.0f}</b></div>",
            unsafe_allow_html=True)
        st.dataframe(t2_show, use_container_width=True, height=320)

        _b2 = io.BytesIO()
        with pd.ExcelWriter(_b2, engine='openpyxl') as _xw:
            from openpyxl.styles import PatternFill as _PF,Font as _F,Alignment as _A
            t2_show.to_excel(_xw, index=False, sheet_name='Unlinked Visits')
            _ws=_xw.sheets['Unlinked Visits']
            for _c in _ws[1]:
                _c.fill=_PF('solid',fgColor='78350F')
                _c.font=_F(bold=True,color='FFFFFF',name='Arial',size=10)
                _c.alignment=_A(horizontal='center',wrap_text=True)
            for _i,_r in enumerate(_ws.iter_rows(min_row=2),2):
                for _c in _r: _c.fill=_PF('solid',fgColor='FEF3C7' if _i%2==0 else 'FFFFFF')
        _b2.seek(0)
        st.download_button('⬇️ Download Table 2 (.xlsx)', _b2.getvalue(),
            'table2_unlinked.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl2')

    st.markdown("<hr style='border-color:#1e2a38;margin:28px 0'>", unsafe_allow_html=True)

    # ── TABLE 3 — VERIFIED ────────────────────────────────────────────────────
    st.markdown(f"""
<div class='fd-card fd-green'>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div>
      <span style='font-size:16px;font-weight:800;color:#4ade80;font-family:Syne,sans-serif'>
        ✅ Table 3 — Verified: Visit Confirmed
      </span>
      <span class='badge badge-green' style='margin-left:10px'>{len(matched):,} vouchers</span>
      <span class='badge badge-green' style='margin-left:6px'>RWF {match_ins:,.0f}</span>
    </div>
    <span style='font-size:11px;color:#14532d;font-family:monospace'>
      Legitimate journey confirmed — clinic visit linked to pharmacy dispensing
    </span>
  </div>
</div>""", unsafe_allow_html=True)

    with st.expander('View verified records', expanded=False):
        if not matched.empty:
            cc = matched['conf'].value_counts()
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric('🟢 HIGH confidence',   cc.get('HIGH',0))
            mc2.metric('🟡 MEDIUM confidence', cc.get('MEDIUM',0))
            mc3.metric('🔴 LOW confidence',    cc.get('LOW',0))

        t3s = st.text_input('🔍 Search verified', key='t3s')
        t3 = matched.copy()
        if t3s: t3 = t3[t3.apply(lambda c:c.astype(str).str.contains(t3s,case=False,na=False)).any(axis=1)]

        t3_show = pd.DataFrame({
            'Voucher':           t3['ph_vou'],
            'Pharmacy Patient':  t3['ph_name'],
            'RAMA':              t3['ph_rama'],
            'Pharmacy Date':     pd.to_datetime(t3['ph_date'],errors='coerce').dt.strftime('%d/%m/%Y').fillna('—'),
            'Insurance RWF':     t3['ph_ins'],
            'Hosp. Beneficiary': t3['h_bene'],
            'Hosp. Affiliate':   t3['h_affil'],
            'Facility Date':     pd.to_datetime(t3['h_date'],errors='coerce').dt.strftime('%d/%m/%Y').fillna('—'),
            'Fac. Voucher':      t3['h_vou'],
            'Days Apart':        t3['days'],
            'Name Score':        t3['score'],
            'Matched Via':       t3['via'],
            'Confidence':        t3['conf'],
            'Facility':          t3['h_src'],
        }).sort_values('Days Apart').reset_index(drop=True)
        t3_show.index = t3_show.index+1
        st.dataframe(t3_show, use_container_width=True, height=320)

    st.markdown("<hr style='border-color:#1e2a38;margin:28px 0'>", unsafe_allow_html=True)

    # ── Full report ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">⬇️ Full Fraud Detection Report</div>',
                unsafe_allow_html=True)

    if st.button('📊 Generate Full Report (4 sheets)', type='primary', key='fd_gen'):
        from openpyxl import Workbook as _WB
        from openpyxl.styles import (PatternFill as _PF,Font as _F,
                                     Alignment as _AL,Border as _BD,Side as _SD)
        from openpyxl.utils import get_column_letter as _gcl

        _wb=_WB(); _wb.remove(_wb.active)
        _TH=_SD(border_style='thin',color='CCCCCC')
        _BDR=_BD(left=_TH,right=_TH,top=_TH,bottom=_TH)

        def _fmtd(s):
            return pd.to_datetime(s,errors='coerce').dt.strftime('%d/%m/%Y').fillna('')

        def _sht(wb,title,data,hcol,evencol,oddcol='FFFFFF'):
            ws=wb.create_sheet(title)
            for ci,col in enumerate(data.columns,1):
                c=ws.cell(1,ci,col)
                c.fill=_PF('solid',fgColor=hcol)
                c.font=_F(bold=True,color='FFFFFF',name='Arial',size=10)
                c.alignment=_AL(horizontal='center',wrap_text=True)
                c.border=_BDR
                ws.column_dimensions[_gcl(ci)].width=max(14,min(len(str(col))+4,36))
            ws.row_dimensions[1].height=30; ws.freeze_panes='A2'
            for ri,(_,row) in enumerate(data.iterrows(),2):
                bg=evencol if ri%2==0 else oddcol
                for ci,val in enumerate(row,1):
                    v='' if isinstance(val,float) and math.isnan(val) else val
                    c=ws.cell(ri,ci,v)
                    c.font=_F(name='Arial',size=10)
                    c.fill=_PF('solid',fgColor=bg)
                    c.border=_BDR; c.alignment=_AL(horizontal='left')

        # Summary
        _ws0=_wb.create_sheet('Summary'); _ws0.sheet_view.showGridLines=False
        _rws=[
            ('Report Period','January 2025'),
            ('Pharmacy Vouchers',len(ph)),
            ('Facilities Loaded',fac['_src'].nunique()),
            ('Facility RAMA Numbers',len(fac_ramas)),
            ('',''),
            ('✅ VERIFIED',len(matched)),
            ('   % of vouchers',f"{100*len(matched)/max(len(ph),1):.1f}%"),
            ('   Insurance RWF',match_ins),
            ('',''),
            ('🟡 UNLINKED (RAMA found, no date/name link)',len(unlinked)),
            ('   % of vouchers',f"{100*len(unlinked)/max(len(ph),1):.1f}%"),
            ('   Insurance RWF',unlink_ins),
            ('',''),
            ('🔴 NO FACILITY RECORD',len(no_rec)),
            ('   % of vouchers',f"{100*len(no_rec)/max(len(ph),1):.1f}%"),
            ('   Insurance RWF',norec_ins),
            ('',''),
            ('TOTAL AT RISK (RWF)',norec_ins+unlink_ins),
            ('As % of total claims',f"{100*(norec_ins+unlink_ins)/max(total_ins,1):.1f}%"),
        ]
        for ri,(lbl,val) in enumerate(_rws,2):
            lc=_ws0.cell(ri,1,lbl); vc=_ws0.cell(ri,2,val)
            is_t='TOTAL' in str(lbl).upper()
            lc.font=_F(name='Arial',size=11,bold=is_t or str(lbl).startswith(('✅','🟡','🔴','Report','Facil')))
            vc.font=_F(name='Arial',size=11,bold=is_t)
            if is_t:
                for c in(lc,vc):
                    c.fill=_PF('solid',fgColor='7F1D1D')
                    c.font=_F(name='Arial',size=12,bold=True,color='FFFFFF')
        _ws0.column_dimensions['A'].width=46; _ws0.column_dimensions['B'].width=22

        # Sheet 1
        _d1=no_rec[['ph_vou','ph_name','ph_rama','ph_date','ph_ins','ph_tot','ph_doc','ph_dpt']].copy()
        _d1.columns=['Voucher','Patient Name','RAMA','Dispensing Date','Insurance RWF','Total RWF','Prescriber','Specialty']
        _d1['Dispensing Date']=_fmtd(_d1['Dispensing Date'])
        _sht(_wb,'1 - No Facility Record',_d1,'7F1D1D','FFE4E4')

        # Sheet 2
        _d2=unlinked[['ph_vou','ph_name','ph_rama','ph_date','ph_ins','h_bene','h_affil','h_date','h_src','days','score','via','conf']].copy()
        _d2.columns=['Voucher','Patient','RAMA','Pharmacy Date','Insurance RWF','Hosp. Beneficiary','Hosp. Affiliate','Facility Date','Facility','Days Apart','Name Score','Matched Via','Confidence']
        _d2['Pharmacy Date']=_fmtd(_d2['Pharmacy Date']); _d2['Facility Date']=_fmtd(_d2['Facility Date'])
        _sht(_wb,'2 - Unlinked Visits',_d2,'78350F','FEF3C7')

        # Sheet 3
        _d3=matched[['ph_vou','ph_name','ph_rama','ph_date','ph_ins','h_vou','h_bene','h_affil','h_date','h_src','days','score','via','conf']].copy()
        _d3.columns=['Voucher','Patient','RAMA','Pharmacy Date','Insurance RWF','Fac. Voucher','Hosp. Beneficiary','Hosp. Affiliate','Facility Date','Facility','Days Apart','Name Score','Matched Via','Confidence']
        _d3['Pharmacy Date']=_fmtd(_d3['Pharmacy Date']); _d3['Facility Date']=_fmtd(_d3['Facility Date'])
        _sht(_wb,'3 - Verified',_d3,'14532D','E7F5EC')

        _buf=io.BytesIO(); _wb.save(_buf); _buf.seek(0)
        st.success('✅ Report ready!')
        st.download_button(
            '⬇️ Download Full Fraud Detection Report (.xlsx)',
            _buf.getvalue(), 'fraud_detection_report.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='dl_full')
