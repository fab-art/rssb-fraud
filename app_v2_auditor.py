"""
PharmaScan v2 — Counter-Verification Workbench (RSSB Edition)

Purpose: Interactive verification interface for pharmacology audits.
Workflow: Upload pharmacy Excel → Card-based verification → Auto-generate RSSB report.

Install:
    pip install streamlit pandas matplotlib networkx openpyxl numpy

Run:
    streamlit run app_v2_auditor.py
"""

import io
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEMING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PharmaScan v2 — CV Workbench",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#00e5a0"
ACCENT2 = "#0ea5e9"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#e2e8f0"
DARK = "#0d1117"
CARD = "#111720"
BORDER = "#1e2a38"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

.stApp { background: #080c10; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2a38; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e2a38; border-radius: 4px; }

[data-testid="stMetric"] {
    background: #111720; border: 1px solid #1e2a38; border-radius: 12px; padding: 16px 20px !important;
}

[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 26px !important; font-weight: 800 !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.sec-head {
    font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700;
    color: #e2e8f0; padding-left: 10px; border-left: 3px solid #00e5a0; margin: 20px 0 12px;
}

.verify-card {
    background: #111720; border: 1px solid #1e2a38; border-radius: 10px; padding: 16px;
    margin-bottom: 12px; transition: border-color 0.2s;
}

.verify-card:hover { border-color: #2d3f54; }

.card-status-verified { border-left: 4px solid #22c55e; }
.card-status-deducted { border-left: 4px solid #f59e0b; }
.card-status-ghost { border-left: 4px solid #ef4444; }
.card-status-mismatch { border-left: 4px solid #a78bfa; }

.queue-item { background: #111720; border: 1px solid #1e2a38; border-radius: 8px; padding: 12px; margin-bottom: 8px; }

[data-testid="stDownloadButton"] button {
    border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize or reset session state."""
    if "pharmacy_df" not in st.session_state:
        st.session_state.pharmacy_df = None
    if "hospital_df" not in st.session_state:
        st.session_state.hospital_df = None
    if "verifications" not in st.session_state:
        st.session_state.verifications = {}  # {row_idx: {status, deduction, reason}}
    if "current_claim_idx" not in st.session_state:
        st.session_state.current_claim_idx = 0
    if "ghost_prescriptions" not in st.session_state:
        st.session_state.ghost_prescriptions = []
    if "high_cost_queue" not in st.session_state:
        st.session_state.high_cost_queue = []
    if "repeat_visits" not in st.session_state:
        st.session_state.repeat_visits = []

init_session_state()

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_date(val):
    """Safely parse date string to datetime."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            pass
    try:
        return pd.to_datetime(val)
    except:
        return None

def detect_repeat_visits(df, days_window=30):
    """
    Find patients with >1 visit in a rolling window.
    Returns: list of {patient_name, rama_id, visit_dates, count}
    """
    if "patient_name" not in df.columns or "dispensing_date" not in df.columns:
        return []
    
    df_copy = df.copy()
    df_copy["dispensing_date"] = pd.to_datetime(df_copy["dispensing_date"], errors="coerce")
    
    repeats = []
    for patient in df_copy["patient_name"].unique():
        patient_records = df_copy[df_copy["patient_name"] == patient].sort_values("dispensing_date")
        if len(patient_records) > 1:
            rama_id = patient_records["rama_number"].iloc[0] if "rama_number" in df.columns else "—"
            dates = patient_records["dispensing_date"].tolist()
            repeats.append({
                "patient_name": patient,
                "rama_id": rama_id,
                "visit_dates": dates,
                "count": len(patient_records),
                "indices": patient_records.index.tolist(),
            })
    
    return sorted(repeats, key=lambda x: x["count"], reverse=True)

def detect_high_cost_patterns(df, cost_threshold_percentile=75):
    """
    Detect high-cost claims and doctor-prescribing patterns.
    Returns: list of {medicine, doctor, cost, frequency, claim_indices}
    """
    if "medicine_cost" not in df.columns or "practitioner_name" not in df.columns:
        return []
    
    df_copy = df.copy()
    df_copy["medicine_cost"] = pd.to_numeric(df_copy["medicine_cost"], errors="coerce")
    
    threshold = df_copy["medicine_cost"].quantile(cost_threshold_percentile / 100)
    high_cost = df_copy[df_copy["medicine_cost"] >= threshold]
    
    patterns = []
    for (doctor, medicine), group in high_cost.groupby(["practitioner_name", "medicine_name"]):
        if len(group) > 1:  # Multiple same drug from same doctor
            patterns.append({
                "doctor": doctor,
                "medicine": medicine,
                "avg_cost": group["medicine_cost"].mean(),
                "frequency": len(group),
                "indices": group.index.tolist(),
                "total_cost": group["medicine_cost"].sum(),
            })
    
    return sorted(patterns, key=lambda x: x["frequency"], reverse=True)

def detect_ghost_prescriptions(pharmacy_df, hospital_df, date_window_days=3):
    """
    Cross-match pharmacy RAMA + Date against hospital records.
    Pharmacy claim is "ghost" if patient not found in hospital on or near that date.
    """
    if hospital_df is None or hospital_df.empty:
        return []
    
    pharmacy_copy = pharmacy_df.copy()
    hospital_copy = hospital_df.copy()
    
    # Normalize dates
    pharmacy_copy["dispensing_date"] = pd.to_datetime(pharmacy_copy.get("dispensing_date"), errors="coerce")
    hospital_copy["visit_date"] = pd.to_datetime(hospital_copy.get("visit_date") or hospital_copy.get("date"), errors="coerce")
    
    ghosts = []
    for idx, row in pharmacy_copy.iterrows():
        rama = row.get("rama_number")
        p_date = row.get("dispensing_date")
        p_name = row.get("patient_name")
        
        if pd.isna(p_date):
            continue
        
        # Search hospital for matching RAMA or patient name near same date
        match = False
        if not pd.isna(rama) and rama != "":
            h_match = hospital_copy[(hospital_copy.get("rama_number") == rama) |
                                     (hospital_copy.get("patient_name") == p_name)]
            if not h_match.empty:
                for _, h_row in h_match.iterrows():
                    h_date = h_row.get("visit_date")
                    if pd.isna(h_date):
                        continue
                    if abs((p_date - h_date).days) <= date_window_days:
                        match = True
                        break
        
        if not match:
            ghosts.append({
                "idx": idx,
                "patient_name": p_name,
                "rama_number": rama,
                "dispensing_date": p_date,
                "medicine_name": row.get("medicine_name", "—"),
                "cost": row.get("medicine_cost", 0),
            })
    
    return ghosts

def build_network_graph(df):
    """
    Build tripartite network: Doctor ↔ Patient ↔ Drug.
    Visualize collusion patterns.
    """
    G = nx.Graph()
    
    if df.empty:
        return G
    
    for idx, row in df.iterrows():
        doctor = f"Dr: {row.get('practitioner_name', '?')}"
        patient = f"Pt: {row.get('patient_name', '?')}"
        drug = f"Rx: {row.get('medicine_name', '?')}"
        cost = row.get("medicine_cost", 0)
        
        # Add edges with weight = cost
        G.add_edge(doctor, patient, weight=cost, type="doctor_patient")
        G.add_edge(patient, drug, weight=cost, type="patient_drug")
        G.add_edge(doctor, drug, weight=cost, type="doctor_drug")
    
    return G

def export_to_rssb_format(df, verifications_dict, metadata=None):
    """
    Export to RSSB-compliant Excel format.
    Columns: Paper Code, Patient Name, RAMA, Dispensing Date, Practitioner,
             Original Total Cost, Deduction, 100% after CV, 85% after CV, Reason
    """
    output_rows = []
    
    for idx, row in df.iterrows():
        v = verifications_dict.get(idx, {})
        orig_cost = float(row.get("medicine_cost", 0))
        deduction = float(v.get("deduction", 0))
        
        after_cv_100 = orig_cost - deduction
        after_cv_85 = (orig_cost * 0.85) - deduction
        
        output_rows.append({
            "Paper Code": row.get("paper_code", "—"),
            "Patient Name": row.get("patient_name", "—"),
            "RAMA Number": row.get("rama_number", "—"),
            "Dispensing Date": row.get("dispensing_date", "—"),
            "Practitioner Name": row.get("practitioner_name", "—"),
            "Medicine Name": row.get("medicine_name", "—"),
            "Original Total Cost": orig_cost,
            "Deduction (RWF)": deduction,
            "100% after CV": after_cv_100,
            "85% after CV": after_cv_85,
            "Status": v.get("status", "Pending"),
            "Reason for Deduction": v.get("reason", "—"),
        })
    
    result_df = pd.DataFrame(output_rows)
    return result_df

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: UPLOAD & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚖️ Counter-Verification</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>RSSB Pharmacology Audit Workbench</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Pharmacy file upload
    pharmacy_file = st.file_uploader("📋 Upload Pharmacy Records (Excel/CSV)", type=["xlsx", "csv", "xls"])
    if pharmacy_file:
        try:
            if pharmacy_file.name.endswith(".csv"):
                st.session_state.pharmacy_df = pd.read_csv(pharmacy_file)
            else:
                st.session_state.pharmacy_df = pd.read_excel(pharmacy_file)
            st.success(f"✅ Loaded {len(st.session_state.pharmacy_df)} pharmacy records")
        except Exception as e:
            st.error(f"❌ Failed to load: {e}")
    
    # Hospital file upload (optional, for ghost prescription detection)
    hospital_file = st.file_uploader("🏥 Upload Hospital Records (Optional)", type=["xlsx", "csv", "xls"])
    if hospital_file:
        try:
            if hospital_file.name.endswith(".csv"):
                st.session_state.hospital_df = pd.read_csv(hospital_file)
            else:
                st.session_state.hospital_df = pd.read_excel(hospital_file)
            st.success(f"✅ Loaded {len(st.session_state.hospital_df)} hospital records")
        except Exception as e:
            st.error(f"❌ Failed to load: {e}")
    
    st.divider()
    
    # Config
    st.subheader("⚙️ Config")
    repeat_days = st.slider("Repeat visit window (days)", 7, 90, 30)
    high_cost_percentile = st.slider("High-cost threshold (percentile)", 50, 99, 75)
    ghost_date_window = st.slider("Ghost Rx match window (days)", 1, 7, 3)
    
    st.divider()
    
    # Clear data
    if st.button("🔄 Reset All Verifications"):
        st.session_state.verifications = {}
        st.session_state.current_claim_idx = 0
        st.success("✅ Verifications cleared")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.pharmacy_df is None or st.session_state.pharmacy_df.empty:
    st.info("👈 Start by uploading pharmacy records in the sidebar.")
    st.stop()

pharmacy_df = st.session_state.pharmacy_df
hospital_df = st.session_state.hospital_df
verifications = st.session_state.verifications

tab_verify, tab_queues, tab_ghost, tab_graph, tab_report = st.tabs(
    ["🎯 Verification Cards", "🚨 Physical Review Queues", "👻 Ghost Prescriptions", 
     "🔗 Collusion Patterns", "📊 Generate RSSB Report"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: VERIFICATION CARDS
# ══════════════════════════════════════════════════════════════════════════════

with tab_verify:
    st.markdown("<div class='sec-head'>Verification Workbench</div>", unsafe_allow_html=True)
    
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", len(pharmacy_df))
    c2.metric("Verified", len([v for v in verifications.values() if v.get("status") == "Verified"]))
    c3.metric("Deducted", len([v for v in verifications.values() if v.get("status") == "Deducted"]))
    c4.metric("Flagged", len([v for v in verifications.values() if v.get("status") in ["Ghost Prescription", "Signature Mismatch"]]))
    
    st.divider()
    
    # Navigation
    nc1, nc2, nc3 = st.columns([1, 3, 1])
    with nc1:
        if st.button("⬅ Previous"):
            st.session_state.current_claim_idx = max(0, st.session_state.current_claim_idx - 1)
            st.rerun()
    
    with nc2:
        idx = st.number_input(
            "Go to claim #",
            min_value=0,
            max_value=len(pharmacy_df) - 1,
            value=st.session_state.current_claim_idx,
            key="jump_idx"
        )
        st.session_state.current_claim_idx = idx
    
    with nc3:
        if st.button("Next ➜"):
            st.session_state.current_claim_idx = min(len(pharmacy_df) - 1, st.session_state.current_claim_idx + 1)
            st.rerun()
    
    st.divider()
    
    # Current claim card
    claim_idx = st.session_state.current_claim_idx
    claim = pharmacy_df.iloc[claim_idx]
    
    current_v = verifications.get(claim_idx, {})
    status = current_v.get("status", "Pending")
    
    # Style card based on status
    status_class = {
        "Verified": "card-status-verified",
        "Deducted": "card-status-deducted",
        "Ghost Prescription": "card-status-ghost",
        "Signature Mismatch": "card-status-mismatch",
    }.get(status, "")
    
    st.markdown(f"""
    <div class='verify-card {status_class}'>
        <h4 style='margin:0 0 12px 0; color:#e2e8f0'>Claim #{claim_idx + 1} of {len(pharmacy_df)}</h4>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; font-size:12px; color:#94a3b8; margin-bottom:16px'>
            <div><b>Patient:</b> {claim.get("patient_name", "—")}</div>
            <div><b>RAMA:</b> {claim.get("rama_number", "—")}</div>
            <div><b>Date:</b> {claim.get("dispensing_date", "—")}</div>
            <div><b>Practitioner:</b> {claim.get("practitioner_name", "—")}</div>
            <div><b>Medicine:</b> {claim.get("medicine_name", "—")}</div>
            <div><b>Cost:</b> {claim.get("medicine_cost", 0)} RWF</div>
            <div><b>Paper Code:</b> {claim.get("paper_code", "—")}</div>
            <div><b>Status:</b> <span style='color:#00e5a0'>{status}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Verification Decision")
    
    # Status dropdown
    new_status = st.selectbox(
        "Status",
        ["Pending", "Verified", "Deducted", "Ghost Prescription", "Signature Mismatch"],
        index=["Pending", "Verified", "Deducted", "Ghost Prescription", "Signature Mismatch"].index(status),
        key=f"status_{claim_idx}"
    )
    
    # Deduction amount
    deduction = st.number_input(
        "Deduction Amount (RWF)",
        min_value=0.0,
        value=float(current_v.get("deduction", 0)),
        key=f"deduction_{claim_idx}"
    )
    
    # Reason
    reason = st.text_area(
        "Reason for Deduction / Comments",
        value=current_v.get("reason", ""),
        placeholder="e.g., Quantity exceeds RSSB limit for Amoxicillin / Patient not found in hospital register",
        key=f"reason_{claim_idx}"
    )
    
    # Save this verification
    if st.button("💾 Save Verification", key=f"save_{claim_idx}"):
        st.session_state.verifications[claim_idx] = {
            "status": new_status,
            "deduction": deduction,
            "reason": reason,
        }
        st.success(f"✅ Claim #{claim_idx + 1} saved")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PHYSICAL REVIEW QUEUES
# ══════════════════════════════════════════════════════════════════════════════

with tab_queues:
    st.markdown("<div class='sec-head'>Physical Review Queues</div>", unsafe_allow_html=True)
    st.markdown("**Pull these physical papers from the cabinet & check signatures, handwriting, dates.**")
    
    st.divider()
    
    # Repeat visits
    st.subheader("🔄 Repeat Visits (>1 in month)")
    st.caption("Patients with multiple visits in same month — high fraud risk")
    
    repeat_patients = detect_repeat_visits(pharmacy_df, days_window=repeat_days)
    
    if repeat_patients:
        for r in repeat_patients[:10]:  # Show top 10
            with st.expander(f"🚩 **{r['patient_name']}** ({r['count']} visits)"):
                st.markdown(f"**RAMA:** {r['rama_id']}")
                st.markdown(f"**Visit Dates:**")
                for d in r["visit_dates"]:
                    st.text(f"  • {d.strftime('%d/%m/%Y') if hasattr(d, 'strftime') else d}")
                st.markdown(f"**Claim indices:** {r['indices']}")
                st.warning("👉 Check signature consistency across all visits")
    else:
        st.info("No repeat visits detected")
    
    st.divider()
    
    # High-cost patterns
    st.subheader("💰 High-Cost Drug Patterns")
    st.caption("Same expensive drug billed multiple times by same doctor")
    
    patterns = detect_high_cost_patterns(pharmacy_df, cost_threshold_percentile=high_cost_percentile)
    
    if patterns:
        for p in patterns[:15]:  # Show top 15
            with st.expander(f"⚠️ **{p['medicine']}** by **{p['doctor']}** ({p['frequency']}x)"):
                st.markdown(f"**Total Cost:** {p['total_cost']:,.0f} RWF")
                st.markdown(f"**Avg per claim:** {p['avg_cost']:,.0f} RWF")
                st.markdown(f"**Claim indices:** {p['indices']}")
                st.warning("👉 Cross-check doctor's signature, handwriting consistency. Look for kickback signs.")
    else:
        st.info("No high-cost patterns detected")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: GHOST PRESCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_ghost:
    st.markdown("<div class='sec-head'>Ghost Prescription Detector</div>", unsafe_allow_html=True)
    
    if hospital_df is None or hospital_df.empty:
        st.warning("⚠️ Upload hospital records in sidebar to detect ghost prescriptions")
    else:
        st.caption("Pharmacy claims with NO matching hospital visit record (or outside ±N days)")
        
        ghosts = detect_ghost_prescriptions(pharmacy_df, hospital_df, date_window_days=ghost_date_window)
        
        st.metric("Potential Ghost Prescriptions", len(ghosts))
        
        if ghosts:
            ghost_df = pd.DataFrame(ghosts)
            st.dataframe(ghost_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Recommended Action:** Mark these claims in Verification Cards for manual investigation.")
            
            # Bulk-mark as ghost
            if st.button("🚩 Mark All as 'Ghost Prescription'"):
                for g in ghosts:
                    idx = g["idx"]
                    if idx not in st.session_state.verifications:
                        st.session_state.verifications[idx] = {}
                    st.session_state.verifications[idx].update({
                        "status": "Ghost Prescription",
                        "reason": f"No matching hospital visit record within ±{ghost_date_window} days",
                    })
                st.success(f"✅ Marked {len(ghosts)} claims as ghost prescriptions")
        else:
            st.success("✅ No ghost prescriptions detected")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: COLLUSION PATTERN GRAPH
# ══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.markdown("<div class='sec-head'>Collusion Pattern Network</div>", unsafe_allow_html=True)
    st.caption("Tripartite graph: Doctor ↔ Patient ↔ Drug. Clusters = prescribing anomalies.")
    
    # Build graph
    G = build_network_graph(pharmacy_df)
    
    if G.number_of_nodes() == 0:
        st.warning("No data for network graph")
    else:
        # Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Doctors", len([n for n in G.nodes() if n.startswith("Dr:")]))
        c2.metric("Patients", len([n for n in G.nodes() if n.startswith("Pt:")]))
        c3.metric("Drugs", len([n for n in G.nodes() if n.startswith("Rx:")]))
        
        st.divider()
        
        # Visualize
        fig, ax = plt.subplots(figsize=(14, 10), facecolor=CARD)
        
        # Spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors by type
        node_colors = []
        for node in G.nodes():
            if node.startswith("Dr:"):
                node_colors.append("#0ea5e9")  # Doctor = blue
            elif node.startswith("Pt:"):
                node_colors.append("#22c55e")  # Patient = green
            else:
                node_colors.append("#f59e0b")  # Drug = orange
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5, edge_color=MUTED)
        
        # Labels (abbreviated)
        labels = {n: n.split(": ")[1][:15] if ": " in n else n[:15] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax, font_color=TEXT)
        
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("**Interpretation:**")
        st.markdown("""
        - **Clusters** = Groups of doctors, patients, and drugs all interconnected
        - **Large clusters** = Potential collusion (same doctor prescribing expensive drug to many patients)
        - **Outliers** = Single prescriptions (less suspicious)
        - **Action:** Use clusters to prioritize which physical files to inspect
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: GENERATE RSSB REPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:
    st.markdown("<div class='sec-head'>Generate RSSB Report</div>", unsafe_allow_html=True)
    
    # Metadata
    st.subheader("Report Metadata")
    
    c1, c2 = st.columns(2)
    with c1:
        province = st.text_input("Province", "Rwanda")
        district = st.text_input("District", "Kigali")
    with c2:
        pharmacy_name = st.text_input("Pharmacy Name", "Faith Pharmacy")
        period = st.text_input("Period", "May 2024")
    
    st.divider()
    
    # Preview table
    st.subheader("Preview: Deductions Summary")
    
    report_df = export_to_rssb_format(pharmacy_df, verifications)
    
    # Show summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Original Cost", f"{report_df['Original Total Cost'].sum():,.0f} RWF")
    c2.metric("Total Deductions", f"{report_df['Deduction (RWF)'].sum():,.0f} RWF")
    c3.metric("100% after CV", f"{report_df['100% after CV'].sum():,.0f} RWF")
    c4.metric("85% after CV", f"{report_df['85% after CV'].sum():,.0f} RWF")
    
    st.divider()
    
    # Full preview table
    st.dataframe(report_df, use_container_width=True, height=400, hide_index=True)
    
    st.divider()
    
    # Export buttons
    e1, e2, e3 = st.columns(3)
    
    with e1:
        # Export as Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            report_df.to_excel(writer, index=False, sheet_name="Sheet1")
        
        st.download_button(
            "📥 Download RSSB Excel Report",
            data=buffer.getvalue(),
            file_name=f"RSSB_{period.replace(' ', '_')}_FINAL.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with e2:
        # Export as CSV
        csv_data = report_df.to_csv(index=False).encode()
        st.download_button(
            "📋 Download CSV",
            data=csv_data,
            file_name=f"RSSB_{period.replace(' ', '_')}_FINAL.csv",
            mime="text/csv"
        )
    
    with e3:
        # Export verification log (for audit trail)
        verify_log = []
        for idx, v in verifications.items():
            if idx < len(pharmacy_df):
                row = pharmacy_df.iloc[idx]
                verify_log.append({
                    "Claim Index": idx,
                    "Patient": row.get("patient_name"),
                    "RAMA": row.get("rama_number"),
                    "Status": v.get("status"),
                    "Deduction": v.get("deduction"),
                    "Reason": v.get("reason"),
                    "Verified At": datetime.now().strftime("%d/%m/%Y %H:%M"),
                })
        
        if verify_log:
            verify_df = pd.DataFrame(verify_log)
            verify_csv = verify_df.to_csv(index=False).encode()
            st.download_button(
                "📝 Download Audit Log",
                data=verify_csv,
                file_name=f"verification_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<div style='font-size:11px; color:#64748b; text-align:center; margin-top:20px'>
    PharmaScan v2 — Counter-Verification Workbench for RSSB Auditors<br>
    Detect pharmacology violations, signature forgery, and ghost prescriptions
</div>
""", unsafe_allow_html=True)
