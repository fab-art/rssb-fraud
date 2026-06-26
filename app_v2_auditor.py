"""
PharmaScan v3 — Counter-Verification Workbench (RSSB Edition)

Purpose: Interactive verification interface with data preparation layer.
Workflow: Upload → Map Columns → Verify Claims → Auto-generate RSSB report.

Install:
    pip install -r requirements.txt

Run:
    streamlit run app_v3_with_dataprep.py
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
    page_title="PharmaScan v3 — CV Workbench",
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

.mapping-row { background: #111720; border: 1px solid #1e2a38; border-radius: 8px; padding: 12px; margin-bottom: 8px; }
.mapping-success { border-left: 4px solid #22c55e; }
.mapping-warning { border-left: 4px solid #f59e0b; }
.mapping-error { border-left: 4px solid #ef4444; }

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
    if "pharmacy_df_raw" not in st.session_state:
        st.session_state.pharmacy_df_raw = None
    if "pharmacy_df" not in st.session_state:
        st.session_state.pharmacy_df = None
    if "hospital_df_raw" not in st.session_state:
        st.session_state.hospital_df_raw = None
    if "hospital_df" not in st.session_state:
        st.session_state.hospital_df = None
    if "pharmacy_mapping" not in st.session_state:
        st.session_state.pharmacy_mapping = {}
    if "hospital_mapping" not in st.session_state:
        st.session_state.hospital_mapping = {}
    if "verifications" not in st.session_state:
        st.session_state.verifications = {}
    if "current_claim_idx" not in st.session_state:
        st.session_state.current_claim_idx = 0
    if "data_prep_done" not in st.session_state:
        st.session_state.data_prep_done = False

init_session_state()

# ══════════════════════════════════════════════════════════════════════════════
# REQUIRED FIELDS SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

PHARMACY_REQUIRED = {
    "paper_code": "Unique ID for each claim/voucher",
    "patient_name": "Full name of patient/recipient",
    "rama_number": "Patient's insurance/affiliation ID",
    "dispensing_date": "Date medication was dispensed",
    "practitioner_name": "Name of prescribing doctor/practitioner",
    "medicine_name": "Name of medication/drug dispensed",
    "medicine_cost": "Cost of medication in RWF",
}

HOSPITAL_REQUIRED = {
    "patient_name": "Patient name",
    "rama_number": "Patient's ID/affiliation number",
    "visit_date": "Hospital visit date",
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_date(val):
    """Safely parse date string to datetime."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            pass
    try:
        return pd.to_datetime(val)
    except:
        return None

def find_best_column_match(target_field, available_columns):
    """
    Find best matching column for a target field using fuzzy matching.
    Returns (column_name, match_score) or (None, 0)
    """
    if not available_columns:
        return None, 0
    
    target_lower = target_field.lower()
    col_lower = [c.lower() for c in available_columns]
    
    # Exact match
    for i, c in enumerate(col_lower):
        if c == target_lower:
            return available_columns[i], 100
    
    # Keyword matching
    keywords = {
        "paper_code": ["paper", "code", "voucher", "id", "claim"],
        "patient_name": ["patient", "name", "recipient", "client"],
        "rama_number": ["rama", "affiliation", "insurance", "card", "member"],
        "dispensing_date": ["dispensing", "date", "issue", "dispense"],
        "practitioner_name": ["practitioner", "doctor", "provider", "prescriber"],
        "medicine_name": ["medicine", "drug", "medication", "product"],
        "medicine_cost": ["cost", "price", "amount", "fee", "total"],
        "visit_date": ["visit", "date", "admission", "appointment"],
    }
    
    target_keywords = keywords.get(target_field, [target_lower.split()])
    
    best_match = None
    best_score = 0
    
    for col, col_name in zip(col_lower, available_columns):
        score = sum(1 for kw in target_keywords if kw in col)
        if score > best_score:
            best_score = score
            best_match = col_name
    
    return best_match, min(best_score * 25, 90) if best_score > 0 else 0

def validate_and_transform(df, mapping, required_fields, data_type="pharmacy"):
    """
    Validate mapped columns, detect data types, and transform.
    Returns: (transformed_df, validation_report)
    """
    report = {
        "status": "success",
        "errors": [],
        "warnings": [],
        "rows_loaded": len(df),
        "field_stats": {}
    }
    
    transformed = pd.DataFrame()
    
    for system_field, original_col in mapping.items():
        if original_col not in df.columns:
            report["errors"].append(f"Column '{original_col}' not found in file")
            continue
        
        col_data = df[original_col].copy()
        
        # Handle special transformations
        if system_field in ["dispensing_date", "visit_date"]:
            col_data = col_data.apply(parse_date)
            null_count = col_data.isna().sum()
            if null_count > 0:
                report["warnings"].append(f"{system_field}: {null_count} unparseable dates")
        
        elif system_field == "medicine_cost":
            col_data = pd.to_numeric(col_data, errors="coerce")
            null_count = col_data.isna().sum()
            if null_count > 0:
                report["warnings"].append(f"{system_field}: {null_count} non-numeric values")
        
        else:
            col_data = col_data.astype(str).str.strip()
        
        transformed[system_field] = col_data
        
        # Stats
        report["field_stats"][system_field] = {
            "non_null": transformed[system_field].notna().sum(),
            "null": transformed[system_field].isna().sum(),
            "type": str(transformed[system_field].dtype)
        }
    
    # Validate required fields present
    for req_field in required_fields:
        if req_field not in transformed.columns:
            report["errors"].append(f"Required field missing: {req_field}")
        elif transformed[req_field].isna().sum() == len(transformed):
            report["errors"].append(f"Required field '{req_field}' is completely empty")
    
    if report["errors"]:
        report["status"] = "error"
    elif report["warnings"]:
        report["status"] = "warning"
    
    return transformed, report

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div style='font-family:Syne;font-size:22px;font-weight:800;color:#e2e8f0'>⚖️ PharmaScan v3</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#64748b;margin-bottom:20px'>Counter-Verification Workbench</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Pharmacy file upload
    pharmacy_file = st.file_uploader("📋 Upload Pharmacy Records", type=["xlsx", "csv", "xls"])
    if pharmacy_file:
        try:
            if pharmacy_file.name.endswith(".csv"):
                df = pd.read_csv(pharmacy_file)
            else:
                df = pd.read_excel(pharmacy_file)
            st.session_state.pharmacy_df_raw = df
            st.session_state.pharmacy_mapping = {}
            st.session_state.data_prep_done = False
            st.success(f"✅ Loaded {len(df)} records")
        except Exception as e:
            st.error(f"❌ Load error: {e}")
    
    # Hospital file upload
    hospital_file = st.file_uploader("🏥 Upload Hospital Records (Optional)", type=["xlsx", "csv", "xls"])
    if hospital_file:
        try:
            if hospital_file.name.endswith(".csv"):
                df = pd.read_csv(hospital_file)
            else:
                df = pd.read_excel(hospital_file)
            st.session_state.hospital_df_raw = df
            st.session_state.hospital_mapping = {}
            st.success(f"✅ Loaded {len(df)} records")
        except Exception as e:
            st.error(f"❌ Load error: {e}")
    
    st.divider()
    
    # Clear data
    if st.button("🔄 Reset All"):
        st.session_state.pharmacy_df_raw = None
        st.session_state.pharmacy_df = None
        st.session_state.hospital_df_raw = None
        st.session_state.hospital_df = None
        st.session_state.pharmacy_mapping = {}
        st.session_state.hospital_mapping = {}
        st.session_state.verifications = {}
        st.session_state.data_prep_done = False
        st.success("✅ All data cleared")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN: DATA PREPARATION IF NEEDED
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.pharmacy_df_raw is None:
    st.info("👈 Upload pharmacy records in the sidebar to start")
    st.stop()

if not st.session_state.data_prep_done:
    st.markdown("## 📊 Data Preparation: Column Mapping")
    st.markdown("Map your Excel columns to the system fields required for verification.")
    
    st.divider()
    
    # PHARMACY MAPPING
    st.subheader("📋 Pharmacy Records Mapping")
    st.caption(f"Detected {len(st.session_state.pharmacy_df_raw.columns)} columns in your file")
    
    with st.expander("📂 Available Columns", expanded=False):
        st.code(", ".join(st.session_state.pharmacy_df_raw.columns.tolist()))
    
    pharmacy_mapping = {}
    
    # Auto-suggest mappings
    auto_mappings = {}
    for req_field in PHARMACY_REQUIRED.keys():
        best_col, score = find_best_column_match(req_field, st.session_state.pharmacy_df_raw.columns.tolist())
        if best_col:
            auto_mappings[req_field] = best_col
    
    # Mapping UI
    mapping_cols = st.columns([2, 2, 1])
    mapping_cols[0].markdown("**System Field**")
    mapping_cols[1].markdown("**Your Column**")
    mapping_cols[2].markdown("**Status**")
    
    st.divider()
    
    for system_field, description in PHARMACY_REQUIRED.items():
        c1, c2, c3 = st.columns([2, 2, 1])
        
        with c1:
            st.markdown(f"**{system_field}**")
            st.caption(description)
        
        with c2:
            suggested = auto_mappings.get(system_field)
            selected_col = st.selectbox(
                "Select column",
                [None] + st.session_state.pharmacy_df_raw.columns.tolist(),
                index=([None] + st.session_state.pharmacy_df_raw.columns.tolist()).index(suggested) if suggested else 0,
                key=f"pharm_map_{system_field}",
                label_visibility="collapsed"
            )
            pharmacy_mapping[system_field] = selected_col
        
        with c3:
            if selected_col:
                st.markdown("✅")
            else:
                st.markdown("⚠️")
    
    # Validate pharmacy mapping
    pharmacy_complete = all(v is not None for v in pharmacy_mapping.values())
    
    st.divider()
    
    # HOSPITAL MAPPING (if file exists)
    if st.session_state.hospital_df_raw is not None:
        st.subheader("🏥 Hospital Records Mapping (Optional)")
        st.caption(f"Detected {len(st.session_state.hospital_df_raw.columns)} columns")
        
        hospital_mapping = {}
        
        # Auto-suggest
        auto_mappings_hosp = {}
        for req_field in HOSPITAL_REQUIRED.keys():
            best_col, score = find_best_column_match(req_field, st.session_state.hospital_df_raw.columns.tolist())
            if best_col:
                auto_mappings_hosp[req_field] = best_col
        
        # Mapping UI
        mapping_cols = st.columns([2, 2, 1])
        mapping_cols[0].markdown("**System Field**")
        mapping_cols[1].markdown("**Your Column**")
        mapping_cols[2].markdown("**Status**")
        
        st.divider()
        
        for system_field, description in HOSPITAL_REQUIRED.items():
            c1, c2, c3 = st.columns([2, 2, 1])
            
            with c1:
                st.markdown(f"**{system_field}**")
                st.caption(description)
            
            with c2:
                suggested = auto_mappings_hosp.get(system_field)
                selected_col = st.selectbox(
                    "Select column",
                    [None] + st.session_state.hospital_df_raw.columns.tolist(),
                    index=([None] + st.session_state.hospital_df_raw.columns.tolist()).index(suggested) if suggested else 0,
                    key=f"hosp_map_{system_field}",
                    label_visibility="collapsed"
                )
                hospital_mapping[system_field] = selected_col
            
            with c3:
                if selected_col:
                    st.markdown("✅")
                else:
                    st.markdown("⚠️")
    else:
        hospital_mapping = {}
    
    st.divider()
    
    # PREVIEW & VALIDATION
    if pharmacy_complete:
        st.subheader("✅ Data Preview & Validation")
        
        # Transform pharmacy
        pharm_transformed, pharm_report = validate_and_transform(
            st.session_state.pharmacy_df_raw,
            pharmacy_mapping,
            PHARMACY_REQUIRED.keys(),
            "pharmacy"
        )
        
        # Show validation result
        if pharm_report["status"] == "error":
            st.error(f"❌ Validation errors: {pharm_report['errors']}")
            st.info("⚠️ Fix errors above before proceeding")
        elif pharm_report["status"] == "warning":
            st.warning(f"⚠️ Minor warnings (can proceed): {pharm_report['warnings']}")
            st.info("💡 A small number of records have incomplete/unparseable values. You can proceed — they'll be handled gracefully during analysis.")
        else:
            st.success("✅ All validations passed!")
        
        # Preview table
        st.markdown("**Transformed Data Preview (first 5 rows):**")
        st.dataframe(pharm_transformed.head(5), use_container_width=True, hide_index=True)
        
        # Field stats
        with st.expander("📊 Field Statistics"):
            stats_df = pd.DataFrame([
                {
                    "Field": field,
                    "Non-Null": stats["non_null"],
                    "Null": stats["null"],
                    "Type": stats["type"]
                }
                for field, stats in pharm_report["field_stats"].items()
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Hospital validation
        if hospital_mapping and any(hospital_mapping.values()):
            hosp_transformed, hosp_report = validate_and_transform(
                st.session_state.hospital_df_raw,
                {k: v for k, v in hospital_mapping.items() if v is not None},
                HOSPITAL_REQUIRED.keys(),
                "hospital"
            )
            
            if hosp_report["status"] != "error":
                st.success("✅ Hospital records validated")
        
        # CONFIRM & PROCEED
        st.divider()
        
        # Allow proceeding if no errors (warnings are OK)
        can_proceed = pharm_report["status"] != "error"
        
        confirm_col, _ = st.columns([1, 3])
        with confirm_col:
            if st.button("✅ Confirm & Proceed to Verification", type="primary", use_container_width=True, disabled=not can_proceed):
                st.session_state.pharmacy_mapping = pharmacy_mapping
                st.session_state.hospital_mapping = hospital_mapping
                st.session_state.pharmacy_df = pharm_transformed
                
                if hospital_mapping and any(hospital_mapping.values()):
                    st.session_state.hospital_df = hosp_transformed
                
                st.session_state.data_prep_done = True
                st.success("✅ Data preparation complete! Reloading...")
                st.rerun()
    else:
        st.warning("⚠️ Please map all required fields to proceed")

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION INTERFACE (only after data prep)
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.data_prep_done:
    st.stop()

pharmacy_df = st.session_state.pharmacy_df
hospital_df = st.session_state.hospital_df
verifications = st.session_state.verifications

# Config
with st.sidebar:
    st.divider()
    st.subheader("⚙️ Config")
    repeat_days = st.slider("Repeat visit window (days)", 7, 90, 30)
    high_cost_percentile = st.slider("High-cost threshold (%ile)", 50, 99, 75)
    ghost_date_window = st.slider("Ghost match window (days)", 1, 7, 3)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════

def detect_repeat_visits(df, days_window=30):
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
    if "medicine_cost" not in df.columns or "practitioner_name" not in df.columns:
        return []
    
    df_copy = df.copy()
    df_copy["medicine_cost"] = pd.to_numeric(df_copy["medicine_cost"], errors="coerce")
    
    threshold = df_copy["medicine_cost"].quantile(cost_threshold_percentile / 100)
    high_cost = df_copy[df_copy["medicine_cost"] >= threshold]
    
    patterns = []
    for (doctor, medicine), group in high_cost.groupby(["practitioner_name", "medicine_name"]):
        if len(group) > 1:
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
    if hospital_df is None or hospital_df.empty:
        return []
    
    pharmacy_copy = pharmacy_df.copy()
    hospital_copy = hospital_df.copy()
    
    pharmacy_copy["dispensing_date"] = pd.to_datetime(pharmacy_copy.get("dispensing_date"), errors="coerce")
    hospital_copy["visit_date"] = pd.to_datetime(hospital_copy.get("visit_date"), errors="coerce")
    
    ghosts = []
    for idx, row in pharmacy_copy.iterrows():
        rama = row.get("rama_number")
        p_date = row.get("dispensing_date")
        p_name = row.get("patient_name")
        
        if pd.isna(p_date):
            continue
        
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
    G = nx.Graph()
    
    if df.empty:
        return G
    
    for idx, row in df.iterrows():
        doctor = f"Dr: {row.get('practitioner_name', '?')}"
        patient = f"Pt: {row.get('patient_name', '?')}"
        drug = f"Rx: {row.get('medicine_name', '?')}"
        cost = row.get("medicine_cost", 0)
        
        G.add_edge(doctor, patient, weight=cost, type="doctor_patient")
        G.add_edge(patient, drug, weight=cost, type="patient_drug")
        G.add_edge(doctor, drug, weight=cost, type="doctor_drug")
    
    return G

def export_to_rssb_format(df, verifications_dict, metadata=None):
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
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_verify, tab_queues, tab_ghost, tab_graph, tab_report = st.tabs(
    ["🎯 Verification Cards", "🚨 Physical Review Queues", "👻 Ghost Prescriptions", 
     "🔗 Collusion Patterns", "📊 Generate RSSB Report"]
)

# ═══════════════════ TAB 1: VERIFICATION CARDS ═══════════════════

with tab_verify:
    st.markdown("<div class='sec-head'>Verification Workbench</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", len(pharmacy_df))
    c2.metric("Verified", len([v for v in verifications.values() if v.get("status") == "Verified"]))
    c3.metric("Deducted", len([v for v in verifications.values() if v.get("status") == "Deducted"]))
    c4.metric("Flagged", len([v for v in verifications.values() if v.get("status") in ["Ghost Prescription", "Signature Mismatch"]]))
    
    st.divider()
    
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
    
    claim_idx = st.session_state.current_claim_idx
    claim = pharmacy_df.iloc[claim_idx]
    
    current_v = verifications.get(claim_idx, {})
    status = current_v.get("status", "Pending")
    
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
    
    new_status = st.selectbox(
        "Status",
        ["Pending", "Verified", "Deducted", "Ghost Prescription", "Signature Mismatch"],
        index=["Pending", "Verified", "Deducted", "Ghost Prescription", "Signature Mismatch"].index(status),
        key=f"status_{claim_idx}"
    )
    
    deduction = st.number_input(
        "Deduction Amount (RWF)",
        min_value=0.0,
        value=float(current_v.get("deduction", 0)),
        key=f"deduction_{claim_idx}"
    )
    
    reason = st.text_area(
        "Reason for Deduction / Comments",
        value=current_v.get("reason", ""),
        placeholder="e.g., Quantity exceeds RSSB limit / Patient not found in hospital",
        key=f"reason_{claim_idx}"
    )
    
    if st.button("💾 Save Verification", key=f"save_{claim_idx}"):
        st.session_state.verifications[claim_idx] = {
            "status": new_status,
            "deduction": deduction,
            "reason": reason,
        }
        st.success(f"✅ Claim #{claim_idx + 1} saved")

# ═══════════════════ TAB 2: PHYSICAL REVIEW QUEUES ═══════════════════

with tab_queues:
    st.markdown("<div class='sec-head'>Physical Review Queues</div>", unsafe_allow_html=True)
    st.markdown("**Pull these physical papers from the cabinet & check signatures, handwriting, dates.**")
    
    st.divider()
    
    st.subheader("🔄 Repeat Visits (>1 in month)")
    st.caption("Patients with multiple visits in same month — high fraud risk")
    
    repeat_patients = detect_repeat_visits(pharmacy_df, days_window=repeat_days)
    
    if repeat_patients:
        for r in repeat_patients[:10]:
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
    
    st.subheader("💰 High-Cost Drug Patterns")
    st.caption("Same expensive drug billed multiple times by same doctor")
    
    patterns = detect_high_cost_patterns(pharmacy_df, cost_threshold_percentile=high_cost_percentile)
    
    if patterns:
        for p in patterns[:15]:
            with st.expander(f"⚠️ **{p['medicine']}** by **{p['doctor']}** ({p['frequency']}x)"):
                st.markdown(f"**Total Cost:** {p['total_cost']:,.0f} RWF")
                st.markdown(f"**Avg per claim:** {p['avg_cost']:,.0f} RWF")
                st.markdown(f"**Claim indices:** {p['indices']}")
                st.warning("👉 Cross-check doctor's signature, handwriting consistency. Look for kickback signs.")
    else:
        st.info("No high-cost patterns detected")

# ═══════════════════ TAB 3: GHOST PRESCRIPTIONS ═══════════════════

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

# ═══════════════════ TAB 4: COLLUSION PATTERN GRAPH ═══════════════════

with tab_graph:
    st.markdown("<div class='sec-head'>Collusion Pattern Network</div>", unsafe_allow_html=True)
    st.caption("Tripartite graph: Doctor ↔ Patient ↔ Drug. Clusters = prescribing anomalies.")
    
    G = build_network_graph(pharmacy_df)
    
    if G.number_of_nodes() == 0:
        st.warning("No data for network graph")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Doctors", len([n for n in G.nodes() if n.startswith("Dr:")]))
        c2.metric("Patients", len([n for n in G.nodes() if n.startswith("Pt:")]))
        c3.metric("Drugs", len([n for n in G.nodes() if n.startswith("Rx:")]))
        
        st.divider()
        
        fig, ax = plt.subplots(figsize=(14, 10), facecolor=CARD)
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        node_colors = []
        for node in G.nodes():
            if node.startswith("Dr:"):
                node_colors.append("#0ea5e9")
            elif node.startswith("Pt:"):
                node_colors.append("#22c55e")
            else:
                node_colors.append("#f59e0b")
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5, edge_color=MUTED)
        
        labels = {n: n.split(": ")[1][:15] if ": " in n else n[:15] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax, font_color=TEXT)
        
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("**Interpretation:** Clusters = potential collusion. Use to prioritize file audits.")

# ═══════════════════ TAB 5: GENERATE RSSB REPORT ═══════════════════

with tab_report:
    st.markdown("<div class='sec-head'>Generate RSSB Report</div>", unsafe_allow_html=True)
    
    st.subheader("Report Metadata")
    
    c1, c2 = st.columns(2)
    with c1:
        province = st.text_input("Province", "Rwanda")
        district = st.text_input("District", "Kigali")
    with c2:
        pharmacy_name = st.text_input("Pharmacy Name", "Faith Pharmacy")
        period = st.text_input("Period", "May 2024")
    
    st.divider()
    
    st.subheader("Preview: Deductions Summary")
    
    report_df = export_to_rssb_format(pharmacy_df, verifications)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Original Cost", f"{report_df['Original Total Cost'].sum():,.0f} RWF")
    c2.metric("Total Deductions", f"{report_df['Deduction (RWF)'].sum():,.0f} RWF")
    c3.metric("100% after CV", f"{report_df['100% after CV'].sum():,.0f} RWF")
    c4.metric("85% after CV", f"{report_df['85% after CV'].sum():,.0f} RWF")
    
    st.divider()
    
    st.dataframe(report_df, use_container_width=True, height=400, hide_index=True)
    
    st.divider()
    
    e1, e2, e3 = st.columns(3)
    
    with e1:
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
        csv_data = report_df.to_csv(index=False).encode()
        st.download_button(
            "📋 Download CSV",
            data=csv_data,
            file_name=f"RSSB_{period.replace(' ', '_')}_FINAL.csv",
            mime="text/csv"
        )
    
    with e3:
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

st.divider()
st.markdown("""
<div style='font-size:11px; color:#64748b; text-align:center; margin-top:20px'>
    PharmaScan v3 — Counter-Verification Workbench with Data Preparation<br>
    Detect pharmacology violations, signature forgery, and ghost prescriptions
</div>
""", unsafe_allow_html=True)
