"""
PharmaScan — Pharmacy Voucher Intelligence (Streamlit Edition)

Refactored modular version with improved code organization.

Install:
    pip install streamlit pandas matplotlib networkx openpyxl odfpy

Run:
    streamlit run pharmascan_app.py
"""

import warnings

import streamlit as st

from pharmascan.components import setup_plot_style
from pharmascan.config import (
    ACCENT,
    ACCENT2,
    BORDER,
    CARD,
    DARK,
    DANGER,
    MUTED,
    PURPLE,
    TEXT,
    WARN,
)
from pharmascan.processors import load_and_process
from pharmascan.utils import fmt_number

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaScan",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup plot style
setup_plot_style()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">💊 PharmaScan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Pharmacy Voucher Intelligence</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload voucher data",
        type=["csv", "xlsx", "xls", "ods"],
        help="Upload CSV, Excel, or ODS file containing pharmacy voucher records",
    )

    rapid_days = st.slider(
        "Rapid revisit window (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Flag patient visits within this many days as 'rapid revisits'",
    )

    top_n = st.slider(
        "Top N to display",
        min_value=5,
        max_value=50,
        value=15,
        help="Number of top patients/doctors/facilities to show in charts",
    )

    st.divider()

    if uploaded_file:
        st.success(f"📄 {uploaded_file.name}", icon="✅")
    else:
        st.info(
            "👆 Upload a data file to begin analysis.\n\n"
            "**Expected columns:**\n- Patient ID or Name\n- Visit Date\n- Doctor/Practitioner\n- Amount/Cost\n- Facility (optional)",
            icon="ℹ️",
        )


# ── Main application logic ────────────────────────────────────────────────────

if uploaded_file is None:
    st.markdown(
        """
        <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 20px;text-align:center'>
            <div style='font-size:64px;margin-bottom:24px;opacity:.3'>💊</div>
            <h1 style='font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:#e2e8f0;margin-bottom:12px'>
                Welcome to PharmaScan
            </h1>
            <p style='font-size:15px;color:#64748b;max-width:520px;line-height:1.7'>
                Upload pharmacy voucher data to detect patterns, identify repeat patients, 
                and analyze prescribing behavior across facilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Load and process data
try:
    df, col_map, s, repeat_groups, repeat_detail, rapid = load_and_process(
        uploaded_file.getvalue(), uploaded_file.name, rapid_days
    )
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

# Display column mapping info
if col_map:
    mapped_cols = {v: k for k, v in col_map.items() if k != v}
    if mapped_cols:
        st.caption(f"✓ Auto-mapped {len(mapped_cols)} columns")

# Create tabs for different analysis views
tabs = st.tabs(
    [
        "📊 Summary",
        "🔁 Repeat Patients",
        f"⚡ Rapid Revisits ({len(rapid)})",
        "📋 All Records",
    ]
)

# ── SUMMARY TAB ───────────────────────────────────────────────────────────────
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Total Records", f"{s['total_rows']:,}")
    c[1].metric(
        "Unique Patients", f"{s.get('unique_patients', '—'):,}" if "unique_patients" in s else "—"
    )
    c[2].metric(
        "Repeat Patients", f"{s.get('repeat_patients', '—'):,}" if "repeat_patients" in s else "—"
    )
    c[3].metric("Rapid Revisits", str(len(rapid)), delta=f"≤{rapid_days} day window")

    c2 = st.columns(4)
    c2[0].metric(
        "Unique Practitioners",
        f"{s.get('unique_doctors', '—'):,}" if "unique_doctors" in s else "—",
    )
    c2[1].metric("Max Visits / Patient", str(s.get("max_visits", "—")))
    c2[2].metric(
        "Total Cost (RWF)", fmt_number(s["total_amount"]) if "total_amount" in s else "—"
    )
    c2[3].metric("Avg Cost / Visit", fmt_number(s["avg_amount"]) if "avg_amount" in s else "—")

    if "date_min" in s:
        st.markdown(
            f'<p style="font-size:12px;color:#64748b;font-family:monospace;margin:8px 0 20px">'
            f'📅 {s["date_min"]} — {s["date_max"]}</p>',
            unsafe_allow_html=True,
        )

    from pharmascan.components.charts import time_series_chart

    fig_t = time_series_chart(df)
    if fig_t:
        st.pyplot(fig_t, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "top_patients" in s:
            from pharmascan.components.charts import hbar_chart

            td = s["top_patients"].head(top_n)
            colors = [
                DANGER if v >= 10 else WARN if v >= 5 else ACCENT for v in td["visits"]
            ]
            fig = hbar_chart(
                [str(x)[:22] for x in td["id"]],
                td["visits"].tolist(),
                colors,
                "Top Patients by Visit Count",
                "Visits",
            )
            st.pyplot(fig, use_container_width=True)

    with right:
        if "top_doctors" in s:
            from pharmascan.components.charts import hbar_chart

            td = s["top_doctors"].head(top_n)
            fig = hbar_chart(
                [str(x)[:22] for x in td["doctor"]],
                td["visits"].tolist(),
                ACCENT2,
                "Top Practitioners by Visit Volume",
                "Visits",
            )
            st.pyplot(fig, use_container_width=True)

# ── REPEAT PATIENTS TAB ───────────────────────────────────────────────────────
with tabs[1]:
    if not repeat_groups:
        st.info("No repeat patients found in the dataset.", icon="ℹ️")
    else:
        st.metric("Repeat Patient Groups", len(repeat_groups))
        
        # Show summary table
        repeat_df = pd.DataFrame(repeat_groups)
        st.dataframe(
            repeat_df.head(100),
            use_container_width=True,
            column_config={
                "visits": st.column_config.NumberColumn("Visits"),
            },
        )

# ── RAPID REVISITS TAB ────────────────────────────────────────────────────────
with tabs[2]:
    if not rapid:
        st.info(f"No rapid revisits found within {rapid_days} days.", icon="ℹ️")
    else:
        st.metric("Rapid Revisit Cases", len(rapid))
        
        # Show histogram
        from pharmascan.components.charts import rapid_histogram
        
        fig = rapid_histogram(rapid)
        if fig:
            st.pyplot(fig, use_container_width=True)
        
        # Show details
        st.subheader("Rapid Revisit Details")
        rapid_df = pd.DataFrame(rapid)
        st.dataframe(rapid_df.head(100), use_container_width=True)

# ── ALL RECORDS TAB ───────────────────────────────────────────────────────────
with tabs[3]:
    st.dataframe(df, use_container_width=True)
