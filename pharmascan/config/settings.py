"""
Configuration constants for PharmaScan application.
"""

# ── Colours ───────────────────────────────────────────────────────────────────
ACCENT = "#00e5a0"
ACCENT2 = "#0ea5e9"
PURPLE = "#a78bfa"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#e2e8f0"
DARK = "#0d1117"
BG = DARK  # alias for matplotlib axes
CARD = "#111720"
BORDER = "#1e2a38"

# ── Column normalisation mapping ──────────────────────────────────────────────
COLUMN_MAP = {
    # ── Exact columns from the real data ──────────────────────────────────────
    r"#": "row_number",
    r"paper.?code": "voucher_id",
    r"dispensing.?date": "visit_date",
    r"patient.?name": "patient_name",
    r"patient.?type": "patient_type",
    r"gender": "gender",
    r"is.?newborn": "is_newborn",
    r"rama.?number": "patient_id",
    r"practitioner.?name": "doctor_name",
    r"practitioner.?type": "doctor_type",
    r"total.?cost": "amount",
    r"patient.?co.?payment": "patient_copay",
    r"insurance.?co.?payment": "insurance_copay",
    r"medicine.?cost": "medicine_cost",
    # ── Generic fallbacks ─────────────────────────────────────────────────────
    r"patient.?(id|no|num|number|code)?": "patient_id",
    r"pat.?id|pid": "patient_id",
    r"doctor.?(id|no|num|code)?": "doctor_id",
    r"(doctor|dr|physician|prescriber).?name": "doctor_name",
    r"doc.?id|did": "doctor_id",
    r"prescriber": "doctor_name",
    r"(visit|service|rx|voucher).?date": "visit_date",
    r"date.?(of.?)?(visit|service|dispensing)?": "visit_date",
    r"date": "visit_date",
    r"(pharmacy|facility|clinic|hospital|branch).?(name|id|code)?": "facility",
    r"(drug|medicine|medication|item|product).?(name|description|desc)?": "drug_name",
    r"(drug|medicine|medication|item|product).?(code|id)?": "drug_code",
    r"(amount|cost|price|value|total|charge)": "amount",
    r"quantity|qty": "quantity",
    r"(diagnosis|diag|icd|condition)": "diagnosis",
    r"(voucher|claim|ref|reference).?(no|number|id|code)?": "voucher_id",
}

# ── Default settings ──────────────────────────────────────────────────────────
DEFAULT_RAPID_DAYS = 7
DEFAULT_MAX_NETWORK_NODES = 100
DEFAULT_MIN_EDGE_WEIGHT = 1
DEFAULT_TOP_N = 15
