"""
Data loading and processing functions for PharmaScan.

This module handles:
- Loading data from various file formats (CSV, XLSX, XLS, ODS)
- Column name normalization
- Date parsing
- Summary statistics calculation
- Repeat visit detection
- Rapid revisit detection
"""

import io
import re
from typing import Optional

import pandas as pd

from pharmascan.config.settings import COLUMN_MAP


# Pre-compile regex patterns for column matching
_COLUMN_PATTERNS = [(re.compile(pattern), target) for pattern, target in COLUMN_MAP.items()]


def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Load data from uploaded file bytes.
    
    Args:
        file_bytes: Raw bytes of the uploaded file
        filename: Name of the uploaded file (used to detect format)
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    fname = filename.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(
            io.BytesIO(file_bytes), encoding="utf-8", on_bad_lines="skip", dtype_backend="pyarrow"
        )
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), dtype_backend="pyarrow")
    elif fname.endswith(".ods"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="odf", dtype_backend="pyarrow")
    else:
        raise ValueError("Unsupported file type. Use CSV, XLSX, XLS, or ODS.")


def normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize column names using predefined mapping patterns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with normalized columns, mapping dictionary)
    """
    renamed, used = {}, {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "_", col.lower().strip())
        key = re.sub(r"_+", "_", key).strip("_")
        matched = False
        for pattern, target in _COLUMN_PATTERNS:
            if pattern.fullmatch(key):
                if target not in used:
                    renamed[col] = target
                    used[target] = col
                    matched = True
                break
        if not matched:
            renamed[col] = key
    return df.rename(columns=renamed), renamed


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns in the DataFrame.
    
    First tries 'visit_date' column, then searches for date-like columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with parsed dates
    """
    df = df.copy()
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
    return df


def calculate_summary_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    s = {"total_rows": len(df), "columns": list(df.columns)}
    
    # Patient statistics
    id_col = (
        "patient_id"
        if "patient_id" in df.columns
        else "patient_name" if "patient_name" in df.columns else None
    )
    if id_col:
        vc = df[id_col].value_counts()
        s["patient_col"] = id_col
        s["unique_patients"] = int(df[id_col].nunique())
        s["repeat_patients"] = int((vc > 1).sum())
        s["max_visits"] = int(vc.max())
        s["top_patients"] = vc.head(15).rename_axis("id").reset_index(name="visits")
    
    # Doctor statistics
    dcol = (
        "doctor_name"
        if "doctor_name" in df.columns
        else "doctor_id" if "doctor_id" in df.columns else None
    )
    if dcol:
        dvc = df[dcol].value_counts()
        s["unique_doctors"] = int(df[dcol].nunique())
        s["top_doctors"] = dvc.head(15).rename_axis("doctor").reset_index(name="visits")
        s["doctor_col"] = dcol
    
    # Date range
    if "visit_date" in df.columns:
        v = df["visit_date"].dropna()
        if len(v):
            s["date_min"] = str(v.min().date())
            s["date_max"] = str(v.max().date())
    
    # Facility statistics
    if "facility" in df.columns:
        fvc = df["facility"].value_counts()
        s["unique_facilities"] = int(df["facility"].nunique())
        s["top_facilities"] = fvc.head(10).rename_axis("name").reset_index(name="visits")
    
    # Amount statistics
    for amt_col in ["amount", "medicine_cost", "insurance_copay", "patient_copay"]:
        if amt_col in df.columns:
            df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    if "amount" in df.columns:
        s["total_amount"] = round(float(df["amount"].sum()), 2)
        s["avg_amount"] = round(float(df["amount"].mean()), 2)
    
    return s


def detect_repeat_visits(df: pd.DataFrame, id_col: Optional[str] = None) -> tuple[list, pd.DataFrame]:
    """
    Detect patients with repeat visits.
    
    Args:
        df: Input DataFrame
        id_col: Column to use as patient identifier (auto-detected if None)
        
    Returns:
        Tuple of (list of repeat visit groups, DataFrame with repeat visit details)
    """
    if id_col is None:
        id_col = (
            "patient_id"
            if "patient_id" in df.columns
            else "patient_name" if "patient_name" in df.columns else None
        )
    
    if not id_col:
        return [], pd.DataFrame()
    
    vc2 = df[id_col].value_counts()
    repeat_ids = vc2[vc2 > 1].index.tolist()
    rdf = df[df[id_col].isin(repeat_ids)].copy()
    
    if "visit_date" in rdf.columns:
        rdf = rdf.sort_values([id_col, "visit_date"])
    repeat_detail = rdf.head(500)
    
    repeat_groups = []
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
    return repeat_groups, repeat_detail


def detect_rapid_revisits(
    df: pd.DataFrame, rapid_days: int, id_col: Optional[str] = None
) -> list[dict]:
    """
    Detect rapid revisits (visits within a specified number of days).
    
    Uses vectorized operations for efficiency.
    
    Args:
        df: Input DataFrame
        rapid_days: Maximum number of days between visits to consider "rapid"
        id_col: Column to use as patient identifier (auto-detected if None)
        
    Returns:
        List of dicts containing rapid revisit information
    """
    if id_col is None:
        id_col = (
            "patient_id"
            if "patient_id" in df.columns
            else "patient_name" if "patient_name" in df.columns else None
        )
    
    if not id_col or "visit_date" not in df.columns:
        return []
    
    dcol = (
        "doctor_name"
        if "doctor_name" in df.columns
        else "doctor_id" if "doctor_id" in df.columns else None
    )
    
    cols = [id_col, "visit_date"]
    if "patient_name" in df.columns and id_col != "patient_name":
        cols.append("patient_name")
    if dcol:
        cols.append(dcol)
    
    sub = df[cols].dropna(subset=[id_col, "visit_date"]).sort_values([id_col, "visit_date"])
    
    # Use vectorized diff to calculate days between visits
    sub["_prev_date"] = sub.groupby(id_col)["visit_date"].shift(1)
    sub["_days_diff"] = (sub["visit_date"] - sub["_prev_date"]).dt.days
    
    # Filter for rapid revisits
    rapid_mask = (sub["_days_diff"] > 0) & (sub["_days_diff"] <= rapid_days)
    rapid_df = sub[rapid_mask].copy()
    
    rapid = []
    if len(rapid_df) > 0:
        rapid_dict = {
            "patient_id": rapid_df[id_col].astype(str).tolist(),
            "patient_name": (
                rapid_df["patient_name"].astype(str).tolist()
                if "patient_name" in rapid_df.columns
                else rapid_df[id_col].astype(str).tolist()
            ),
            "visit_1": rapid_df["_prev_date"].dt.strftime("%Y-%m-%d").tolist(),
            "visit_2": rapid_df["visit_date"].dt.strftime("%Y-%m-%d").tolist(),
            "days_apart": rapid_df["_days_diff"].astype(int).tolist(),
        }
        if dcol and dcol in rapid_df.columns:
            rapid_dict["doctor"] = rapid_df[dcol].astype(str).tolist()
        else:
            rapid_dict["doctor"] = ["—"] * len(rapid_df)
        
        rapid_keys = list(rapid_dict.keys())
        rapid = [dict(zip(rapid_keys, row)) for row in zip(*rapid_dict.values())]
        rapid.sort(key=lambda x: x["days_apart"])
    
    # Clean up temporary columns
    sub.drop(columns=["_prev_date", "_days_diff"], inplace=True, errors="ignore")
    
    return rapid


def load_and_process(file_bytes: bytes, filename: str, rapid_days: int):
    """
    Main function to load and process uploaded data file.
    
    Args:
        file_bytes: Raw bytes of the uploaded file
        filename: Name of the uploaded file
        rapid_days: Threshold for detecting rapid revisits
        
    Returns:
        Tuple of (df, col_map, stats, repeat_groups, repeat_detail, rapid)
    """
    # Load data
    df = load_data(file_bytes, filename)
    
    # Normalize column names
    df, col_map = normalize_column_names(df)
    
    # Parse dates
    df = parse_dates(df)
    
    # Calculate summary statistics
    stats = calculate_summary_stats(df)
    
    # Detect repeat visits
    id_col = stats.get("patient_col")
    repeat_groups, repeat_detail = detect_repeat_visits(df, id_col)
    
    # Detect rapid revisits
    rapid = detect_rapid_revisits(df, rapid_days, id_col)
    
    return df, col_map, stats, repeat_groups, repeat_detail, rapid
