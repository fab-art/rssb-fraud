"""
Cross-Facility Matching Engine for PharmaScan.

Improvements over the original inline matching logic:

  1. RAMA normalization — strips common prefixes (RW/, RSSB/), separators, and
     leading zeros so that "RW/012345" matches "12345" correctly.

  2. Richer name matching — delegates to _match_score() (token subset, fuzzy
     token, and sequence similarity) instead of the simple Jaccard tokenizer
     that was defined inline in the app.

  3. Directional date window — pharmacy dispensing should be AFTER the facility
     visit (prescription → dispensing flow). A match is valid only when
     -1 ≤ (ph_date − fac_date).days ≤ date_window. The −1 day buffer absorbs
     timezone/same-day entry discrepancies; beyond that, a pharmacy date that
     pre-dates the hospital visit is treated as unlinked regardless of RAMA.

  4. O(1) RAMA lookup — facility records are pre-indexed by normalized RAMA into
     a dict so matching is O(n) not O(n×m).

  5. Best UNLINKED context — for UNLINKED rows, returns the facility record with
     the smallest absolute date gap (not an arbitrary iloc[0]).

  6. Composite confidence score — MATCHED rows carry a 0–1 score
     (40 % name quality + 60 % date proximity) so investigators can quickly
     distinguish strong matches from marginal ones.
"""
import re
from collections import defaultdict
from typing import Optional

import pandas as pd

from pharmascan.utils.normalization import _match_score

_PREFIX_RE = re.compile(r"^(RWA?/?|RSSB/?)\s*", re.IGNORECASE)
_SEP_RE = re.compile(r"[\s/\-]")
_LZERO_RE = re.compile(r"^(0+)(\d+)$")


def normalize_rama(x) -> str:
    """
    Return a canonical form of a RAMA/affiliation number for cross-system matching.

    Handles:
      - Prefixes: RW/, RWA/, RSSB/  (case-insensitive)
      - Separators: spaces, slashes, hyphens
      - Leading zeros on purely-numeric IDs
    """
    s = _PREFIX_RE.sub("", str(x).strip().upper())
    s = _SEP_RE.sub("", s)
    m = _LZERO_RE.match(s)
    return m.group(2) if m else s


def run_match(
    ph_work: pd.DataFrame,
    fac_df: pd.DataFrame,
    name_thresh: float = 0.4,
    date_window: int = 7,
    require_name: bool = True,
) -> pd.DataFrame:
    """
    Match pharmacy vouchers against facility visit records.

    Columns consumed from ph_work : _rama, _name, _date, _vou, _ins, _tot, _doc, _dpt
    Columns consumed from fac_df  : _rama, _name, _date, _source, voucher_id

    Parameters
    ----------
    ph_work       Pharmacy working set (see pharmascan_app.py build section).
    fac_df        Concatenated facility records from all uploaded files.
    name_thresh   Minimum name similarity (0–1) for a directional match.
    date_window   Maximum days (ph_date − fac_date) accepted as a match.
                  Directional: pharmacy must come after the facility visit.
    require_name  If False, match on RAMA and date only (no name gate).

    Returns
    -------
    DataFrame with one row per pharmacy voucher and columns:
        status       – MATCHED | UNLINKED | NO_RECORD
        confidence   – 0–1 composite (MATCHED only; 0.0 for others)
        ph_voucher, ph_patient, ph_rama, ph_date, ph_ins, ph_total,
        ph_doctor, ph_dept
        fac_voucher, fac_name, fac_date, fac_source  (None for NO_RECORD)
        days_apart   – signed int: ph_date − fac_date
                       (negative ⇒ pharmacy dispensed BEFORE the facility visit)
        name_score   – raw name similarity 0–1
    """
    ph = ph_work.copy()
    fac = fac_df.copy()

    ph["_rn"] = ph["_rama"].apply(normalize_rama)
    fac["_rn"] = fac["_rama"].apply(normalize_rama)

    # Build O(1) RAMA index from facility records
    fac_index: dict[str, list[dict]] = defaultdict(list)
    for fr in fac.to_dict("records"):
        fac_index[fr["_rn"]].append(fr)

    output_rows = []
    for pr in ph.to_dict("records"):
        candidates = fac_index.get(pr["_rn"], [])

        if not candidates:
            output_rows.append(_no_record_row(pr))
            continue

        # Search for a candidate that passes both name and date gates
        best_fr = None
        best_conf = -1.0
        best_days: Optional[int] = None
        best_ns = 0.0

        for fr in candidates:
            ns, _ = _match_score(str(pr["_name"]), str(fr["_name"]))
            name_ok = (ns >= name_thresh) if require_name else True
            days, date_ok = _check_date(pr["_date"], fr["_date"], date_window)
            if name_ok and date_ok:
                conf = _composite_conf(ns, days, date_window)
                if conf > best_conf:
                    best_conf, best_fr, best_days, best_ns = conf, fr, days, ns

        if best_fr is not None:
            output_rows.append(_matched_row(pr, best_fr, best_conf, best_days, best_ns))
        else:
            # UNLINKED: RAMA found but no record passes the combined gate.
            # Surface the facility record with the smallest absolute date gap
            # so investigators see the most contextually relevant comparison.
            closest, closest_days = _closest_by_date(candidates, pr["_date"])
            closest_ns, _ = _match_score(str(pr["_name"]), str(closest["_name"]))
            output_rows.append(_unlinked_row(pr, closest, closest_days, closest_ns))

    return pd.DataFrame(output_rows)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_date(ph_date, fac_date, window: int) -> tuple[Optional[int], bool]:
    """
    Directional date check.

    Returns (signed_days, is_valid) where signed_days = ph_date − fac_date.
    Valid range: −1 ≤ days ≤ window
      • Negative values outside [−1, 0] mean pharmacy came well before the
        facility visit, which is clinically invalid.
      • The −1 buffer absorbs same-day records entered with different timestamps.
    """
    if pd.isna(ph_date) or pd.isna(fac_date):
        return None, False
    days = (pd.Timestamp(ph_date) - pd.Timestamp(fac_date)).days
    return days, (-1 <= days <= window)


def _composite_conf(name_score: float, days: Optional[int], window: int) -> float:
    """
    Composite match confidence: 40 % name quality + 60 % date proximity.

    Date proximity is 1.0 at same-day (days=0) and 0.0 at days=window+1,
    so a same-day perfect-name match yields confidence = 1.0.
    """
    if days is None:
        return 0.0
    day_proximity = 1.0 - max(days, 0) / (window + 1)
    return round(0.4 * name_score + 0.6 * day_proximity, 3)


def _closest_by_date(candidates: list[dict], ph_date) -> tuple[dict, Optional[int]]:
    """
    Return (facility_record, signed_days) for the candidate with the smallest
    absolute date difference from ph_date.  Falls back to candidates[0] when
    no dates are available.
    """
    best = candidates[0]
    best_gap: Optional[int] = None
    best_signed: Optional[int] = None

    for fr in candidates:
        if pd.notna(ph_date) and pd.notna(fr["_date"]):
            signed = (pd.Timestamp(ph_date) - pd.Timestamp(fr["_date"])).days
            gap = abs(signed)
            if best_gap is None or gap < best_gap:
                best_gap, best_signed, best = gap, signed, fr

    return best, best_signed


def _base_fields(pr: dict) -> dict:
    return {
        "ph_voucher": pr["_vou"],
        "ph_patient": pr["_name"],
        "ph_rama":    pr["_rama"],
        "ph_date":    pr["_date"],
        "ph_ins":     pr["_ins"],
        "ph_total":   pr["_tot"],
        "ph_doctor":  pr["_doc"],
        "ph_dept":    pr["_dpt"],
    }


def _no_record_row(pr: dict) -> dict:
    return {
        **_base_fields(pr),
        "status":      "NO_RECORD",
        "confidence":  0.0,
        "fac_voucher": None,
        "fac_name":    None,
        "fac_date":    None,
        "fac_source":  None,
        "days_apart":  None,
        "name_score":  None,
    }


def _matched_row(pr: dict, fr: dict, conf: float, days: Optional[int], ns: float) -> dict:
    return {
        **_base_fields(pr),
        "status":      "MATCHED",
        "confidence":  conf,
        "fac_voucher": fr.get("voucher_id"),
        "fac_name":    fr["_name"],
        "fac_date":    fr["_date"],
        "fac_source":  fr["_source"],
        "days_apart":  days,
        "name_score":  round(ns, 2),
    }


def _unlinked_row(pr: dict, fr: dict, days: Optional[int], ns: float) -> dict:
    return {
        **_base_fields(pr),
        "status":      "UNLINKED",
        "confidence":  0.0,
        "fac_voucher": fr.get("voucher_id"),
        "fac_name":    fr["_name"],
        "fac_date":    fr["_date"],
        "fac_source":  fr["_source"],
        "days_apart":  days,
        "name_score":  round(ns, 2),
    }
