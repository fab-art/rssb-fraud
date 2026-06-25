import pandas as pd
from collections import defaultdict as _dd
from .normalization import normalize_rama, match_score

def run_match(ph_work: pd.DataFrame, fac_df: pd.DataFrame, name_thresh: float = 0.4, date_window: int = 7, require_name: bool = True) -> pd.DataFrame:
    ph  = ph_work.copy()
    fac = fac_df.copy()
    ph["_rn"]  = ph["_rama"].apply(normalize_rama)
    fac["_rn"] = fac["_rama"].apply(normalize_rama)
    fac_index = _dd(list)
    for fr in fac.to_dict("records"):
        fac_index[fr["_rn"]].append(fr)
    output_rows = []
    for pr in ph.to_dict("records"):
        candidates = fac_index.get(pr["_rn"], [])
        if not candidates:
            output_rows.append(_cf_no_record(pr))
            continue
        best_fr, best_conf, best_days, best_ns = None, -1.0, None, 0.0
        for fr in candidates:
            ns, _ = match_score(str(pr["_name"]), str(fr["_name"]))
            name_ok = (ns >= name_thresh) if require_name else True
            days, dok = _cf_check_date(pr["_date"], fr["_date"], date_window)
            if name_ok and dok:
                conf = _cf_conf(ns, days, date_window)
                if conf > best_conf: best_conf, best_fr, best_days, best_ns = conf, fr, days, ns
        if best_fr is not None:
            output_rows.append(_cf_matched(pr, best_fr, best_conf, best_days, best_ns))
        else:
            closest, closest_days = _cf_closest(candidates, pr["_date"])
            closest_ns, _ = match_score(str(pr["_name"]), str(closest["_name"]))
            output_rows.append(_cf_unlinked(pr, closest, closest_days, closest_ns))
    return pd.DataFrame(output_rows)

def _cf_check_date(ph_date, fac_date, window):
    if pd.isna(ph_date) or pd.isna(fac_date): return None, False
    days = (pd.Timestamp(ph_date) - pd.Timestamp(fac_date)).days
    return days, (-1 <= days <= window)

def _cf_conf(name_score, days, window):
    if days is None: return 0.0
    day_prox = 1.0 - max(days, 0) / (window + 1)
    return round(0.4 * name_score + 0.6 * day_prox, 3)

def _cf_closest(candidates, ph_date):
    best, best_gap, best_signed = candidates[0], None, None
    for fr in candidates:
        if pd.notna(ph_date) and pd.notna(fr["_date"]):
            signed = (pd.Timestamp(ph_date) - pd.Timestamp(fr["_date"])).days
            gap = abs(signed)
            if best_gap is None or gap < best_gap: best_gap, best_signed, best = gap, signed, fr
    return best, best_signed

def _cf_base(pr):
    return {"ph_voucher": pr["_vou"], "ph_patient": pr["_name"], "ph_rama": pr["_rama"], "ph_date": pr["_date"], "ph_ins": pr["_ins"], "ph_total": pr["_tot"], "ph_doctor": pr["_doc"], "ph_dept": pr["_dpt"]}

def _cf_no_record(pr):
    return {**_cf_base(pr), "status": "NO_RECORD", "confidence": 0.0, "fac_voucher": None, "fac_name": None, "fac_date": None, "fac_source": None, "days_apart": None, "name_score": None}

def _cf_matched(pr, fr, conf, days, ns):
    return {**_cf_base(pr), "status": "MATCHED", "confidence": conf, "fac_voucher": fr.get("voucher_id"), "fac_name": fr["_name"], "fac_date": fr["_date"], "fac_source": fr["_source"], "days_apart": days, "name_score": round(ns, 2)}

def _cf_unlinked(pr, fr, days, ns):
    return {**_cf_base(pr), "status": "UNLINKED", "confidence": 0.0, "fac_voucher": fr.get("voucher_id"), "fac_name": fr["_name"], "fac_date": fr["_date"], "fac_source": fr["_source"], "days_apart": days, "name_score": round(ns, 2)}
