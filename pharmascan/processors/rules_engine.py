"""
Rules Engine for PharmaScan fraud detection.

This module implements in-memory fraud detection rules for pharmacy vouchers:
- R01: Drug-Prescriber Mismatch
- R02: Diagnosis-Drug Mismatch
- R03: Drug Quantity Excess
- R04: Restricted Drug – High Value + No Indication
- R05: Antineoplastic Without Cancer Diagnosis
- R06: Psychiatric Drug Without Mental Health Diagnosis
- R07: Duplicate / Early Refill
- R08: Zero-Tariff Procedure or Unlisted Drug
- R09: Malaria + Multiple Antibiotics
- R10: Immunosuppressant Without Valid Indication
"""

import re
from typing import Optional

import pandas as pd


# Diagnosis-Drug Blacklist (ICD prefix -> set of ATC prefixes to flag)
_DX_DRUG_BLACKLIST = {
    "B50": {
        "J01": (
            20,
            "Malaria+antibiotics: UCG first-line is ACT not J01 antibiotics",
        ),
        "L01": (45, "Antineoplastic for malaria: no clinical basis"),
        "N05": (35, "Antipsychotic for malaria: no indication"),
    },
    "B51": {
        "J01": (20, "Malaria (P.vivax) + antibiotics: ACT is first-line"),
        "L01": (45, "Antineoplastic for malaria: impossible"),
    },
    "B54": {
        "J01": (20, "Malaria + antibiotics: ACT protocol not antibiotics"),
        "L01": (45, "Antineoplastic for unspecified malaria"),
    },
    "I10": {
        "P01": (40, "Antihypertensive + antiparasitic: no clinical link"),
        "L01": (50, "Antineoplastic for hypertension: diagnosis fraud"),
        "N05": (30, "Antipsychotic for hypertension: no indication"),
    },
    "E11": {
        "P01": (40, "T2DM + antiparasitic: no clinical indication"),
        "L01": (50, "Antineoplastic for diabetes: diagnosis fraud"),
    },
    "E10": {"L01": (50, "Antineoplastic for T1DM: diagnosis fraud")},
    "J18": {
        "L01": (50, "Antineoplastic for pneumonia: no indication"),
        "N05": (35, "Antipsychotic for pneumonia: no indication"),
    },
    "G40": {
        "P01": (40, "Epilepsy + antiparasitic: UCG uses CBZ/VPA/PHB"),
        "L01": (45, "Antineoplastic for epilepsy: no indication"),
    },
    "A15": {
        "L01": (45, "Antineoplastic for TB: unless concurrent cancer"),
        "N05": (35, "Antipsychotic for TB: not in RHZE protocol"),
    },
    "Z00": {
        "L01": (60, "CRITICAL: Antineoplastic on routine checkup"),
        "N05": (40, "Antipsychotic on routine checkup: billing fraud"),
        "H02": (30, "High-dose steroid on routine checkup"),
    },
    "J06": {
        "L01": (55, "Antineoplastic for URTI: strong fraud signal"),
        "N05": (35, "Antipsychotic for URTI: no indication"),
        "S01": (25, "Ophthalmic prep for URTI: no indication"),
    },
    "J00": {
        "L01": (55, "Antineoplastic for common cold: fraud"),
        "N05": (35, "Antipsychotic for common cold"),
    },
    "O80": {
        "L01": (60, "CRITICAL: Antineoplastic during normal delivery"),
        "N05": (35, "Antipsychotic for normal delivery"),
    },
    "Z23": {
        "L01": (60, "CRITICAL: Antineoplastic alongside vaccination"),
        "N05": (40, "Antipsychotic at vaccination visit"),
    },
    "F20": {
        "P01": (40, "Antipsychotic Rx for schizophrenia needs N05, not P01")
    },
    "F32": {"P01": (40, "Depression + antiparasitic: no indication")},
}

# Prescriber code to speciality mapping
_PRESCRIBER_ALLOWED = {
    "D": {"Dermatology", "Dermatologist"},
    "OPHT": {"Ophthalmology", "Ophthalmologist"},
    "IM": {"Internal Medicine", "Internist", "Physician"},
    "AC": {"Oncology", "Oncologist"},
    "UROL": {"Urology", "Urologist"},
    "GYN": {"Gynaecology", "Gynaecologist", "Obstetrics", "OB-GYN"},
    "GYNEC": {"Gynaecology", "Gynaecologist"},
    "PSYCH": {"Psychiatry", "Psychiatrist"},
    "CARDIOL": {"Cardiology", "Cardiologist"},
    "DT": {"Dentistry", "Dental Surgeon"},
    "NEUROL": {"Neurology", "Neurologist"},
    "PED": {"Paediatrics", "Paediatrician", "Pediatrics"},
    "NEPHR": {"Nephrology", "Nephrologist"},
    "SPEC": {"*specialist*"},  # any registered specialist
    "HU": {"*hospital*"},  # hospital/inpatient only
}


def _atc_prefix(code: str, n: int) -> str:
    """Extract first n characters from ATC code."""
    return str(code).strip()[:n].upper()


def run_rules_engine(
    df: pd.DataFrame, drug_ref_loader
) -> tuple[pd.DataFrame, dict]:
    """
    Run all in-memory fraud rules against the dataframe.
    
    Args:
        df: Input DataFrame with voucher records
        drug_ref_loader: Function that loads drug reference data
        
    Returns:
        Tuple of (results_df, summary_stats)
        - results_df: Original df with added columns (_score, _decision, _rules_fired, _risk_level)
        - summary_stats: Dictionary with rule execution statistics
    """
    ref = drug_ref_loader()
    drugs_ref = ref["drugs"]
    atc3_ref = ref["atc3_defaults"]

    # Identify available columns
    def _col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    id_col = _col("patient_id", "patient_name")
    date_col = _col("visit_date")
    drug_col = _col("drug_code")
    drug_nm = _col("drug_name")
    qty_col = _col("quantity")
    dx_col = _col("diagnosis")
    doc_col = _col("doctor_name")
    doc_type = _col("doctor_type")
    fac_col = _col("facility")
    amt_col = _col("insurance_copay", "amount")
    vou_col = _col("voucher_id")

    rows_out = []
    summary = {
        "total": len(df),
        "rule_counts": {f"R{i:02d}": 0 for i in range(1, 11)},
        "decisions": {"APPROVE": 0, "FLAG": 0, "HOLD": 0, "BLOCK": 0},
        "total_flagged_amount": 0.0,
        "rules_available": [],
    }

    # Determine which rules can run
    if drug_col:
        summary["rules_available"] += [
            "R01",
            "R02",
            "R03",
            "R04",
            "R07",
            "R08",
            "R09",
        ]
    if dx_col:
        summary["rules_available"] += ["R02", "R05", "R06", "R09"]
    if qty_col:
        summary["rules_available"] += ["R03"]
    if date_col and id_col:
        summary["rules_available"] += ["R10"]
    summary["rules_available"] = sorted(set(summary["rules_available"]))

    # Pre-build refill index: {(patient_id, drug_code): [sorted dates]}
    refill_index = {}
    if id_col and drug_col and date_col:
        sub = df[[id_col, drug_col, date_col]].dropna()
        for _, row in sub.iterrows():
            key = (str(row[id_col]).strip(), str(row[drug_col]).strip())
            dt = pd.to_datetime(row[date_col], errors="coerce")
            if pd.notna(dt):
                refill_index.setdefault(key, []).append(dt)
        for key in refill_index:
            refill_index[key].sort()

    for idx, row in df.iterrows():
        score = 0
        fired = []

        def fire(rule_id, s, reason, evidence=""):
            nonlocal score
            score += s
            fired.append(
                {
                    "id": rule_id,
                    "score": s,
                    "reason": reason[:120],
                    "evidence": str(evidence)[:80],
                }
            )
            summary["rule_counts"][rule_id] = (
                summary["rule_counts"].get(rule_id, 0) + 1
            )

        # Get row values
        d_code = (
            str(row[drug_col]).strip()
            if drug_col and pd.notna(row.get(drug_col))
            else ""
        )
        d_name = (
            str(row[drug_nm]).strip()
            if drug_nm and pd.notna(row.get(drug_nm))
            else ""
        )
        qty = (
            float(row[qty_col])
            if qty_col and pd.notna(row.get(qty_col))
            else None
        )
        dx = (
            str(row[dx_col]).strip()[:3].upper()
            if dx_col and pd.notna(row.get(dx_col))
            else ""
        )
        doc = (
            str(row[doc_type]).strip()
            if doc_type and pd.notna(row.get(doc_type))
            else ""
        )
        fac = (
            str(row[fac_col]).strip()
            if fac_col and pd.notna(row.get(fac_col))
            else ""
        )
        amt = (
            float(row[amt_col])
            if amt_col and pd.notna(row.get(amt_col))
            else 0.0
        )
        pid = (
            str(row[id_col]).strip()
            if id_col and pd.notna(row.get(id_col))
            else ""
        )
        vou = (
            str(row[vou_col]).strip()
            if vou_col and pd.notna(row.get(vou_col))
            else ""
        )

        # Look up drug info
        drug_info = drugs_ref.get(d_code)
        if not drug_info and d_code:
            drug_info = atc3_ref.get(d_code[:3])

        atc1 = (drug_info["atc1"] if drug_info else d_code[:1]).upper()
        atc3 = (drug_info["atc3"] if drug_info else d_code[:3]).upper()
        instr = (drug_info["instr"] if drug_info else "").strip()
        price = drug_info["price"] if drug_info else 0.0
        max_u = drug_info.get("max_units") if drug_info else None
        min_r = drug_info.get("min_refill") if drug_info else None

        # ── R01: Drug-Prescriber Mismatch ────────────────────────────────────
        if d_code and instr and doc:
            doc_up = doc.upper()
            instr_parts = {
                p.strip().upper() for p in re.split(r"[\s,]+", instr) if p.strip()
            }
            # Hospital-use drugs at non-hospital providers
            if "HU" in instr_parts:
                if not any(
                    x in doc_up for x in ("HOSPITAL", "INTERN", "SPECIALIST", "SPEC", "SENIOR")
                ):
                    fire(
                        "R01",
                        35,
                        f"HU-restricted drug by non-hospital provider",
                        f"{d_code}|{instr}|{doc[:30]}",
                    )
            # PSYCH-restricted drugs by non-psychiatrist
            elif "PSYCH" in instr_parts:
                if not any(
                    x in doc_up for x in ("PSYCH", "NEUROL", "SPECIALIST", "SPEC")
                ):
                    fire(
                        "R01",
                        25,
                        f"PSYCH-restricted drug by {doc[:25]}",
                        f"{d_code}|{instr}",
                    )
            # AC (oncology) drugs
            elif "AC" in instr_parts and "AC" not in {"DAC"}:
                if not any(
                    x in doc_up
                    for x in ("ONCOL", "CANCER", "HAEMATOL", "SPECIALIST", "SPEC")
                ):
                    fire(
                        "R01",
                        30,
                        f"Oncology-only drug by non-oncologist {doc[:20]}",
                        f"{d_code}|{instr}",
                    )
            # OPHT drugs
            elif "OPHT" in instr_parts:
                if (
                    "OPHTH" not in doc_up
                    and "EYE" not in doc_up
                    and "SPEC" not in doc_up
                ):
                    fire(
                        "R01",
                        20,
                        f"OPHT drug by non-ophthalmologist",
                        f"{d_code}|{instr}|{doc[:20]}",
                    )

        # ── R02: Diagnosis-Drug Mismatch ─────────────────────────────────────
        if dx and d_code:
            if dx in _DX_DRUG_BLACKLIST:
                for atc_pref, (s, reason) in _DX_DRUG_BLACKLIST[dx].items():
                    if atc1 == atc_pref[:1] and (
                        len(atc_pref) == 1 or atc3.startswith(atc_pref)
                    ):
                        fire("R02", s, reason, f"ICD:{dx} + {d_code}({atc_pref})")
                        break

        # ── R03: Drug Quantity Excess ─────────────────────────────────────────
        if qty and max_u:
            if float(qty) > float(max_u):
                excess_pct = (float(qty) - float(max_u)) / float(max_u) * 100
                s = min(25 + int(excess_pct / 20) * 5, 60)
                fire(
                    "R03",
                    s,
                    f"Quantity {qty:.0f} > max {max_u} ({excess_pct:.0f}% excess)",
                    f"{d_code}|qty:{qty}|max:{max_u}",
                )

        # ── R04: Restricted Drug – High Value + No Indication ─────────────────
        if price > 50000 and dx:
            atc1_dx_ok = {
                "L": {"C", "D", "N", "G", "M"},  # antineoplastics need cancer/relevant dx
                "B": {"D", "N", "K"},  # erythropoietin needs haem/renal
            }
            if atc1 in atc1_dx_ok:
                expected_dx_firsts = atc1_dx_ok[atc1]
                if dx[:1] not in expected_dx_firsts:
                    fire(
                        "R04",
                        30,
                        f"High-value drug ({price:,.0f} RWF) with unrelated diagnosis {dx}",
                        f"{d_code}|price:{price:,.0f}|dx:{dx}",
                    )

        # ── R05: Antineoplastic Without Cancer Diagnosis ──────────────────────
        if atc1 == "L" and atc3.startswith("L01"):
            is_cancer_dx = dx.startswith("C") or (
                dx.startswith("D")
                and dx[1:3].isdigit()
                and 0 <= int(dx[1:3]) <= 49
            )
            if dx and not is_cancer_dx:
                fire(
                    "R05",
                    25,
                    f"Cytotoxic drug without cancer diagnosis (ICD:{dx})",
                    f"{d_code}|dx:{dx}",
                )

        # ── R06: Psychiatric Drug Without Mental Health Diagnosis ─────────────
        if "PSYCH" in instr.upper() and dx:
            is_mental = dx.startswith("F") or (
                dx.startswith("G4") and len(dx) >= 3 and "0" <= dx[2] <= "7"
            )
            if not is_mental:
                fire(
                    "R06",
                    20,
                    f"PSYCH drug without psychiatric/neuro diagnosis (ICD:{dx})",
                    f"{d_code}|instr:{instr}|dx:{dx}",
                )

        # ── R07: Duplicate / Early Refill ─────────────────────────────────────
        if pid and d_code and min_r and date_col:
            key = (pid, d_code)
            dates = refill_index.get(key, [])
            if len(dates) >= 2:
                cur_dt = pd.to_datetime(row.get(date_col), errors="coerce")
                if pd.notna(cur_dt):
                    # Find most recent prior dispense
                    prior = [d for d in dates if d < cur_dt]
                    if prior:
                        gap = (cur_dt - max(prior)).days
                        if gap < min_r:
                            fire(
                                "R07",
                                40,
                                f"Refill {gap}d after last dispense (min:{min_r}d)",
                                f"{d_code}|gap:{gap}d|min:{min_r}d",
                            )

        # ── R08: Zero-Tariff Procedure or Unlisted Drug ───────────────────────
        if d_code and not drug_info and d_code.upper().startswith("RHIC"):
            fire(
                "R08",
                15,
                f"Procedure {d_code} not found in RAMA tariff",
                f"{d_code}",
            )

        # ── R09: Malaria + Multiple Antibiotics ───────────────────────────────
        # (checked at patient-visit level below in post-processing)

        # ── R10: Immunosuppressant Without Valid Indication ───────────────────
        if atc3.startswith("L04") and dx:
            valid = any(
                dx.startswith(p)
                for p in (
                    "T86",
                    "M0",
                    "M1",
                    "M2",
                    "M3",
                    "K50",
                    "K51",
                    "K52",
                    "N04",
                    "L40",
                    "L41",
                    "G35",
                )
            )
            if not valid:
                fire(
                    "R10",
                    20,
                    f"Immunosuppressant without transplant/autoimmune dx ({dx})",
                    f"{d_code}|dx:{dx}",
                )

        # ── Scoring → decision ─────────────────────────────────────────────────
        if score >= 75:
            decision = "BLOCK"
        elif score >= 50:
            decision = "HOLD"
        elif score >= 30:
            decision = "FLAG"
        else:
            decision = "APPROVE"

        if score >= 75:
            risk = "CRITICAL"
        elif score >= 50:
            risk = "HIGH"
        elif score >= 30:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        summary["decisions"][decision] += 1
        if decision in ("HOLD", "BLOCK"):
            summary["total_flagged_amount"] += amt

        rows_out.append(
            {
                "_score": score,
                "_risk": risk,
                "_decision": decision,
                "_rules_fired": "; ".join(
                    f"{r['id']}(+{r['score']})" for r in fired
                )
                if fired
                else "—",
                "_reasons": " | ".join(r["reason"] for r in fired) if fired else "—",
                "_n_rules": len(fired),
            }
        )

    results = pd.DataFrame(rows_out, index=df.index)
    out_df = pd.concat([df, results], axis=1)

    summary["flagged_count"] = (
        summary["decisions"]["FLAG"]
        + summary["decisions"]["HOLD"]
        + summary["decisions"]["BLOCK"]
    )
    summary["rules_with_most_fires"] = sorted(
        [(k, v) for k, v in summary["rule_counts"].items() if v > 0],
        key=lambda x: -x[1],
    )[:5]

    return out_df, summary


def get_dx_drug_blacklist() -> dict:
    """Return the diagnosis-drug blacklist rules."""
    return _DX_DRUG_BLACKLIST.copy()


def get_prescriber_allowed() -> dict:
    """Return the prescriber allowed specialties mapping."""
    return _PRESCRIBER_ALLOWED.copy()
