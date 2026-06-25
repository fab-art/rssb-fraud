import re
import pandas as pd
from .drug_ref import load_drug_ref

_DX_DRUG_BLACKLIST = {
    "B50": {"J01": (20,"Malaria+antibiotics: UCG first-line is ACT not J01 antibiotics"), "L01": (45,"Antineoplastic for malaria: no clinical basis"), "N05": (35,"Antipsychotic for malaria: no indication")},
    "B51": {"J01": (20,"Malaria (P.vivax) + antibiotics: ACT is first-line"), "L01": (45,"Antineoplastic for malaria: impossible")},
    "B54": {"J01": (20,"Malaria + antibiotics: ACT protocol not antibiotics"), "L01": (45,"Antineoplastic for unspecified malaria")},
    "I10": {"P01": (40,"Antihypertensive + antiparasitic: no clinical link"), "L01": (50,"Antineoplastic for hypertension: diagnosis fraud"), "N05": (30,"Antipsychotic for hypertension: no indication")},
    "E11": {"P01": (40,"T2DM + antiparasitic: no clinical indication"), "L01": (50,"Antineoplastic for diabetes: diagnosis fraud")},
    "E10": {"L01": (50,"Antineoplastic for T1DM: diagnosis fraud")},
    "J18": {"L01": (50,"Antineoplastic for pneumonia: no indication"), "N05": (35,"Antipsychotic for pneumonia: no indication")},
    "G40": {"P01": (40,"Epilepsy + antiparasitic: UCG uses CBZ/VPA/PHB"), "L01": (45,"Antineoplastic for epilepsy: no indication")},
    "A15": {"L01": (45,"Antineoplastic for TB: unless concurrent cancer"), "N05": (35,"Antipsychotic for TB: not in RHZE protocol")},
    "Z00": {"L01": (60,"CRITICAL: Antineoplastic on routine checkup"), "N05": (40,"Antipsychotic on routine checkup: billing fraud"), "H02": (30,"High-dose steroid on routine checkup")},
    "J06": {"L01": (55,"Antineoplastic for URTI: strong fraud signal"), "N05": (35,"Antipsychotic for URTI: no indication"), "S01": (25,"Ophthalmic prep for URTI: no indication")},
    "J00": {"L01": (55,"Antineoplastic for common cold: fraud"), "N05": (35,"Antipsychotic for common cold")},
    "O80": {"L01": (60,"CRITICAL: Antineoplastic during normal delivery"), "N05": (35,"Antipsychotic for normal delivery")},
    "Z23": {"L01": (60,"CRITICAL: Antineoplastic alongside vaccination"), "N05": (40,"Antipsychotic at vaccination visit")},
    "F20": {"P01": (40,"Antipsychotic Rx for schizophrenia needs N05, not P01")},
    "F32": {"P01": (40,"Depression + antiparasitic: no indication")},
}

_PRESCRIBER_ALLOWED = {
    "D": {"Dermatology","Dermatologist"}, "OPHT": {"Ophthalmology","Ophthalmologist"}, "IM": {"Internal Medicine","Internist","Physician"}, "AC": {"Oncology","Oncologist"}, "UROL": {"Urology","Urologist"}, "GYN": {"Gynaecology","Gynaecologist","Obstetrics","OB-GYN"}, "GYNEC": {"Gynaecology","Gynaecologist"}, "PSYCH": {"Psychiatry","Psychiatrist"}, "CARDIOL": {"Cardiology","Cardiologist"}, "DT": {"Dentistry","Dental Surgeon"}, "NEUROL": {"Neurology","Neurologist"}, "PED": {"Paediatrics","Paediatrician","Pediatrics"}, "NEPHR": {"Nephrology","Nephrologist"}, "SPEC": {"*specialist*"}, "HU": {"*hospital*"}
}

def run_rules_engine(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ref = load_drug_ref()
    drugs_ref = ref["drugs"]
    atc3_ref = ref["atc3_defaults"]
    def _col(*names):
        for n in names:
            if n in df.columns: return n
        return None
    id_col = _col("patient_id", "patient_name"); date_col = _col("visit_date"); drug_col = _col("drug_code"); drug_nm = _col("drug_name"); qty_col = _col("quantity"); dx_col = _col("diagnosis"); doc_col = _col("doctor_name"); doc_type = _col("doctor_type"); fac_col = _col("facility"); amt_col = _col("insurance_copay", "amount"); vou_col = _col("voucher_id")
    rows_out = []
    summary = {"total": len(df), "rule_counts": {f"R{i:02d}": 0 for i in range(1, 11)}, "decisions": {"APPROVE": 0, "FLAG": 0, "HOLD": 0, "BLOCK": 0}, "total_flagged_amount": 0.0, "rules_available": []}
    if drug_col: summary["rules_available"] += ["R01","R02","R03","R04","R07","R08","R09"]
    if dx_col: summary["rules_available"] += ["R02","R05","R06","R09"]
    if qty_col: summary["rules_available"] += ["R03"]
    if date_col and id_col: summary["rules_available"] += ["R10"]
    summary["rules_available"] = sorted(set(summary["rules_available"]))
    refill_index = {}
    if id_col and drug_col and date_col:
        sub = df[[id_col, drug_col, date_col]].copy(); sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce"); sub = sub.dropna(subset=[id_col, drug_col, date_col]); sub[id_col] = sub[id_col].astype(str).str.strip(); sub[drug_col] = sub[drug_col].astype(str).str.strip()
        for (pid, dcode), grp in sub.groupby([id_col, drug_col], sort=False): refill_index[(pid, dcode)] = sorted(grp[date_col].tolist())
    for idx, row in df.iterrows():
        score = 0; fired = []
        def fire(rule_id, s, reason, evidence=""):
            nonlocal score; score += s; fired.append({"id": rule_id, "score": s, "reason": reason[:120], "evidence": str(evidence)[:80]})
            summary["rule_counts"][rule_id] = summary["rule_counts"].get(rule_id, 0) + 1
        d_code = str(row[drug_col]).strip() if drug_col and pd.notna(row.get(drug_col)) else ""; d_name = str(row[drug_nm]).strip() if drug_nm and pd.notna(row.get(drug_nm)) else ""; qty = float(row[qty_col]) if qty_col and pd.notna(row.get(qty_col)) else None; dx = str(row[dx_col]).strip()[:3].upper() if dx_col and pd.notna(row.get(dx_col)) else ""; doc = str(row[doc_type]).strip() if doc_type and pd.notna(row.get(doc_type)) else ""; fac = str(row[fac_col]).strip() if fac_col and pd.notna(row.get(fac_col)) else ""; amt = float(row[amt_col]) if amt_col and pd.notna(row.get(amt_col)) else 0.0; pid = str(row[id_col]).strip() if id_col and pd.notna(row.get(id_col)) else ""; vou = str(row[vou_col]).strip() if vou_col and pd.notna(row.get(vou_col)) else ""
        drug_info = drugs_ref.get(d_code) or atc3_ref.get(d_code[:3])
        atc1 = (drug_info["atc1"] if drug_info else d_code[:1]).upper(); atc3 = (drug_info["atc3"] if drug_info else d_code[:3]).upper(); instr = (drug_info["instr"] if drug_info else "").strip(); price = drug_info["price"] if drug_info else 0.0; max_u = drug_info.get("max_units") if drug_info else None; min_r = drug_info.get("min_refill") if drug_info else None
        if d_code and instr and doc:
            doc_up = doc.upper(); instr_parts = {p.strip().upper() for p in re.split(r"[\s,]+", instr) if p.strip()}
            if "HU" in instr_parts and not any(x in doc_up for x in ("HOSPITAL","INTERN","SPECIALIST","SPEC","SENIOR")): fire("R01", 35, "HU-restricted drug by non-hospital provider", f"{d_code}|{instr}|{doc[:30]}")
            elif "PSYCH" in instr_parts and not any(x in doc_up for x in ("PSYCH","NEUROL","SPECIALIST","SPEC")): fire("R01", 25, f"PSYCH-restricted drug by {doc[:25]}", f"{d_code}|{instr}")
            elif "AC" in instr_parts and "AC" not in {"DAC"} and not any(x in doc_up for x in ("ONCOL","CANCER","HAEMATOL","SPECIALIST","SPEC")): fire("R01", 30, f"Oncology-only drug by non-oncologist {doc[:20]}", f"{d_code}|{instr}")
            elif "OPHT" in instr_parts and "OPHTH" not in doc_up and "EYE" not in doc_up and "SPEC" not in doc_up: fire("R01", 20, "OPHT drug by non-ophthalmologist", f"{d_code}|{instr}|{doc[:20]}")
        if dx and d_code and dx in _DX_DRUG_BLACKLIST:
            for atc_pref, (s, reason) in _DX_DRUG_BLACKLIST[dx].items():
                if atc1 == atc_pref[:1] and (len(atc_pref) == 1 or atc3.startswith(atc_pref)): fire("R02", s, reason, f"ICD:{dx} + {d_code}({atc_pref})"); break
        if qty and max_u and float(qty) > float(max_u):
            excess_pct = (float(qty) - float(max_u)) / float(max_u) * 100; s = min(25 + int(excess_pct / 20) * 5, 60); fire("R03", s, f"Quantity {qty:.0f} > max {max_u} ({excess_pct:.0f}% excess)", f"{d_code}|qty:{qty}|max:{max_u}")
        if price > 50000 and dx:
            atc1_dx_ok = {"L": {"C", "D", "N", "G", "M"}, "B": {"D", "N", "K"}}
            if atc1 in atc1_dx_ok and dx[:1] not in atc1_dx_ok[atc1]: fire("R04", 30, f"High-value drug ({price:,.0f} RWF) with unrelated diagnosis {dx}", f"{d_code}|price:{price:,.0f}|dx:{dx}")
        if atc1 == "L" and atc3.startswith("L01"):
            is_cancer_dx = dx.startswith("C") or (dx.startswith("D") and dx[1:3].isdigit() and int(dx[1:3]) <= 49)
            if dx and not is_cancer_dx: fire("R05", 25, f"Cytotoxic drug without cancer diagnosis (ICD:{dx})", f"{d_code}|dx:{dx}")
        if "PSYCH" in instr.upper() and dx:
            is_mental = dx.startswith("F") or (dx.startswith("G4") and len(dx) >= 3 and "0" <= dx[2] <= "7")
            if not is_mental: fire("R06", 20, f"PSYCH drug without psychiatric/neuro diagnosis (ICD:{dx})", f"{d_code}|instr:{instr}|dx:{dx}")
        if pid and d_code and min_r and date_col:
            dates = refill_index.get((pid, d_code), []); cur_dt = pd.to_datetime(row.get(date_col), errors="coerce")
            if len(dates) >= 2 and pd.notna(cur_dt):
                prior = [d for d in dates if d < cur_dt]
                if prior:
                    gap = (cur_dt - max(prior)).days
                    if gap < min_r: fire("R07", 40, f"Refill {gap}d after last dispense (min:{min_r}d)", f"{d_code}|gap:{gap}d|min:{min_r}d")
        if d_code and not drug_info and d_code.upper().startswith("RHIC"): fire("R08", 15, f"Procedure {d_code} not found in RAMA tariff", f"{d_code}")
        if atc3.startswith("L04") and dx:
            if not any(dx.startswith(p) for p in ("T86","M0","M1","M2","M3","K50","K51","K52","N04","L40","L41","G35")): fire("R10", 20, f"Immunosuppressant without transplant/autoimmune dx ({dx})", f"{d_code}|dx:{dx}")
        decision = "BLOCK" if score >= 75 else "HOLD" if score >= 50 else "FLAG" if score >= 30 else "APPROVE"
        risk = "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MEDIUM" if score >= 30 else "LOW"
        summary["decisions"][decision] += 1
        if decision in ("HOLD", "BLOCK"): summary["total_flagged_amount"] += amt
        rows_out.append({"_score": score, "_risk": risk, "_decision": decision, "_rules_fired": "; ".join(f"{r['id']}(+{r['score']})" for r in fired) if fired else "—", "_reasons": " | ".join(r["reason"] for r in fired) if fired else "—", "_n_rules": len(fired)})
    results = pd.DataFrame(rows_out, index=df.index); out_df = pd.concat([df, results], axis=1)
    summary["flagged_count"] = summary["decisions"]["FLAG"] + summary["decisions"]["HOLD"] + summary["decisions"]["BLOCK"]
    summary["rules_with_most_fires"] = sorted([(k, v) for k, v in summary["rule_counts"].items() if v > 0], key=lambda x: -x[1])[:5]
    return out_df, summary
