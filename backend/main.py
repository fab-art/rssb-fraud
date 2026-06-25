import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.engine.profiling import auto_map_columns, apply_column_mapping
from backend.engine.matching import run_match
from backend.engine.rules import run_rules_engine
from backend.engine.reporting import generate_counter_verification_xlsx
from backend.database import SessionLocal, init_db, Case, PharmacyRecord, FacilityRecord, Match, CaseStatus, VerificationStatus, MatchStatus

init_db()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class VerificationUpdate(BaseModel):
    status: str
    deductionAmount: Optional[float] = 0
    deductionReason: Optional[str] = ""

@app.post("/api/cases")
async def create_case(name: str = Form(...), db: Session = Depends(get_db)):
    db_case = Case(name=name, meta={"pharmacy": "PHARMACIE VINCA GISENYI LTD"})
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case

@app.post("/api/cases/{case_id}/upload-pharmacy")
async def upload_pharmacy(case_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents)) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(io.BytesIO(contents))
    mapping, _, _ = auto_map_columns(df)
    mapped_df = apply_column_mapping(df, mapping)
    processed_df, _ = run_rules_engine(mapped_df)

    for _, row in processed_df.iterrows():
        db_rec = PharmacyRecord(
            caseId=case_id,
            voucherId=str(row.get("voucher_id", "")),
            patientId=str(row.get("patient_id", "")),
            patientName=str(row.get("patient_name", "")),
            visitDate=pd.to_datetime(row.get("visit_date")).to_pydatetime() if pd.notna(row.get("visit_date")) else None,
            amount=float(row.get("amount", 0)),
            insuranceCopay=float(row.get("insurance_copay", 0)),
            status=VerificationStatus.PENDING
        )
        db.add(db_rec)
    db.commit()
    return {"message": "Uploaded"}

@app.post("/api/cases/{case_id}/upload-facility")
async def upload_facility(case_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents)) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(io.BytesIO(contents))
    for _, row in df.iterrows():
        db_rec = FacilityRecord(
            caseId=case_id,
            ramaNumber=str(row.get("RAMA Number", "")),
            patientName=str(row.get("Patient Name", "")),
            visitDate=pd.to_datetime(row.get("Date")).to_pydatetime() if pd.notna(row.get("Date")) else None,
            sourceFile=file.filename
        )
        db.add(db_rec)
    db.commit()
    return {"message": "Uploaded"}

@app.post("/api/cases/{case_id}/process")
async def process_case(case_id: str, db: Session = Depends(get_db)):
    ph_recs = db.query(PharmacyRecord).filter(PharmacyRecord.caseId == case_id).all()
    fac_recs = db.query(FacilityRecord).filter(FacilityRecord.caseId == case_id).all()

    ph_df = pd.DataFrame([{ "patient_id": r.patientId, "patient_name": r.patientName, "visit_date": r.visitDate, "voucher_id": r.voucherId, "insurance_copay": r.insuranceCopay, "amount": r.amount, "id": r.id } for r in ph_recs])
    fac_df = pd.DataFrame([{ "_rama": r.ramaNumber, "_name": r.patientName, "_date": r.visitDate, "_source": r.sourceFile } for r in fac_recs])

    ph_work = ph_df.copy()
    ph_work["_rama"] = ph_work["patient_id"]; ph_work["_name"] = ph_work["patient_name"]; ph_work["_date"] = ph_work["visit_date"]
    ph_work["_vou"] = ph_work["voucher_id"]; ph_work["_ins"] = ph_work["insurance_copay"]; ph_work["_tot"] = ph_work["amount"]
    ph_work["_doc"] = ""; ph_work["_dpt"] = ""

    matches_df = run_match(ph_work, fac_df)

    for i, row in matches_df.iterrows():
        ph_id = ph_df.iloc[i]["id"]
        status_str = row["status"]
        db_match = Match(
            pharmacyRecordId=ph_id,
            status=MatchStatus[status_str],
            confidence=float(row["confidence"]),
            daysApart=float(row["days_apart"]) if pd.notna(row["days_apart"]) else None
        )
        db.add(db_match)

    db.query(Case).filter(Case.id == case_id).update({"status": CaseStatus.REVIEWING})
    db.commit()
    return {"message": "Processed"}

@app.post("/api/cases/{case_id}/records/{voucher_id}")
async def update_record(case_id: str, voucher_id: str, update: VerificationUpdate, db: Session = Depends(get_db)):
    db.query(PharmacyRecord).filter(PharmacyRecord.caseId == case_id, PharmacyRecord.voucherId == voucher_id).update({
        "status": VerificationStatus[update.status],
        "deductionAmount": update.deductionAmount,
        "deductionReason": update.deductionReason
    })
    db.commit()
    return {"message": "Updated"}

@app.get("/api/cases/{case_id}/report")
async def export_report(case_id: str, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.id == case_id).first()
    ph_recs = db.query(PharmacyRecord).filter(PharmacyRecord.caseId == case_id).all()
    df = pd.DataFrame([{ "voucher_id": r.voucherId, "patient_id": r.patientId, "patient": r.patientName, "visit_date": r.visitDate, "insurance_copay": r.insuranceCopay, "amount": r.amount } for r in ph_recs])
    deductions = [{ "paper_code": r.voucherId, "rama_no": r.patientId, "amount": -abs(r.deductionAmount), "explanation": r.deductionReason } for r in ph_recs if r.deductionAmount > 0]
    xlsx_bytes = generate_counter_verification_xlsx(df=df, deductions=deductions, meta=case.meta, prepared_by="Analyst", verified_by="Manager", approved_by="Director", pc_col="voucher_id", ins_col="insurance_copay", tot_col="amount")
    return Response(content=xlsx_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=report_{case_id}.xlsx"})

@app.get("/api/cases/{case_id}")
async def get_case(case_id: str, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.id == case_id).first()
    ph_recs = db.query(PharmacyRecord).filter(PharmacyRecord.caseId == case_id).all()
    # Serialize for JSON
    res = { "id": case.id, "name": case.name, "status": case.status.value, "createdAt": case.createdAt.isoformat(), "pharmacyRecords": [] }
    for r in ph_recs:
        m = db.query(Match).filter(Match.pharmacyRecordId == r.id).first()
        res["pharmacyRecords"].append({
            "id": r.id, "voucher_id": r.voucherId, "patient_id": r.patientId, "patient_name": r.patientName, "visit_date": r.visitDate.isoformat() if r.visitDate else None, "amount": r.amount, "insurance_copay": r.insuranceCopay, "status": r.status.value, "deductionAmount": r.deductionAmount, "deductionReason": r.deductionReason,
            "match": { "status": m.status.value, "confidence": m.confidence, "days_apart": m.daysApart } if m else None
        })
    return res

@app.get("/api/cases")
async def list_cases(db: Session = Depends(get_db)):
    return db.query(Case).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
