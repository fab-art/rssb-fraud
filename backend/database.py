from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum
from datetime import datetime
import uuid

Base = declarative_base()

class CaseStatus(enum.Enum):
    OPEN = "OPEN"
    REVIEWING = "REVIEWING"
    CLOSED = "CLOSED"
    ARCHIVED = "ARCHIVED"

class VerificationStatus(enum.Enum):
    PENDING = "PENDING"
    FLAGGED = "FLAGGED"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class MatchStatus(enum.Enum):
    MATCHED = "MATCHED"
    UNLINKED = "UNLINKED"
    NO_RECORD = "NO_RECORD"

def generate_uuid():
    return str(uuid.uuid4())

class Case(Base):
    __tablename__ = "cases"
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    status = Column(Enum(CaseStatus), default=CaseStatus.OPEN)
    createdAt = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)

    pharmacyRecords = relationship("PharmacyRecord", back_populates="case", cascade="all, delete-orphan")
    facilityRecords = relationship("FacilityRecord", back_populates="case", cascade="all, delete-orphan")

class PharmacyRecord(Base):
    __tablename__ = "pharmacy_records"
    id = Column(String, primary_key=True, default=generate_uuid)
    caseId = Column(String, ForeignKey("cases.id"))

    voucherId = Column(String)
    patientId = Column(String)
    patientName = Column(String)
    visitDate = Column(DateTime)
    amount = Column(Float)
    insuranceCopay = Column(Float)
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    deductionAmount = Column(Float, default=0.0)
    deductionReason = Column(String)

    case = relationship("Case", back_populates="pharmacyRecords")
    match = relationship("Match", back_populates="pharmacyRecord", uselist=False, cascade="all, delete-orphan")

class FacilityRecord(Base):
    __tablename__ = "facility_records"
    __tablename__ = "facility_records"
    id = Column(String, primary_key=True, default=generate_uuid)
    caseId = Column(String, ForeignKey("cases.id"))
    ramaNumber = Column(String)
    patientName = Column(String)
    visitDate = Column(DateTime)
    sourceFile = Column(String)

    case = relationship("Case", back_populates="facilityRecords")

class Match(Base):
    __tablename__ = "matches"
    id = Column(String, primary_key=True, default=generate_uuid)
    pharmacyRecordId = Column(String, ForeignKey("pharmacy_records.id"))
    status = Column(Enum(MatchStatus))
    confidence = Column(Float)
    daysApart = Column(Float)

    pharmacyRecord = relationship("PharmacyRecord", back_populates="match")

# SQLite for demo/local persistence
engine = create_engine("sqlite:///pharmascan.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
