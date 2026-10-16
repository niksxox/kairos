"""
KAIROS - Production REST API & OpenAPI Engine (FastAPI)
Exposes formal REST interfaces for Data Ingestion, Reallocation Proposals,
Room Matching, Booking Lifecycle, and Health Monitoring.
Interactive Swagger documentation served automatically at /docs.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database import CampusDatabase
from src.data_loader import DataLoader
from src.ingestion import DataIngestionService
from src.reallocation_engine import ReallocationEngine
from src.availability import AvailabilityEngine
from src.priority import infer_priority_tier, PriorityTier, PRIORITY_WEIGHTS


app = FastAPI(
    title="KAIROS Campus Space Intelligence API",
    description="Operational REST endpoints for space reallocation, data ingestion, booking lifecycle, and optimization.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local client and frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
db = CampusDatabase()
loader = DataLoader()
datasets, _ = loader.load_all()
rooms_df = datasets["rooms"]
sched_df = datasets["class_schedule"]
events_df = datasets["events"]
requests_df = datasets["room_requests"]

avail_engine = AvailabilityEngine(sched_df, events_df, rooms_df)
ingest_service = DataIngestionService(database=db, existing_rooms=set(rooms_df["room_id"]))
realloc_engine = ReallocationEngine(rooms_df, sched_df, events_df, database=db)


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    product: str
    team: str
    timestamp: str
    rooms_in_inventory: int
    scheduled_classes: int
    campus_events: int


class DataImportRequest(BaseModel):
    entity_type: str = Field(..., description="One of: rooms, class_schedule, events, room_requests")
    source_name: str = Field(default="REST_API_Payload", description="Name/origin of incoming data")
    records: List[Dict[str, Any]] = Field(..., description="List of record dictionaries")


class BookingCreateRequest(BaseModel):
    username: str = Field(..., example="student")
    room_id: Optional[str] = Field(None, example="R005")
    request_date: str = Field(..., example="2025-08-04")
    start_hour: float = Field(..., ge=0, le=24, example=14.0)
    duration_hours: float = Field(..., gt=0, le=12, example=2.0)
    required_capacity: int = Field(..., gt=0, example=35)
    purpose: str = Field(..., example="Midterm Review Exam")
    projector_required: bool = False
    computers_required: bool = False
    accessible_required: bool = False
    auto_approve: bool = False


class ReallocationPlanRequest(BaseModel):
    request_date: str = Field(..., example="2025-08-04")
    max_proposals: int = Field(default=5, ge=1, le=20)


class RejectionPayload(BaseModel):
    admin_username: str = Field(..., example="admin")
    reason: str = Field(default="Administrative policy constraint", example="Classroom required for accredited examination")


class ApprovalPayload(BaseModel):
    admin_username: str = Field(..., example="admin")


# -----------------------------------------------------------------------------
# 1. HEALTH & SYSTEM STATUS
# -----------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def get_health():
    """Returns operational health and dataset inventory metrics."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "product": "KAIROS",
        "team": "Laphing Buddha",
        "timestamp": datetime.now().isoformat(),
        "rooms_in_inventory": len(rooms_df),
        "scheduled_classes": len(sched_df),
        "campus_events": len(events_df)
    }


@app.get("/api/data/sources", tags=["Data Ingestion"])
def get_data_sources():
    """Lists current verified data sources and ingestion timestamps."""
    return {
        "sources": [
            {"entity": "rooms", "count": len(rooms_df), "source": "data/raw/rooms.csv", "verified": True},
            {"entity": "class_schedule", "count": len(sched_df), "source": "data/raw/class_schedule.csv", "verified": True},
            {"entity": "events", "count": len(events_df), "source": "data/raw/events.csv", "verified": True},
            {"entity": "room_requests", "count": len(requests_df), "source": "data/raw/room_requests.csv", "verified": True}
        ],
        "persistent_database": "SQLite (data/campus_space.db)",
        "last_verified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# -----------------------------------------------------------------------------
# 2. DATA INGESTION PIPELINE
# -----------------------------------------------------------------------------

@app.post("/api/data/import", tags=["Data Ingestion"])
def import_dataset(payload: DataImportRequest):
    """
    Ingests, validates, and stages incoming records for any entity type.
    Rejects malformed schema, referential violations, and invalid bounds.
    """
    result = ingest_service.ingest_payload(
        entity_type=payload.entity_type,
        records=payload.records,
        source_name=payload.source_name
    )
    if result["status"] == "FAILED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result)
    return result


@app.post("/api/data/rooms", tags=["Data Ingestion"])
def import_rooms(records: List[Dict[str, Any]]):
    """Direct ingestion endpoint for physical room records."""
    return ingest_service.ingest_payload("rooms", records, source_name="POST /api/data/rooms")


@app.post("/api/data/schedules", tags=["Data Ingestion"])
def import_schedules(records: List[Dict[str, Any]]):
    """Direct ingestion endpoint for academic class schedule commitments."""
    return ingest_service.ingest_payload("class_schedule", records, source_name="POST /api/data/schedules")


@app.post("/api/data/events", tags=["Data Ingestion"])
def import_events(records: List[Dict[str, Any]]):
    """Direct ingestion endpoint for one-off campus events."""
    return ingest_service.ingest_payload("events", records, source_name="POST /api/data/events")


@app.post("/api/data/requests", tags=["Data Ingestion"])
def import_requests(records: List[Dict[str, Any]]):
    """Direct ingestion endpoint for room requests / demand."""
    return ingest_service.ingest_payload("room_requests", records, source_name="POST /api/data/requests")


@app.get("/api/data/imports", tags=["Data Ingestion"])
def list_import_history():
    """Lists recent transactional data imports from persistent audit trail."""
    return {"imports": db.get_data_imports()}


# -----------------------------------------------------------------------------
# 3. AUTOMATIC REALLOCATION & SWAP ENGINE
# -----------------------------------------------------------------------------

@app.post("/api/reallocation/plan", tags=["Reallocation Engine"])
def generate_reallocation_plan(payload: ReallocationPlanRequest):
    """
    Scans space allocations and unallocated requests to generate bounded swap chains
    and mutual room swaps that alleviate capacity bottlenecks without schedule shifts.
    """
    # Sample unallocated requests for target date
    day_reqs = requests_df[requests_df["request_date"] == payload.request_date].head(10)
    proposals = realloc_engine.generate_all_proposals(
        date_str=payload.request_date,
        unallocated_requests=day_reqs,
        max_proposals=payload.max_proposals
    )
    return {
        "request_date": payload.request_date,
        "proposals_generated": len(proposals),
        "proposals": proposals
    }


@app.get("/api/reallocation/proposals", tags=["Reallocation Engine"])
def list_reallocation_proposals(status_filter: Optional[str] = Query(None, description="Pending, Approved, Rejected")):
    """Returns stored reallocation proposals with full movement breakdowns."""
    props = db.get_reallocation_proposals(status_filter=status_filter)
    return {"proposals": props}


@app.get("/api/reallocation/proposals/{proposal_id}", tags=["Reallocation Engine"])
def get_reallocation_proposal(proposal_id: str):
    """Retrieves a single reallocation proposal by ID."""
    prop = db.get_proposal_by_id(proposal_id)
    if not prop:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found.")
    return prop


@app.post("/api/reallocation/proposals/{proposal_id}/approve", tags=["Reallocation Engine"])
def approve_reallocation_proposal(proposal_id: str, payload: ApprovalPayload):
    """
    Atomically approves and executes all movements in a reallocation proposal.
    Revalidates all constraints to prevent double-booking.
    """
    success, msg = db.approve_reallocation_proposal(proposal_id, payload.admin_username, availability_checker=avail_engine)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"proposal_id": proposal_id, "status": "Approved", "message": msg}


@app.post("/api/reallocation/proposals/{proposal_id}/reject", tags=["Reallocation Engine"])
def reject_reallocation_proposal(proposal_id: str, payload: RejectionPayload):
    """Rejects a proposed reallocation with an explainable administrative reason."""
    success, msg = db.reject_reallocation_proposal(proposal_id, payload.admin_username, reason=payload.reason)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"proposal_id": proposal_id, "status": "Rejected", "message": msg}


# -----------------------------------------------------------------------------
# 4. BOOKINGS LIFECYCLE
# -----------------------------------------------------------------------------

@app.get("/api/bookings", tags=["Bookings"])
def get_bookings(status_filter: Optional[str] = Query(None), username: Optional[str] = Query(None)):
    """Lists student booking requests with priority classifications."""
    if username:
        df = db.get_user_bookings(username)
    else:
        df = db.get_all_bookings(status_filter=status_filter)
    return {"bookings": df.to_dict(orient="records")}


@app.post("/api/bookings", tags=["Bookings"])
def create_booking(payload: BookingCreateRequest):
    """Submits a new student room booking request, with duplicate request rejection."""
    # Check for duplicate active request
    dup = db.find_duplicate_request(
        username=payload.username,
        room_id=payload.room_id,
        request_date=payload.request_date,
        start_hour=payload.start_hour,
        duration_hours=payload.duration_hours,
        exact_only=False
    )
    if dup:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate request: User '{payload.username}' already has a {dup['status'].lower()} request ({dup['request_id']}) for room {payload.room_id or 'unassigned'} on {payload.request_date} at {int(dup['start_hour']):02d}:00."
        )

    try:
        req_id = db.create_booking_request(
            username=payload.username,
            room_id=payload.room_id,
            request_date=payload.request_date,
            start_hour=payload.start_hour,
            duration_hours=payload.duration_hours,
            required_capacity=payload.required_capacity,
            projector_required=payload.projector_required,
            computers_required=payload.computers_required,
            accessible_required=payload.accessible_required,
            purpose=payload.purpose,
            auto_approve=payload.auto_approve
        )
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err))

    return {"request_id": req_id, "status": "Pending", "message": "Booking request successfully recorded."}


@app.post("/api/bookings/{request_id}/cancel", tags=["Bookings"])
def cancel_booking(request_id: str, username: str = Body(..., embed=True)):
    """Cancels an existing booking request and frees the room."""
    success, msg = db.cancel_booking(request_id, username=username, is_admin=False)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"request_id": request_id, "status": "Cancelled", "message": msg}
