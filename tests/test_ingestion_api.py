"""
Unit tests for Data Ingestion Pipeline & FastAPI REST API
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.ingestion import DataIngestionService


@pytest.fixture
def api_client():
    return TestClient(app)


def test_ingestion_service_schema_validation():
    service = DataIngestionService(existing_rooms={"R001", "R002"})

    # 1. Missing required field
    bad_batch = [{"room_id": "R003"}]  # missing building, capacity
    res = service.ingest_payload("rooms", bad_batch)
    assert res["status"] == "FAILED"
    assert res["total_rejected"] == 1

    # 2. Duplicate identifier
    dup_batch = [
        {"room_id": "R003", "building": "East", "capacity": 50},
        {"room_id": "R003", "building": "East", "capacity": 50}
    ]
    res_dup = service.ingest_payload("rooms", dup_batch)
    assert res_dup["status"] == "PARTIAL"
    assert res_dup["total_accepted"] == 1
    assert res_dup["total_rejected"] == 1

    # 3. Orphan room_id
    orphan_sched = [{
        "schedule_id": "CS999", "room_id": "NON_EXISTENT_ROOM",
        "course_code": "CS101", "weekday": "Mon", "start_hour": 10.0, "duration_hours": 1.0
    }]
    res_orphan = service.ingest_payload("class_schedule", orphan_sched)
    assert res_orphan["status"] == "FAILED"
    assert "Referential integrity violation" in res_orphan["errors"][0]


def test_api_health_endpoint(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["product"] == "KAIROS"
    assert data["rooms_in_inventory"] > 0


def test_api_data_sources_endpoint(api_client):
    r = api_client.get("/api/data/sources")
    assert r.status_code == 200
    data = r.json()
    assert "sources" in data
    assert len(data["sources"]) == 4


def test_api_booking_lifecycle(api_client):
    # Create a booking
    payload = {
        "username": "student",
        "room_id": "R001",
        "request_date": "2025-08-04",
        "start_hour": 14.0,
        "duration_hours": 2.0,
        "required_capacity": 30,
        "purpose": "Final Database Test Exam",
        "auto_approve": False
    }
    r = api_client.post("/api/bookings", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "request_id" in data
    req_id = data["request_id"]

    # Cancel booking
    r_cancel = api_client.post(f"/api/bookings/{req_id}/cancel", json={"username": "student"})
    assert r_cancel.status_code == 200
    assert r_cancel.json()["status"] == "Cancelled"
