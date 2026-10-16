"""
Tests for Room Matching Engine, Hard Constraints, and Explanations
"""

import pytest
import pandas as pd
from src.matching import RoomMatchingEngine
from src.availability import AvailabilityEngine


@pytest.fixture
def test_matching_setup():
    rooms_df = pd.DataFrame([
        {
            "room_id": "R001",
            "building": "Central",
            "floor": 1,
            "room_type": "Classroom",
            "capacity": 30,
            "projector": 1,
            "computers": 0,
            "accessible": 1,
            "power_outlets": 10
        },
        {
            "room_id": "R002",
            "building": "West",
            "floor": 2,
            "room_type": "Lab",
            "capacity": 60,
            "projector": 1,
            "computers": 1,
            "accessible": 0,
            "power_outlets": 25
        },
        {
            "room_id": "R003",
            "building": "North",
            "floor": 3,
            "room_type": "Auditorium",
            "capacity": 150,
            "projector": 1,
            "computers": 0,
            "accessible": 1,
            "power_outlets": 30
        }
    ])
    sched_df = pd.DataFrame([
        {
            "schedule_id": "S001",
            "room_id": "R001",
            "weekday": "Mon",
            "start_hour": 10,
            "duration_hours": 2,
            "expected_attendance": 25
        }
    ])
    events_df = pd.DataFrame([])

    avail = AvailabilityEngine(sched_df, events_df, rooms_df)
    engine = RoomMatchingEngine(rooms_df, avail)
    return engine


def test_capacity_filtering(test_matching_setup):
    engine = test_matching_setup
    # Request 40 seats on 2025-08-04 (Mon) at 14:00
    res = engine.evaluate_request(
        request_date="2025-08-04",
        start_hour=14.0,
        duration_hours=1.0,
        required_capacity=40
    )
    assert res["success"] is True
    recs = res["recommendations"]
    rec_ids = [r["room_id"] for r in recs]
    # R001 has only 30 capacity -> must be excluded from recommendations
    assert "R001" not in rec_ids
    assert "R002" in rec_ids
    assert "R003" in rec_ids


def test_facility_filtering(test_matching_setup):
    engine = test_matching_setup
    # Request requiring computers
    res = engine.evaluate_request(
        request_date="2025-08-04",
        start_hour=14.0,
        duration_hours=1.0,
        required_capacity=20,
        computers_required=True
    )
    rec_ids = [r["room_id"] for r in res["recommendations"]]
    # Only R002 has computers
    assert rec_ids == ["R002"]


def test_accessibility_filtering(test_matching_setup):
    engine = test_matching_setup
    # Request requiring accessibility
    res = engine.evaluate_request(
        request_date="2025-08-04",
        start_hour=14.0,
        duration_hours=1.0,
        required_capacity=20,
        accessible_required=True
    )
    rec_ids = [r["room_id"] for r in res["recommendations"]]
    # R002 is accessible=0 -> excluded
    assert "R002" not in rec_ids
    assert "R001" in rec_ids
    assert "R003" in rec_ids


def test_unavailable_room_exclusion(test_matching_setup):
    engine = test_matching_setup
    # Request on Mon at 10:00 (R001 is booked 10:00-12:00)
    res = engine.evaluate_request(
        request_date="2025-08-04",  # A Monday
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=25
    )
    rec_ids = [r["room_id"] for r in res["recommendations"]]
    assert "R001" not in rec_ids
    # Check that rejection reason for R001 mentions occupation
    r001_rejection = next(r for r in res["rejected"] if r["room_id"] == "R001")
    assert any("Occupied" in reason for reason in r001_rejection["rejection_reasons"])


def test_request_with_no_feasible_room(test_matching_setup):
    engine = test_matching_setup
    # Request 300 seats (max on campus is 150)
    res = engine.evaluate_request(
        request_date="2025-08-04",
        start_hour=14.0,
        duration_hours=1.0,
        required_capacity=300
    )
    assert res["success"] is False
    assert "exceeds largest available room" in res["error"]
    assert len(res["recommendations"]) == 0
