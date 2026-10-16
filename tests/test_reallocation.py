"""
Unit tests for Reallocation & Bounded Swap-Chain Engine
"""

import pytest
import pandas as pd
from pathlib import Path
from src.database import CampusDatabase
from src.reallocation_engine import ReallocationEngine


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_realloc.db"
    return CampusDatabase(db_path=db_file)


def test_mutual_swap_detection():
    # Room 1: 30 seats, Room 2: 100 seats
    rooms = pd.DataFrame([
        {"room_id": "R001", "building": "West", "floor": 1, "room_type": "Classroom", "capacity": 30, "projector": 1, "computers": 0, "accessible": 1, "power_outlets": 8},
        {"room_id": "R002", "building": "East", "floor": 1, "room_type": "Lecture Hall", "capacity": 100, "projector": 1, "computers": 0, "accessible": 1, "power_outlets": 12}
    ])

    # Class A: 75 students booked in Room 1 (severely overflowing!)
    # Class B: 20 students booked in Room 2 (severely underutilized!)
    # Both at Mon 10:00 (2h duration)
    sched = pd.DataFrame([
        {"schedule_id": "CS1", "room_id": "R001", "course_code": "CS301", "department": "CSE", "weekday": "Mon", "start_hour": 10.0, "duration_hours": 2.0, "expected_attendance": 75},
        {"schedule_id": "CS2", "room_id": "R002", "course_code": "HUM101", "department": "Humanities", "weekday": "Mon", "start_hour": 10.0, "duration_hours": 2.0, "expected_attendance": 20}
    ])
    events = pd.DataFrame(columns=["event_id", "room_id", "event_type", "event_date", "start_hour", "duration_hours", "expected_attendance"])

    engine = ReallocationEngine(rooms, sched, events)
    swaps = engine.detect_mutual_swaps(max_proposals=5)

    assert len(swaps) == 1
    s = swaps[0]
    assert s["proposal_type"] == "Mutual_Swap"
    assert s["seat_mismatch_reduction"] > 0
    assert len(s["items"]) == 2
    # Verify proposed rooms swapped
    assert s["items"][0]["proposed_room_id"] == "R002"
    assert s["items"][1]["proposed_room_id"] == "R001"


def test_reallocation_proposal_lifecycle(test_db):
    rooms = pd.DataFrame([
        {"room_id": "R001", "building": "West", "capacity": 30, "projector": 1, "computers": 0, "accessible": 1, "power_outlets": 8},
        {"room_id": "R002", "building": "East", "capacity": 100, "projector": 1, "computers": 0, "accessible": 1, "power_outlets": 12}
    ])
    sched = pd.DataFrame([
        {"schedule_id": "CS1", "room_id": "R001", "course_code": "CS301", "department": "CSE", "weekday": "Mon", "start_hour": 10.0, "duration_hours": 2.0, "expected_attendance": 75},
        {"schedule_id": "CS2", "room_id": "R002", "course_code": "HUM101", "department": "Humanities", "weekday": "Mon", "start_hour": 10.0, "duration_hours": 2.0, "expected_attendance": 20}
    ])
    events = pd.DataFrame(columns=["event_id", "room_id", "event_type", "event_date", "start_hour", "duration_hours", "expected_attendance"])

    engine = ReallocationEngine(rooms, sched, events, database=test_db)
    proposals = engine.generate_all_proposals(date_str="2025-08-04", max_proposals=2)
    assert len(proposals) >= 1

    prop_id = proposals[0]["proposal_id"]
    # Check proposal was persisted in DB
    stored_prop = test_db.get_proposal_by_id(prop_id)
    assert stored_prop is not None
    assert stored_prop["status"] == "Pending"

    # Approve proposal
    ok, msg = test_db.approve_reallocation_proposal(prop_id, "admin")
    assert ok is True
    updated = test_db.get_proposal_by_id(prop_id)
    assert updated["status"] == "Approved"
