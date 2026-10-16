"""
Tests for SQLite Database Persistence, Role Authentication, and Atomic Conflict Revalidation
"""

import pytest
import sqlite3
import pandas as pd
from src.database import CampusDatabase
from src.auth import hash_password, verify_password


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_campus.db"
    return CampusDatabase(db_path=db_file)


def test_default_accounts_and_authentication(test_db):
    # Test valid credentials
    student = test_db.authenticate("student", "student123")
    assert student is not None
    assert student["role"] == "student"

    admin = test_db.authenticate("admin", "admin123")
    assert admin is not None
    assert admin["role"] == "admin"

    # Test invalid credentials
    invalid = test_db.authenticate("student", "wrong_password")
    assert invalid is None


def test_booking_creation_cancellation_and_availability_restoration(test_db):
    # Student creates a booking request
    req_id = test_db.create_booking_request(
        username="student",
        room_id="R001",
        request_date="2025-09-15",
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=30,
        purpose="Project work"
    )
    assert req_id.startswith("K-REQ-")

    bookings = test_db.get_user_bookings("student")
    assert len(bookings) == 1
    assert bookings.iloc[0]["status"] == "Pending"

    # Student cancels their own booking
    ok, msg = test_db.cancel_booking(req_id, username="student")
    assert ok is True
    assert "successfully cancelled" in msg

    bookings_after = test_db.get_user_bookings("student")
    assert bookings_after.iloc[0]["status"] == "Cancelled"


def test_unauthorized_cancellation_prevention(test_db):
    req_id = test_db.create_booking_request(
        username="student",
        room_id="R001",
        request_date="2025-09-15",
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=30
    )
    # Different user tries to cancel
    ok, msg = test_db.cancel_booking(req_id, username="another_user", is_admin=False)
    assert ok is False
    assert "Unauthorized" in msg


def test_admin_approval_and_collision_prevention(test_db):
    # Request 1 for R001 on 2025-09-15 at 14:00 - 16:00
    req1 = test_db.create_booking_request(
        username="student",
        room_id="R001",
        request_date="2025-09-15",
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=25
    )

    # Request 2 also for R001 on 2025-09-15 overlapping at 15:00 - 17:00
    req2 = test_db.create_booking_request(
        username="student",
        room_id="R001",
        request_date="2025-09-15",
        start_hour=15.0,
        duration_hours=2.0,
        required_capacity=20
    )

    # Admin approves req1
    ok1, msg1 = test_db.approve_booking(req1, admin_username="admin", availability_checker=None)
    assert ok1 is True
    assert "confirmed" in msg1

    # Admin attempts to approve req2 for the same room -> COLLISION REJECTION
    ok2, msg2 = test_db.approve_booking(req2, admin_username="admin", availability_checker=None)
    assert ok2 is False
    assert "Approval collision" in msg2


def test_duplicate_request_prevention(test_db):
    import pytest

    # 1. First booking request
    req1 = test_db.create_booking_request(
        username="student_dup",
        room_id="R002",
        request_date="2025-09-20",
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=30,
        purpose="Midterm Study"
    )
    assert req1.startswith("K-REQ-")

    # 2. Duplicate detection helper should find it
    dup = test_db.find_duplicate_request(
        username="student_dup",
        room_id="R002",
        request_date="2025-09-20",
        start_hour=10.0,
        duration_hours=2.0,
        exact_only=True
    )
    assert dup is not None
    assert dup["request_id"] == req1

    # Overlapping duplicate detection should also find it
    dup_overlap = test_db.find_duplicate_request(
        username="student_dup",
        room_id="R002",
        request_date="2025-09-20",
        start_hour=10.5,
        duration_hours=1.0,
        exact_only=False
    )
    assert dup_overlap is not None
    assert dup_overlap["request_id"] == req1

    # 3. Submitting an exact duplicate active request should raise ValueError
    with pytest.raises(ValueError, match="Duplicate request"):
        test_db.create_booking_request(
            username="student_dup",
            room_id="R002",
            request_date="2025-09-20",
            start_hour=10.0,
            duration_hours=2.0,
            required_capacity=30,
            purpose="Midterm Study"
        )


def test_duplicate_request_allowed_after_cancellation(test_db):
    # 1. Create request
    req = test_db.create_booking_request(
        username="student_rebook",
        room_id="R003",
        request_date="2025-09-22",
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=20
    )

    # 2. Cancel request
    ok, _ = test_db.cancel_booking(req, username="student_rebook")
    assert ok is True

    # 3. Duplicate finder should ignore cancelled requests
    dup = test_db.find_duplicate_request(
        username="student_rebook",
        room_id="R003",
        request_date="2025-09-22",
        start_hour=14.0,
        duration_hours=2.0,
        exact_only=True
    )
    assert dup is None

    # 4. Re-booking the room for that slot should succeed now
    new_req = test_db.create_booking_request(
        username="student_rebook",
        room_id="R003",
        request_date="2025-09-22",
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=20
    )
    assert new_req.startswith("K-REQ-")
    assert new_req != req

