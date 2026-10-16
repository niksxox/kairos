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
