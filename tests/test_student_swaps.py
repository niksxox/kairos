"""
Unit and integration tests for Student Room Swap Requests with ML Model Evaluation.
"""

import pytest
import pandas as pd
from starlette.testclient import TestClient

from src.database import CampusDatabase
from src.ml_model import DemandForecaster, CustomLinearRegression
from src.api import app


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_swaps.db"
    return CampusDatabase(db_path=db_file)


@pytest.fixture
def api_client():
    return TestClient(app)


def test_ml_evaluate_swap_request():
    forecaster = DemandForecaster()
    # Mock trained model
    forecaster.model = CustomLinearRegression()
    forecaster.model.weights = [0.0] * 10
    forecaster.model.bias = 4.2  # High demand hour
    forecaster.model.scaler_mean = [0.0] * 10
    forecaster.model.scaler_std = [1.0] * 10

    rooms_df = pd.DataFrame([
        {"room_id": "R001", "capacity": 100},
        {"room_id": "R002", "capacity": 35}
    ])

    # Case 1: Student is in 100-seat room with only 30 students during high demand
    # ML model should recommend swapping to 35-seat room to free up 100-seat hall
    res = forecaster.evaluate_swap_request(
        current_room_id="R001",
        desired_room_id="R002",
        request_date="2025-09-15",
        start_hour=14.0,
        duration_hours=2.0,
        required_capacity=30,
        reason="Voluntarily moving to smaller room",
        rooms_df=rooms_df
    )

    assert res["should_swap"] is True
    assert "Recommended" in res["recommendation"]
    assert res["predicted_demand"] == 4.2
    assert "frees high-capacity room" in res["reasoning"].lower()


def test_student_swap_lifecycle(test_db):
    # 1. Create a booking
    req_id = test_db.create_booking_request(
        username="student_alex",
        room_id="R001",
        request_date="2025-10-01",
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=25,
        purpose="Robotics Workshop"
    )

    # 2. Student requests a swap to R002
    swap_id = test_db.create_student_swap_request(
        request_id=req_id,
        username="student_alex",
        current_room_id="R001",
        desired_room_id="R002",
        request_date="2025-10-01",
        start_hour=10.0,
        duration_hours=2.0,
        required_capacity=30,
        reason="Need lab tables in R002",
        ml_recommendation="Recommended",
        ml_reasoning="Frees up R001 during predicted demand rush",
        predicted_demand=3.8
    )

    assert swap_id.startswith("SWP-")

    # 3. Retrieve swap requests
    swaps = test_db.get_student_swap_requests(username="student_alex")
    assert len(swaps) == 1
    assert swaps[0]["swap_id"] == swap_id
    assert swaps[0]["status"] == "Pending"

    # 4. Admin approves swap
    ok, msg = test_db.approve_student_swap_request(swap_id, admin_username="admin")
    assert ok is True
    assert "reassigned to Room R002" in msg

    # 5. Check booking record updated
    user_bookings = test_db.get_user_bookings("student_alex")
    updated_b = user_bookings[user_bookings["request_id"] == req_id].iloc[0]
    assert updated_b["room_id"] == "R002"

    # 6. Check swap request status is Approved
    updated_swaps = test_db.get_student_swap_requests(username="student_alex")
    assert updated_swaps[0]["status"] == "Approved"


def test_student_swap_rejection(test_db):
    req_id = test_db.create_booking_request(
        username="student_bob",
        room_id="R003",
        request_date="2025-10-02",
        start_hour=14.0,
        duration_hours=1.0,
        required_capacity=15,
        purpose="Math Help Desk"
    )

    swap_id = test_db.create_student_swap_request(
        request_id=req_id,
        username="student_bob",
        current_room_id="R003",
        desired_room_id="R004",
        request_date="2025-10-02",
        start_hour=14.0,
        duration_hours=1.0,
        required_capacity=15,
        reason="Prefer first floor",
        ml_recommendation="Not Recommended",
        ml_reasoning="Desired room R004 has severe predicted demand conflict",
        predicted_demand=5.1
    )

    ok, msg = test_db.reject_student_swap_request(swap_id, admin_username="admin", reason="Room R004 needed for exam")
    assert ok is True

    # Booking remains in R003
    user_bookings = test_db.get_user_bookings("student_bob")
    assert user_bookings.iloc[0]["room_id"] == "R003"


def test_api_student_swap_endpoints(api_client):
    import uuid
    u_name = f"swap_user_{uuid.uuid4().hex[:6]}"

    # Create a booking via API
    r_book = api_client.post("/api/bookings", json={
        "username": u_name,
        "room_id": "R001",
        "request_date": "2025-10-15",
        "start_hour": 15.0,
        "duration_hours": 2.0,
        "required_capacity": 20,
        "purpose": "AI Study Group"
    })
    assert r_book.status_code == 200
    req_id = r_book.json()["request_id"]

    # Submit swap request via API
    r_swap = api_client.post("/api/swaps/request", json={
        "request_id": req_id,
        "username": u_name,
        "desired_room_id": "R002",
        "reason": "Attendance grew to 30 students"
    })
    assert r_swap.status_code == 200
    swap_data = r_swap.json()
    assert "swap_id" in swap_data
    assert "ml_evaluation" in swap_data
    swap_id = swap_data["swap_id"]

    # Admin approves swap via API
    r_approve = api_client.post(f"/api/swaps/{swap_id}/approve", json={
        "admin_username": "admin"
    })
    assert r_approve.status_code == 200
    assert r_approve.json()["status"] == "Approved"
