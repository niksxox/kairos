"""
Tests for Multi-Event Room Optimizer and Comparative Evaluation
"""

import pytest
import pandas as pd
from src.optimizer import MultiEventOptimizer


@pytest.fixture
def optimizer_test_fixture():
    # Deterministic test fixture
    # 2 rooms: R001 (50 cap), R002 (100 cap)
    rooms = pd.DataFrame([
        {"room_id": "R001", "building": "Central", "floor": 1, "room_type": "Classroom", "capacity": 50, "projector": 1, "computers": 0, "accessible": 1, "power_outlets": 10},
        {"room_id": "R002", "building": "North", "floor": 2, "room_type": "Lecture Hall", "capacity": 100, "projector": 1, "computers": 1, "accessible": 1, "power_outlets": 20}
    ])
    sched = pd.DataFrame([])
    events = pd.DataFrame([])
    optimizer = MultiEventOptimizer(rooms, sched, events)
    return optimizer


def test_competing_requests_mutual_exclusion(optimizer_test_fixture):
    optimizer = optimizer_test_fixture
    # Two requests at the exact same time competing for R001
    # Q1: 40 people at 10:00 (fits R001 and R002)
    # Q2: 45 people at 10:00 (fits R001 and R002)
    # Q3: 80 people at 10:00 (only fits R002)
    # Total rooms: 2. Requests: 3. Max accommodated: 2.
    requests = pd.DataFrame([
        {"request_id": "Q1", "request_date": "2025-08-04", "start_hour": 10, "duration_hours": 2, "required_capacity": 40},
        {"request_id": "Q2", "request_date": "2025-08-04", "start_hour": 10, "duration_hours": 2, "required_capacity": 45},
        {"request_id": "Q3", "request_date": "2025-08-04", "start_hour": 10, "duration_hours": 2, "required_capacity": 80}
    ])

    result = optimizer.optimize_allocations(requests)
    
    assert result["hard_constraint_violations"] == 0
    assert result["allocated_count"] == 2
    assert result["unallocated_count"] == 1
    
    # Q3 must get R002 because only R002 can fit 80 people!
    allocs = {a["request_id"]: a["room_id"] for a in result["allocations"]}
    assert allocs["Q3"] == "R002"
    # One of Q1 or Q2 gets R001
    assert "Q1" in allocs or "Q2" in allocs
    assigned_small = "Q1" if "Q1" in allocs else "Q2"
    assert allocs[assigned_small] == "R001"


def test_optimizer_respects_all_hard_constraints(optimizer_test_fixture):
    optimizer = optimizer_test_fixture
    # Request requiring computers: only R002 has computers
    requests = pd.DataFrame([
        {"request_id": "Q_COMP", "request_date": "2025-08-04", "start_hour": 14, "duration_hours": 1, "required_capacity": 30, "computers_required": 1}
    ])
    result = optimizer.optimize_allocations(requests)
    assert result["allocated_count"] == 1
    assert result["allocations"][0]["room_id"] == "R002"
    assert result["hard_constraint_violations"] == 0


def test_baseline_vs_optimizer_comparison(optimizer_test_fixture):
    optimizer = optimizer_test_fixture
    # Requests on different time slots
    requests = pd.DataFrame([
        {"request_id": "Q1", "request_date": "2025-08-04", "start_hour": 9, "duration_hours": 2, "required_capacity": 45},
        {"request_id": "Q2", "request_date": "2025-08-04", "start_hour": 12, "duration_hours": 2, "required_capacity": 85},
        {"request_id": "Q3", "request_date": "2025-08-04", "start_hour": 15, "duration_hours": 2, "required_capacity": 30}
    ])

    comparison = optimizer.compare_optimizer_vs_baseline(requests)
    
    metrics = comparison["metrics"]
    assert metrics["optimizer"]["hard_constraint_violations"] == 0
    assert metrics["baseline"]["hard_constraint_violations"] == 0
    assert metrics["optimizer"]["assigned_count"] == 3
    assert metrics["baseline"]["assigned_count"] == 3
