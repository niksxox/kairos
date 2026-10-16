"""
Tests for Utilization and Demand vs. Supply Analytics
"""

import pytest
import pandas as pd
from src.utilization import UtilizationEngine


def test_utilization_calculations_without_double_counting():
    rooms = pd.DataFrame([
        {"room_id": "R001", "building": "Central", "floor": 1, "room_type": "Classroom", "capacity": 50}
    ])
    # 2 overlapping classes on Monday:
    # Class 1: 09:00 - 11:00 (2h)
    # Class 2: 10:00 - 12:00 (2h)
    # Merged occupied hours must be 3.0 hours (not 4.0!)
    sched = pd.DataFrame([
        {"schedule_id": "S1", "room_id": "R001", "weekday": "Mon", "start_hour": 9, "duration_hours": 2, "expected_attendance": 20},
        {"schedule_id": "S2", "room_id": "R001", "weekday": "Mon", "start_hour": 10, "duration_hours": 2, "expected_attendance": 20}
    ])
    events = pd.DataFrame([])

    engine = UtilizationEngine(rooms, sched, events, day_start=8, day_end=22)
    df = engine.compute_weekly_schedule_utilization()
    
    assert len(df) == 1
    row = df.iloc[0]
    assert row["hours_mon"] == 3.0  # Exactly 3.0 merged hours!
    assert row["weekly_scheduled_hours"] == 3.0
    # Denominator: 6 days * 14h = 84h
    expected_rate = round((3.0 / 84.0) * 100, 1)
    assert row["scheduled_utilization_rate"] == expected_rate


def test_demand_vs_supply():
    rooms = pd.DataFrame([
        {"room_id": "R1", "capacity": 25},
        {"room_id": "R2", "capacity": 50},
        {"room_id": "R3", "capacity": 150}
    ])
    reqs = pd.DataFrame([
        {"request_id": "Q1", "required_capacity": 20},
        {"request_id": "Q2", "required_capacity": 45},
        {"request_id": "Q3", "required_capacity": 120}
    ])
    engine = UtilizationEngine(rooms, pd.DataFrame(), pd.DataFrame())
    dvs = engine.compute_demand_vs_supply(reqs)
    
    assert "data" in dvs
    data_df = dvs["data"]
    assert len(data_df) > 0
    assert "room_supply_count" in data_df.columns
    assert "request_demand_count" in data_df.columns
