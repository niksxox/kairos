"""
Tests for Availability Engine and Interval Merging
"""

import pytest
import pandas as pd
from src.availability import IntervalMerger, AvailabilityEngine


def test_overlapping_interval_detection():
    # [9, 11) and [10, 12) must overlap
    assert IntervalMerger.intervals_overlap((9.0, 11.0), (10.0, 12.0)) is True
    # [9, 10) and [11, 12) must not overlap
    assert IntervalMerger.intervals_overlap((9.0, 10.0), (11.0, 12.0)) is False


def test_adjacent_interval_handling():
    # Half-open intervals [9, 11) and [11, 13) meet at boundary 11.0.
    # Without buffer, they DO NOT overlap.
    assert IntervalMerger.intervals_overlap((9.0, 11.0), (11.0, 13.0), buffer_hours=0.0) is False
    
    # With a 10-minute buffer (0.1667 hours), they DO overlap
    assert IntervalMerger.intervals_overlap((9.0, 11.0), (11.0, 13.0), buffer_hours=10/60.0) is True


def test_interval_merging_without_double_counting():
    # Example from DATATHON spec:
    # Class: 09:00 - 11:00
    # Event: 10:00 - 12:00
    # Combined scheduled interval: 09:00 - 12:00
    # Occupied duration: 3 hours, not 4 hours
    intervals = [(9.0, 11.0), (10.0, 12.0)]
    merged = IntervalMerger.merge_intervals(intervals)
    assert len(merged) == 1
    assert merged[0] == (9.0, 12.0)
    
    total_hours = sum(e - s for s, e in merged)
    assert total_hours == 3.0  # Exactly 3 hours, avoiding double-counting the 10:00-11:00 hour


def test_availability_engine_conflict_detection():
    # Set up class schedule on Friday for R001: 09:00 - 11:00
    sched_df = pd.DataFrame([{
        "schedule_id": "S001",
        "room_id": "R001",
        "weekday": "Fri",
        "start_hour": 9,
        "duration_hours": 2,
        "expected_attendance": 30
    }])
    # Set up event on 2025-08-01 (which is a Friday) for R001: 14:00 - 16:00
    events_df = pd.DataFrame([{
        "event_id": "EV001",
        "room_id": "R001",
        "start_time": "2025-08-01 14:00:00",
        "duration_hours": 2,
        "event_date": "2025-08-01",
        "start_hour": 14,
        "expected_attendance": 40
    }])
    
    engine = AvailabilityEngine(sched_df, events_df)

    # 1. Available in morning before class: 08:00 - 09:00
    avail, reason = engine.is_room_available("R001", "2025-08-01", 8.0, 1.0)
    assert avail is True
    assert reason is None

    # 2. Conflicting with class: 09:00 - 10:00
    avail, reason = engine.is_room_available("R001", "2025-08-01", 9.0, 1.0)
    assert avail is False
    assert "Occupied" in reason

    # 3. Available between class and event: 11:00 - 14:00
    avail, reason = engine.is_room_available("R001", "2025-08-01", 11.0, 2.0)
    assert avail is True

    # 4. Conflicting with event: 15:00 - 17:00
    avail, reason = engine.is_room_available("R001", "2025-08-01", 15.0, 2.0)
    assert avail is False
