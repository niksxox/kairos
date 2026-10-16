"""
Unit tests for Institutional Priority & Importance Engine
"""

import pytest
import pandas as pd
from src.priority import infer_priority_tier, get_priority_info, PriorityTier, PRIORITY_WEIGHTS
from src.optimizer import MultiEventOptimizer


def test_priority_tier_inference():
    # Exams / Tests -> Critical
    assert infer_priority_tier("Final Midterm Exam") == PriorityTier.CRITICAL
    assert infer_priority_tier("Accreditation Assessment Test") == PriorityTier.CRITICAL
    assert infer_priority_tier("Organic Chemistry Quiz") == PriorityTier.CRITICAL

    # Core academic lectures -> High
    assert infer_priority_tier("CS101 Core Lecture") == PriorityTier.HIGH
    assert infer_priority_tier("Physics Laboratory Session") == PriorityTier.HIGH

    # Seminars & workshops -> Medium
    assert infer_priority_tier("Robotics Research Seminar") == PriorityTier.MEDIUM
    assert infer_priority_tier("Department Faculty Workshop") == PriorityTier.MEDIUM

    # Study groups & clubs -> Standard
    assert infer_priority_tier("Study Group Review") == PriorityTier.STANDARD
    assert infer_priority_tier("Hackathon Team Sprint") == PriorityTier.STANDARD

    # Cultural & social events -> Flexible
    assert infer_priority_tier("Annual Cultural Fest Rehearsal") == PriorityTier.FLEXIBLE
    assert infer_priority_tier("Music Society Social Gathering") == PriorityTier.FLEXIBLE


def test_priority_weights_ordering():
    assert PRIORITY_WEIGHTS[PriorityTier.CRITICAL] > PRIORITY_WEIGHTS[PriorityTier.HIGH]
    assert PRIORITY_WEIGHTS[PriorityTier.HIGH] > PRIORITY_WEIGHTS[PriorityTier.MEDIUM]
    assert PRIORITY_WEIGHTS[PriorityTier.MEDIUM] > PRIORITY_WEIGHTS[PriorityTier.STANDARD]
    assert PRIORITY_WEIGHTS[PriorityTier.STANDARD] > PRIORITY_WEIGHTS[PriorityTier.FLEXIBLE]


def test_optimizer_prioritizes_exam_over_cultural_event():
    """
    When only 1 room is available, CP-SAT must allocate the room to the Exam
    and leave the Cultural Event unallocated.
    """
    rooms = pd.DataFrame([{
        "room_id": "R001", "building": "Main", "floor": 1, "room_type": "Classroom",
        "capacity": 50, "projector": 1, "computers": 0, "accessible": 1, "power_outlets": 10
    }])
    sched = pd.DataFrame(columns=["schedule_id", "room_id", "course_code", "weekday", "start_hour", "duration_hours", "expected_attendance", "department"])
    events = pd.DataFrame(columns=["event_id", "room_id", "event_type", "event_date", "start_hour", "duration_hours", "expected_attendance"])

    opt = MultiEventOptimizer(rooms, sched, events)

    # Two competing requests at the exact same hour on the exact same date
    requests = pd.DataFrame([
        {
            "request_id": "REQ_CULTURAL",
            "request_date": "2025-09-15",
            "start_hour": 14.0,
            "duration_hours": 2.0,
            "required_capacity": 40,
            "purpose": "Music Fest Cultural Rehearsal"  # Flexible priority
        },
        {
            "request_id": "REQ_EXAM",
            "request_date": "2025-09-15",
            "start_hour": 14.0,
            "duration_hours": 2.0,
            "required_capacity": 40,
            "purpose": "Final Calculus Examination"    # Critical priority
        }
    ])

    res = opt.optimize_allocations(requests)
    allocated_ids = [a["request_id"] for a in res["allocations"]]
    unallocated_ids = [u["request_id"] for u in res["unallocated"]]

    assert "REQ_EXAM" in allocated_ids, "Critical Exam must be allocated"
    assert "REQ_CULTURAL" in unallocated_ids, "Cultural Event must yield to the Exam when space is constrained"
