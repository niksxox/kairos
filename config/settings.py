"""
Kairos - CampusSpace Intelligence
Configuration and Settings Module
Team: Laphing Buddha | DATATHON 2026
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "admin_decisions.db"

# Operating Hours & Campus Schedule Constraints
CAMPUS_DAY_START_HOUR = 8      # 08:00
CAMPUS_DAY_END_HOUR = 22       # 22:00
OPERATING_HOURS_PER_DAY = CAMPUS_DAY_END_HOUR - CAMPUS_DAY_START_HOUR  # 14 hours

DEFAULT_BUFFER_MINUTES = 0    # Default 0 buffer for half-open intervals [start, end)
SETUP_BUFFER_MINUTES = 10     # Configurable setup/cleanup buffer option

# Room Types & Buildings Supported in DATATHON 2026 Schema
VALID_BUILDINGS = ["Central", "West", "North", "East", "South"]
VALID_ROOM_TYPES = ["Classroom", "Lab", "Seminar", "Study room", "Auditorium"]
VALID_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAY_ORDER = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

# Department List from Class Schedule
DEPARTMENTS = ["EEE", "ECE", "CSE", "Management", "Mechanical", "Sciences"]

# Request Purpose List
REQUEST_PURPOSES = ["Study group", "Class", "Workshop", "Club meeting", "Project work"]

# Matching & Optimization Scoring Weights
# Multi-objective weights for room ranking
DEFAULT_RANKING_WEIGHTS = {
    "capacity_fit": 0.50,      # Minimize wasted excess capacity
    "power_outlets": 0.20,     # Bonus for ample charging ports
    "room_type_match": 0.20,   # Exact room type preference
    "accessibility_bonus": 0.10 # Accessible rooms preferred when needed
}

# Column Mappings (Defensive against schema variations)
COLUMN_MAPPINGS = {
    "rooms": {
        "room_id": "room_id",
        "building": "building",
        "floor": "floor",
        "room_type": "room_type",
        "capacity": "capacity",
        "projector": "projector",
        "computers": "computers",
        "accessible": "accessible",
        "power_outlets": "power_outlets",
    },
    "class_schedule": {
        "schedule_id": "schedule_id",
        "room_id": "room_id",
        "course_code": "course_code",
        "weekday": "weekday",
        "start_hour": "start_hour",
        "duration_hours": "duration_hours",
        "expected_attendance": "expected_attendance",
        "department": "department",
    },
    "events": {
        "event_id": "event_id",
        "room_id": "room_id",
        "event_type": "event_type",
        "start_time": "start_time",
        "duration_hours": "duration_hours",
        "expected_attendance": "expected_attendance",
        "actual_attendance": "actual_attendance",
    },
    "room_requests": {
        "request_id": "request_id",
        "request_date": "request_date",
        "start_hour": "start_hour",
        "duration_hours": "duration_hours",
        "required_capacity": "required_capacity",
        "projector_required": "projector_required",
        "computers_required": "computers_required",
        "purpose": "purpose",
        "status": "status",
    }
}
