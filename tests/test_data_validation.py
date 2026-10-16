"""
Tests for Data Loading, Validation, and Schema Checks
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader, DataSourceType, generate_demo_datasets
from src.data_validation import DataValidator


@pytest.fixture
def sample_loader():
    return DataLoader()


def test_valid_dataset_loading(sample_loader):
    datasets, source = sample_loader.load_all(use_sample=True)
    assert "rooms" in datasets
    assert "class_schedule" in datasets
    assert "events" in datasets
    assert "room_requests" in datasets
    assert len(datasets["rooms"]) > 0
    assert len(datasets["class_schedule"]) > 0
    assert len(datasets["events"]) > 0
    assert len(datasets["room_requests"]) > 0


def test_missing_columns_and_malformed_input():
    # Construct a malformed dataframe missing expected columns
    bad_rooms = pd.DataFrame({
        "room_id": ["R001", "R002"],
        # Missing building, capacity, etc.
    })
    validator = DataValidator({"rooms": bad_rooms})
    report = validator.validate_all()
    audit = report["dataset_audits"]["rooms"]
    assert "building" in audit["missing_expected_columns"]
    assert "capacity" in audit["missing_expected_columns"]


def test_room_id_validation_and_orphan_detection():
    rooms = pd.DataFrame({"room_id": ["R001", "R002"], "capacity": [50, 60]})
    # Schedule references R999 which is not in rooms
    sched = pd.DataFrame({
        "schedule_id": ["S001"],
        "room_id": ["R999"],
        "start_hour": [9],
        "duration_hours": [2],
        "expected_attendance": [30]
    })
    validator = DataValidator({"rooms": rooms, "class_schedule": sched})
    report = validator.validate_all()
    ref = report["referential_integrity"]
    assert "R999" in ref["orphan_schedule_rooms"]
    assert ref["integrity_passed"] is False


def test_missing_optional_columns_and_empty_input():
    loader = DataLoader()
    # Loading non-existent file gracefully returns empty DataFrame
    df = loader.load_dataset("rooms", custom_file_or_buffer=pd.DataFrame())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_demo_generator_consistency():
    demo_data, source = generate_demo_datasets(seed=123)
    assert source == DataSourceType.GENERATED
    assert len(demo_data["rooms"]) == 15
    assert len(demo_data["class_schedule"]) == 40
    # Invariant: all schedule room_ids must exist in rooms
    known_rooms = set(demo_data["rooms"]["room_id"])
    for r in demo_data["class_schedule"]["room_id"]:
        assert r in known_rooms
