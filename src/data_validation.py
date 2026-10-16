"""
Kairos - Data Validation Module
Audits data quality, checks schema invariants, flags malformed records,
and produces explainable validation reports without silent drops.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config.settings import (
    VALID_BUILDINGS,
    VALID_ROOM_TYPES,
    VALID_WEEKDAYS,
    CAMPUS_DAY_START_HOUR,
    CAMPUS_DAY_END_HOUR
)


class DataValidator:
    """
    Validates schemas, referential integrity, and business logic constraints
    for educational campus datasets.
    """

    EXPECTED_COLUMNS = {
        "rooms": ["room_id", "building", "floor", "room_type", "capacity", "projector", "computers", "accessible", "power_outlets"],
        "class_schedule": ["schedule_id", "room_id", "course_code", "weekday", "start_hour", "duration_hours", "expected_attendance", "department"],
        "events": ["event_id", "room_id", "event_type", "start_time", "duration_hours", "expected_attendance", "actual_attendance"],
        "room_requests": ["request_id", "request_date", "start_hour", "duration_hours", "required_capacity", "projector_required", "computers_required", "purpose", "status"]
    }

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets
        self.report: Dict[str, Any] = {}
        self.clean_datasets: Dict[str, pd.DataFrame] = {}
        self.excluded_records: Dict[str, pd.DataFrame] = {}

    def validate_all(self) -> Dict[str, Any]:
        """
        Runs comprehensive validation across all 4 datasets.
        Returns a structured validation dictionary.
        """
        self.report = {
            "summary": {},
            "dataset_audits": {},
            "referential_integrity": {},
            "temporal_audits": {},
            "capacity_audits": {},
            "exclusion_summary": {}
        }

        # 1. Individual dataset audits
        for key, expected_cols in self.EXPECTED_COLUMNS.items():
            df = self.datasets.get(key, pd.DataFrame())
            self.report["dataset_audits"][key] = self._audit_dataset(key, df, expected_cols)

        # 2. Referential integrity (Room ID foreign keys in schedule and events)
        rooms_df = self.datasets.get("rooms", pd.DataFrame())
        known_rooms = set(rooms_df["room_id"].dropna().unique()) if "room_id" in rooms_df.columns else set()

        sched_df = self.datasets.get("class_schedule", pd.DataFrame())
        sched_rooms = set(sched_df["room_id"].dropna().unique()) if "room_id" in sched_df.columns else set()
        orphan_sched_rooms = sched_rooms - known_rooms

        events_df = self.datasets.get("events", pd.DataFrame())
        event_rooms = set(events_df["room_id"].dropna().unique()) if "room_id" in events_df.columns else set()
        orphan_event_rooms = event_rooms - known_rooms

        self.report["referential_integrity"] = {
            "total_rooms_cataloged": len(known_rooms),
            "orphan_schedule_rooms": list(orphan_sched_rooms),
            "orphan_event_rooms": list(orphan_event_rooms),
            "integrity_passed": len(orphan_sched_rooms) == 0 and len(orphan_event_rooms) == 0
        }

        # 3. Temporal consistency
        self.report["temporal_audits"] = self._audit_temporal_data()

        # 4. Capacity consistency
        self.report["capacity_audits"] = self._audit_capacity_data()

        # Overall summary metrics
        total_rows = sum(len(df) for df in self.datasets.values())
        total_nulls = sum(audit["null_values_count"] for audit in self.report["dataset_audits"].values())
        total_duplicates = sum(audit["duplicate_identifiers_count"] for audit in self.report["dataset_audits"].values())
        
        self.report["summary"] = {
            "total_records_ingested": total_rows,
            "total_missing_fields": total_nulls,
            "total_duplicate_ids": total_duplicates,
            "status": "VALID" if (total_duplicates == 0 and self.report["referential_integrity"]["integrity_passed"]) else "FLAGGED"
        }

        return self.report

    def _audit_dataset(self, name: str, df: pd.DataFrame, expected_cols: List[str]) -> Dict[str, Any]:
        audit: Dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_expected_columns": [c for c in expected_cols if c not in df.columns],
            "unexpected_columns": [c for c in df.columns if c not in expected_cols],
            "null_values_by_column": df.isnull().sum().to_dict() if not df.empty else {},
            "null_values_count": int(df.isnull().sum().sum()) if not df.empty else 0,
        }

        # Check primary key duplicates
        id_col = {
            "rooms": "room_id",
            "class_schedule": "schedule_id",
            "events": "event_id",
            "room_requests": "request_id"
        }.get(name)

        if id_col and id_col in df.columns:
            duplicates = df[df.duplicated(subset=[id_col], keep=False)]
            audit["duplicate_identifiers_count"] = len(duplicates)
            audit["duplicate_identifiers"] = duplicates[id_col].tolist() if len(duplicates) < 20 else duplicates[id_col].head(20).tolist()
        else:
            audit["duplicate_identifiers_count"] = 0
            audit["duplicate_identifiers"] = []

        return audit

    def _audit_temporal_data(self) -> Dict[str, Any]:
        temporal: Dict[str, Any] = {
            "invalid_schedule_hours": 0,
            "invalid_schedule_durations": 0,
            "invalid_event_timestamps": 0,
            "invalid_event_durations": 0,
            "invalid_request_hours": 0,
            "invalid_request_durations": 0,
            "notes": []
        }

        # Class schedule checks
        sched = self.datasets.get("class_schedule", pd.DataFrame())
        if not sched.empty and "start_hour" in sched.columns and "duration_hours" in sched.columns:
            invalid_h = sched[(sched["start_hour"] < 0) | (sched["start_hour"] > 23)]
            invalid_dur = sched[sched["duration_hours"] <= 0]
            temporal["invalid_schedule_hours"] = len(invalid_h)
            temporal["invalid_schedule_durations"] = len(invalid_dur)

        # Events checks
        events = self.datasets.get("events", pd.DataFrame())
        if not events.empty and "start_time" in events.columns and "duration_hours" in events.columns:
            try:
                dt_series = pd.to_datetime(events["start_time"], errors="coerce")
                invalid_ts = events[dt_series.isna()]
                temporal["invalid_event_timestamps"] = len(invalid_ts)
            except Exception:
                temporal["invalid_event_timestamps"] = len(events)
            
            invalid_dur = events[events["duration_hours"] <= 0]
            temporal["invalid_event_durations"] = len(invalid_dur)

        # Room requests checks
        reqs = self.datasets.get("room_requests", pd.DataFrame())
        if not reqs.empty and "start_hour" in reqs.columns and "duration_hours" in reqs.columns:
            invalid_h = reqs[(reqs["start_hour"] < 0) | (reqs["start_hour"] > 23)]
            invalid_dur = reqs[reqs["duration_hours"] <= 0]
            temporal["invalid_request_hours"] = len(invalid_h)
            temporal["invalid_request_durations"] = len(invalid_dur)

        return temporal

    def _audit_capacity_data(self) -> Dict[str, Any]:
        cap_audit = {
            "rooms_with_invalid_capacity": 0,
            "schedule_with_invalid_attendance": 0,
            "events_with_invalid_attendance": 0,
            "requests_with_invalid_capacity": 0,
            "requests_exceeding_max_campus_capacity": 0
        }

        rooms = self.datasets.get("rooms", pd.DataFrame())
        max_campus_capacity = rooms["capacity"].max() if not rooms.empty and "capacity" in rooms.columns else 0

        if not rooms.empty and "capacity" in rooms.columns:
            cap_audit["rooms_with_invalid_capacity"] = len(rooms[rooms["capacity"] <= 0])

        sched = self.datasets.get("class_schedule", pd.DataFrame())
        if not sched.empty and "expected_attendance" in sched.columns:
            cap_audit["schedule_with_invalid_attendance"] = len(sched[sched["expected_attendance"] < 0])

        events = self.datasets.get("events", pd.DataFrame())
        if not events.empty and "expected_attendance" in events.columns:
            cap_audit["events_with_invalid_attendance"] = len(events[events["expected_attendance"] < 0])

        reqs = self.datasets.get("room_requests", pd.DataFrame())
        if not reqs.empty and "required_capacity" in reqs.columns:
            cap_audit["requests_with_invalid_capacity"] = len(reqs[reqs["required_capacity"] <= 0])
            if max_campus_capacity > 0:
                cap_audit["requests_exceeding_max_campus_capacity"] = len(reqs[reqs["required_capacity"] > max_campus_capacity])

        return cap_audit
