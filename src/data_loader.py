"""
Kairos - Data Ingestion Module
Handles loading, caching, and source tracking for campus space datasets.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

from config.settings import (
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
    COLUMN_MAPPINGS
)


class DataSourceType:
    OFFICIAL = "Official Synthetic Data (DATATHON 2026)"
    SAMPLE = "Sample Deterministic Subset"
    UPLOADED = "User Uploaded CSVs"
    GENERATED = "Synthetic Demo Generator"


class DataLoader:
    """
    Ingests and maps educational CSV datasets.
    Maintains provenance and prevents mutation of raw data.
    """

    REQUIRED_FILES = {
        "rooms": "rooms.csv",
        "class_schedule": "class_schedule.csv",
        "events": "events.csv",
        "room_requests": "room_requests.csv",
    }

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.source_type = DataSourceType.OFFICIAL

    def load_dataset(
        self,
        name: str,
        custom_file_or_buffer=None,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Loads a single dataset by name or from a buffer/file object.
        Applies defensive column mapping.
        """
        mapping = column_mapping or COLUMN_MAPPINGS.get(name, {})
        
        try:
            if custom_file_or_buffer is not None:
                df = pd.read_csv(custom_file_or_buffer)
            else:
                filepath = self.data_dir / self.REQUIRED_FILES[name]
                if not filepath.exists():
                    raise FileNotFoundError(f"Dataset file '{self.REQUIRED_FILES[name]}' not found in {self.data_dir}")
                df = pd.read_csv(filepath)

            # Apply mapping if needed (case-insensitive strip matching)
            rename_dict = {}
            for target_col, source_col in mapping.items():
                if source_col in df.columns:
                    rename_dict[source_col] = target_col
                else:
                    # Case-insensitive search
                    lower_cols = {c.lower().strip(): c for c in df.columns}
                    if source_col.lower().strip() in lower_cols:
                        rename_dict[lower_cols[source_col.lower().strip()]] = target_col
            
            if rename_dict:
                df = df.rename(columns=rename_dict)

            return df

        except Exception as e:
            # Return empty dataframe with expected columns rather than crashing
            expected_cols = list(mapping.keys())
            print(f"[DataLoader Warning] Failed to load {name}: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=expected_cols)

    def load_all(
        self,
        custom_buffers: Optional[Dict[str, object]] = None,
        use_sample: bool = False
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
        """
        Loads all four datasets: rooms, class_schedule, events, room_requests.
        Returns a dict of DataFrames and a string indicating the active source.
        """
        datasets = {}
        if custom_buffers:
            self.source_type = DataSourceType.UPLOADED
            for key in self.REQUIRED_FILES:
                buf = custom_buffers.get(key)
                datasets[key] = self.load_dataset(key, custom_file_or_buffer=buf)
        elif use_sample or not (self.data_dir / "rooms.csv").exists():
            self.data_dir = SAMPLE_DATA_DIR
            self.source_type = DataSourceType.SAMPLE
            for key in self.REQUIRED_FILES:
                datasets[key] = self.load_dataset(key)
        else:
            self.source_type = DataSourceType.OFFICIAL
            for key in self.REQUIRED_FILES:
                datasets[key] = self.load_dataset(key)

        return datasets, self.source_type


def generate_demo_datasets(seed: int = 42) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Generates deterministic, internally consistent synthetic records
    with known seed if datasets are missing or when user requests demo mode.
    """
    np.random.seed(seed)
    
    # 1. Rooms
    num_rooms = 15
    room_ids = [f"R{i+1:03d}" for i in range(num_rooms)]
    buildings = ["Central", "North", "West", "East", "South"]
    room_types = ["Classroom", "Lab", "Seminar", "Study room", "Auditorium"]
    
    rooms_data = {
        "room_id": room_ids,
        "building": [buildings[i % len(buildings)] for i in range(num_rooms)],
        "floor": [i % 4 for i in range(num_rooms)],
        "room_type": [room_types[i % len(room_types)] for i in range(num_rooms)],
        "capacity": [20 + (i * 10) for i in range(num_rooms)],
        "projector": [1 if i % 2 == 0 else 0 for i in range(num_rooms)],
        "computers": [1 if i % 3 == 0 else 0 for i in range(num_rooms)],
        "accessible": [1 if i % 4 != 0 else 0 for i in range(num_rooms)],
        "power_outlets": [8 + i for i in range(num_rooms)],
    }
    rooms_df = pd.DataFrame(rooms_data)

    # 2. Class schedule
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    departments = ["CSE", "ECE", "EEE", "Management", "Mechanical"]
    sched_rows = []
    for sid in range(1, 41):
        r_id = room_ids[sid % num_rooms]
        w = weekdays[sid % len(weekdays)]
        start_h = 9 + ((sid * 2) % 8)
        dur = 1 if sid % 2 == 0 else 2
        sched_rows.append({
            "schedule_id": f"S{sid:05d}",
            "room_id": r_id,
            "course_code": f"CS{sid:03d}",
            "weekday": w,
            "start_hour": start_h,
            "duration_hours": dur,
            "expected_attendance": 20 + (sid % 40),
            "department": departments[sid % len(departments)]
        })
    sched_df = pd.DataFrame(sched_rows)

    # 3. Events
    event_types = ["Guest lecture", "Seminar", "Club meeting", "Exam", "Workshop"]
    event_rows = []
    for eid in range(1, 26):
        r_id = room_ids[(eid * 3) % num_rooms]
        day = 10 + (eid % 15)
        event_rows.append({
            "event_id": f"EV{eid:05d}",
            "room_id": r_id,
            "event_type": event_types[eid % len(event_types)],
            "start_time": f"2025-09-{day:02d} 14:00:00",
            "duration_hours": 2,
            "expected_attendance": 35 + eid,
            "actual_attendance": 30 + eid
        })
    events_df = pd.DataFrame(event_rows)

    # 4. Room requests
    purposes = ["Study group", "Class", "Workshop", "Club meeting", "Project work"]
    statuses = ["Approved", "Waitlisted", "Rejected", "Cancelled"]
    req_rows = []
    for qid in range(1, 31):
        day = 10 + (qid % 15)
        req_rows.append({
            "request_id": f"Q{qid:05d}",
            "request_date": f"2025-09-{day:02d}",
            "start_hour": 10 + (qid % 8),
            "duration_hours": 1 if qid % 2 == 0 else 2,
            "required_capacity": 15 + (qid * 3),
            "projector_required": 1 if qid % 3 == 0 else 0,
            "computers_required": 1 if qid % 5 == 0 else 0,
            "purpose": purposes[qid % len(purposes)],
            "status": statuses[qid % len(statuses)]
        })
    reqs_df = pd.DataFrame(req_rows)

    return {
        "rooms": rooms_df,
        "class_schedule": sched_df,
        "events": events_df,
        "room_requests": reqs_df,
    }, DataSourceType.GENERATED
