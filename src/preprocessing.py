"""
Kairos - Preprocessing & Normalization Module
Standardizes datatypes, parses datetime intervals, and generates derived temporal features.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class Preprocessor:
    """
    Cleans and standardizes educational datasets for matching and optimization.
    Preserves original columns while creating standardized derived fields.
    """

    @staticmethod
    def process_rooms(rooms_df: pd.DataFrame) -> pd.DataFrame:
        df = rooms_df.copy()
        if df.empty:
            return df
        
        # Standardize strings
        df["room_id"] = df["room_id"].astype(str).str.strip().str.upper()
        df["building"] = df["building"].astype(str).str.strip()
        df["room_type"] = df["room_type"].astype(str).str.strip()
        
        # Ensure numeric fields
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(0).astype(int)
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0).astype(int)
        df["projector"] = pd.to_numeric(df["projector"], errors="coerce").fillna(0).astype(int)
        df["computers"] = pd.to_numeric(df["computers"], errors="coerce").fillna(0).astype(int)
        df["accessible"] = pd.to_numeric(df["accessible"], errors="coerce").fillna(0).astype(int)
        df["power_outlets"] = pd.to_numeric(df["power_outlets"], errors="coerce").fillna(0).astype(int)

        # Capacity band for aggregation
        bins = [0, 30, 60, 100, 200]
        labels = ["Small (<=30)", "Medium (31-60)", "Large (61-100)", "Auditorium/Lecture (101+)"]
        df["capacity_band"] = pd.cut(df["capacity"], bins=bins, labels=labels, right=True)

        return df

    @staticmethod
    def process_schedule(sched_df: pd.DataFrame) -> pd.DataFrame:
        df = sched_df.copy()
        if df.empty:
            return df
        
        df["schedule_id"] = df["schedule_id"].astype(str).str.strip().str.upper()
        df["room_id"] = df["room_id"].astype(str).str.strip().str.upper()
        df["course_code"] = df["course_code"].astype(str).str.strip()
        df["weekday"] = df["weekday"].astype(str).str.strip().str.capitalize()
        df["department"] = df["department"].astype(str).str.strip()
        
        df["start_hour"] = pd.to_numeric(df["start_hour"], errors="coerce").fillna(8).astype(int)
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(1).astype(int)
        df["end_hour"] = df["start_hour"] + df["duration_hours"]
        df["expected_attendance"] = pd.to_numeric(df["expected_attendance"], errors="coerce").fillna(0).astype(int)

        return df

    @staticmethod
    def process_events(events_df: pd.DataFrame) -> pd.DataFrame:
        df = events_df.copy()
        if df.empty:
            return df
        
        df["event_id"] = df["event_id"].astype(str).str.strip().str.upper()
        df["room_id"] = df["room_id"].astype(str).str.strip().str.upper()
        df["event_type"] = df["event_type"].astype(str).str.strip()
        
        # Datetime parsing
        df["start_dt"] = pd.to_datetime(events_df["start_time"], errors="coerce")
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(1).astype(int)
        df["end_dt"] = df["start_dt"] + pd.to_timedelta(df["duration_hours"], unit="h")
        
        # Derived fields
        df["event_date"] = df["start_dt"].dt.strftime("%Y-%m-%d")
        df["weekday"] = df["start_dt"].dt.strftime("%a")
        df["start_hour"] = df["start_dt"].dt.hour
        df["end_hour"] = df["start_hour"] + df["duration_hours"]
        df["expected_attendance"] = pd.to_numeric(df["expected_attendance"], errors="coerce").fillna(0).astype(int)
        df["actual_attendance"] = pd.to_numeric(df["actual_attendance"], errors="coerce").fillna(0).astype(int)

        return df

    @staticmethod
    def process_requests(reqs_df: pd.DataFrame) -> pd.DataFrame:
        df = reqs_df.copy()
        if df.empty:
            return df
        
        df["request_id"] = df["request_id"].astype(str).str.strip().str.upper()
        df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["weekday"] = pd.to_datetime(df["request_date"], errors="coerce").dt.strftime("%a")
        
        df["start_hour"] = pd.to_numeric(df["start_hour"], errors="coerce").fillna(8).astype(int)
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(1).astype(int)
        df["end_hour"] = df["start_hour"] + df["duration_hours"]
        
        df["required_capacity"] = pd.to_numeric(df["required_capacity"], errors="coerce").fillna(0).astype(int)
        df["projector_required"] = pd.to_numeric(df["projector_required"], errors="coerce").fillna(0).astype(int)
        df["computers_required"] = pd.to_numeric(df["computers_required"], errors="coerce").fillna(0).astype(int)
        df["purpose"] = df["purpose"].astype(str).str.strip()
        df["status"] = df["status"].astype(str).str.strip()

        # Capacity band
        bins = [0, 30, 60, 100, 200]
        labels = ["Small (<=30)", "Medium (31-60)", "Large (61-100)", "Auditorium/Lecture (101+)"]
        df["capacity_band"] = pd.cut(df["required_capacity"], bins=bins, labels=labels, right=True)

        return df

    @classmethod
    def process_all(cls, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {
            "rooms": cls.process_rooms(datasets.get("rooms", pd.DataFrame())),
            "class_schedule": cls.process_schedule(datasets.get("class_schedule", pd.DataFrame())),
            "events": cls.process_events(datasets.get("events", pd.DataFrame())),
            "room_requests": cls.process_requests(datasets.get("room_requests", pd.DataFrame())),
        }

    @classmethod
    def preprocess_all(
        cls,
        rooms_df: pd.DataFrame,
        sched_df: pd.DataFrame,
        events_df: pd.DataFrame,
        reqs_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            cls.process_rooms(rooms_df),
            cls.process_schedule(sched_df),
            cls.process_events(events_df),
            cls.process_requests(reqs_df)
        )
