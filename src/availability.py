"""
Kairos - Availability Estimation Engine
Calculates scheduled room conflicts, merges overlapping intervals without double-counting,
and evaluates half-open interval feasibility [start, end).
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, date

from config.settings import (
    CAMPUS_DAY_START_HOUR,
    CAMPUS_DAY_END_HOUR,
    DEFAULT_BUFFER_MINUTES
)


class IntervalMerger:
    """
    Utility for merging overlapping or contiguous time intervals.
    Operates on half-open intervals [start, end).
    """

    @staticmethod
    def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Takes a list of [start, end) intervals, sorts them, and merges overlapping
        or contiguous intervals.
        Example: [(9.0, 11.0), (10.0, 12.0)] -> [(9.0, 12.0)]
        """
        if not intervals:
            return []

        # Sort primarily by start time, secondarily by end time
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged: List[Tuple[float, float]] = []

        curr_start, curr_end = sorted_intervals[0]

        for s, e in sorted_intervals[1:]:
            if s <= curr_end:  # Overlap or boundary touch
                curr_end = max(curr_end, e)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = s, e

        merged.append((curr_start, curr_end))
        return merged

    @staticmethod
    def intervals_overlap(
        int1: Tuple[float, float],
        int2: Tuple[float, float],
        buffer_hours: float = 0.0
    ) -> bool:
        """
        Checks whether two half-open intervals [s1, e1) and [s2, e2) overlap.
        With half-open semantics: [9, 11) and [11, 13) DO NOT overlap if buffer is 0.
        If buffer_hours > 0, interval 1 is expanded by buffer on both sides.
        """
        s1, e1 = int1[0], int1[1] + buffer_hours
        s2, e2 = int2[0], int2[1] + buffer_hours
        # Overlap occurs if max(start) < min(end)
        return max(int1[0], int2[0]) < min(e1, e2)


class AvailabilityEngine:
    """
    Evaluates room availability by combining recurring weekly class schedules
    and specific calendar-dated events.
    """

    def __init__(
        self,
        class_schedule_df: pd.DataFrame,
        events_df: pd.DataFrame,
        rooms_df: Optional[pd.DataFrame] = None,
        buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
        db_instance=None
    ):
        self.sched_df = class_schedule_df
        self.events_df = events_df
        self.rooms_df = rooms_df
        self.buffer_hours = buffer_minutes / 60.0
        self.db = db_instance

    def get_room_intervals_for_date(
        self,
        room_id: str,
        query_date_str: str
    ) -> List[Tuple[float, float]]:
        """
        Extracts and merges all scheduled class and event intervals for a specific room and date.
        Combines:
          1. Class schedules matching the weekday of query_date_str
          2. Events matching query_date_str
        Returns merged list of (start_hour, end_hour) tuples.
        """
        raw_intervals: List[Tuple[float, float]] = []
        q_date = pd.to_datetime(query_date_str).date()
        q_weekday = q_date.strftime("%a")

        # 1. Class schedule (recurring by weekday)
        if not self.sched_df.empty:
            m_sched = self.sched_df[
                (self.sched_df["room_id"] == room_id) &
                (self.sched_df["weekday"] == q_weekday)
            ]
            for _, row in m_sched.iterrows():
                s = float(row["start_hour"])
                e = s + float(row["duration_hours"])
                raw_intervals.append((s, e))

        # 2. Events (by calendar date)
        if not self.events_df.empty:
            if "event_date" in self.events_df.columns:
                m_events = self.events_df[
                    (self.events_df["room_id"] == room_id) &
                    (self.events_df["event_date"] == query_date_str)
                ]
            else:
                ev_dates = pd.to_datetime(self.events_df["start_time"]).dt.date
                m_events = self.events_df[
                    (self.events_df["room_id"] == room_id) &
                    (ev_dates == q_date)
                ]

            for _, row in m_events.iterrows():
                if "start_hour" in row and "duration_hours" in row:
                    s = float(row["start_hour"])
                    dur = float(row["duration_hours"])
                else:
                    dt = pd.to_datetime(row["start_time"])
                    s = dt.hour + dt.minute / 60.0
                    dur = float(row["duration_hours"])
                e = s + dur
                raw_intervals.append((s, e))

        # 3. Dynamic Confirmed Student Bookings in SQLite database
        if self.db is not None:
            try:
                db_bookings = self.db.get_confirmed_bookings_for_date(query_date_str)
                for b in db_bookings:
                    if b.get("room_id") == room_id:
                        s = float(b["start_hour"])
                        e = s + float(b["duration_hours"])
                        raw_intervals.append((s, e))
            except Exception:
                pass

        return IntervalMerger.merge_intervals(raw_intervals)

    def is_room_available(
        self,
        room_id: str,
        query_date_str: str,
        start_hour: float,
        duration_hours: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Determines whether a room is available for [start_hour, start_hour + duration_hours).
        Returns (is_available, conflict_reason_if_any).
        """
        req_start = float(start_hour)
        req_end = req_start + float(duration_hours)
        req_interval = (req_start, req_end)

        occupied_intervals = self.get_room_intervals_for_date(room_id, query_date_str)

        for occ_s, occ_e in occupied_intervals:
            occ_interval = (occ_s, occ_e)
            if IntervalMerger.intervals_overlap(req_interval, occ_interval, self.buffer_hours):
                conflict_desc = f"Occupied by scheduled activity from {int(occ_s):02d}:{int((occ_s%1)*60):02d} to {int(occ_e):02d}:{int((occ_e%1)*60):02d}"
                return False, conflict_desc

        return True, None

    def get_free_intervals(
        self,
        room_id: str,
        query_date_str: str,
        day_start: float = CAMPUS_DAY_START_HOUR,
        day_end: float = CAMPUS_DAY_END_HOUR
    ) -> List[Tuple[float, float]]:
        """
        Computes free time slots for a room on a given date between day_start and day_end.
        """
        occupied = self.get_room_intervals_for_date(room_id, query_date_str)
        free: List[Tuple[float, float]] = []

        curr = day_start
        for occ_s, occ_e in occupied:
            # Clip occupied to operating hours
            occ_s_clipped = max(day_start, occ_s)
            occ_e_clipped = min(day_end, occ_e)
            
            if occ_s_clipped > curr:
                free.append((curr, occ_s_clipped))
            curr = max(curr, occ_e_clipped)

        if curr < day_end:
            free.append((curr, day_end))

        return free

    def calculate_room_scheduled_hours(
        self,
        room_id: str,
        query_date_str: str,
        day_start: float = CAMPUS_DAY_START_HOUR,
        day_end: float = CAMPUS_DAY_END_HOUR
    ) -> float:
        """
        Computes total non-overlapping occupied hours for a room on a given date.
        Clips to operating window to prevent extrapolation error.
        """
        occupied = self.get_room_intervals_for_date(room_id, query_date_str)
        total_hours = 0.0
        for occ_s, occ_e in occupied:
            clip_s = max(day_start, occ_s)
            clip_e = min(day_end, occ_e)
            if clip_e > clip_s:
                total_hours += (clip_e - clip_s)
        return total_hours
