"""
Kairos - Utilization & Demand vs Supply Analytics Engine
Calculates non-overlapping scheduled utilization metrics, hourly/weekday distributions,
and compares student demand distributions against physical space supply.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date

from config.settings import (
    CAMPUS_DAY_START_HOUR,
    CAMPUS_DAY_END_HOUR,
    OPERATING_HOURS_PER_DAY,
    VALID_WEEKDAYS
)
from src.availability import AvailabilityEngine, IntervalMerger


class UtilizationEngine:
    """
    Computes rigorous room and campus utilization statistics.
    All metrics represent SCHEDULED bookings, not physical sensor occupancy.
    """

    def __init__(
        self,
        rooms_df: pd.DataFrame,
        class_schedule_df: pd.DataFrame,
        events_df: pd.DataFrame,
        day_start: int = CAMPUS_DAY_START_HOUR,
        day_end: int = CAMPUS_DAY_END_HOUR
    ):
        self.rooms_df = rooms_df
        self.sched_df = class_schedule_df
        self.events_df = events_df
        self.day_start = day_start
        self.day_end = day_end
        self.operating_hours = day_end - day_start

    def compute_weekly_schedule_utilization(self) -> pd.DataFrame:
        """
        Computes weekly scheduled utilization for each room based on recurring class_schedule.csv.
        Weekly operating denominator = 6 weekdays (Mon-Sat) * operating_hours (14h) = 84 hours.
        Overlapping classes in the same room on the same weekday are merged to avoid double-counting.
        """
        if self.rooms_df.empty:
            return pd.DataFrame()

        total_weekly_hours = len(VALID_WEEKDAYS) * self.operating_hours
        room_stats = []

        for _, rm in self.rooms_df.iterrows():
            r_id = rm["room_id"]
            
            # Find all classes for this room
            total_merged_hours = 0.0
            day_hours_map = {w: 0.0 for w in VALID_WEEKDAYS}

            if not self.sched_df.empty:
                r_sched = self.sched_df[self.sched_df["room_id"] == r_id]
                for w in VALID_WEEKDAYS:
                    w_classes = r_sched[r_sched["weekday"] == w]
                    intervals = []
                    for _, row in w_classes.iterrows():
                        s = float(row["start_hour"])
                        e = s + float(row["duration_hours"])
                        # Clip to operating hours
                        s_clipped = max(self.day_start, s)
                        e_clipped = min(self.day_end, e)
                        if e_clipped > s_clipped:
                            intervals.append((s_clipped, e_clipped))

                    merged = IntervalMerger.merge_intervals(intervals)
                    day_h = sum(end - start for start, end in merged)
                    day_hours_map[w] = day_h
                    total_merged_hours += day_h

            util_rate = round((total_merged_hours / total_weekly_hours) * 100, 1)

            room_stats.append({
                "room_id": r_id,
                "building": rm["building"],
                "floor": rm["floor"],
                "room_type": rm["room_type"],
                "capacity": rm["capacity"],
                "weekly_scheduled_hours": round(total_merged_hours, 1),
                "total_operating_hours": total_weekly_hours,
                "scheduled_utilization_rate": util_rate,
                **{f"hours_{w.lower()}": round(day_hours_map[w], 1) for w in VALID_WEEKDAYS}
            })

        return pd.DataFrame(room_stats)

    def compute_hourly_heatmap_data(self) -> pd.DataFrame:
        """
        Builds a grid of scheduled room occupancies across Weekdays (Mon-Sat)
        and Operating Hours (08:00 - 21:00).
        """
        hours = list(range(self.day_start, self.day_end))
        grid = []

        total_rooms = len(self.rooms_df) if not self.rooms_df.empty else 1

        for w in VALID_WEEKDAYS:
            for h in hours:
                # Count how many unique rooms are occupied at hour h on weekday w
                occupied_rooms = set()
                if not self.sched_df.empty:
                    # An interval [start, start + duration) covers hour h if start <= h < start + duration
                    covering = self.sched_df[
                        (self.sched_df["weekday"] == w) &
                        (self.sched_df["start_hour"] <= h) &
                        (self.sched_df["start_hour"] + self.sched_df["duration_hours"] > h)
                    ]
                    occupied_rooms = set(covering["room_id"].unique())

                count_occupied = len(occupied_rooms)
                pct_occupied = round((count_occupied / total_rooms) * 100, 1)

                grid.append({
                    "weekday": w,
                    "hour": h,
                    "time_slot": f"{h:02d}:00 - {h+1:02d}:00",
                    "rooms_occupied": count_occupied,
                    "total_rooms": total_rooms,
                    "occupancy_rate_pct": pct_occupied
                })

        return pd.DataFrame(grid)

    def compute_demand_vs_supply(self, requests_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compares capacity bands of requested demand vs physical room inventory.
        """
        if self.rooms_df.empty or requests_df.empty:
            return {"bands": [], "data": pd.DataFrame()}

        bins = [0, 30, 60, 100, 250]
        labels = ["Small (<=30)", "Medium (31-60)", "Large (61-100)", "Auditorium/Lecture (101+)"]

        rooms_df = self.rooms_df.copy()
        rooms_df["band"] = pd.cut(rooms_df["capacity"], bins=bins, labels=labels, right=True)
        supply_counts = rooms_df["band"].value_counts().to_dict()

        reqs_df = requests_df.copy()
        reqs_df["band"] = pd.cut(reqs_df["required_capacity"], bins=bins, labels=labels, right=True)
        demand_counts = reqs_df["band"].value_counts().to_dict()

        comparison = []
        for band in labels:
            sup = supply_counts.get(band, 0)
            dem = demand_counts.get(band, 0)
            comparison.append({
                "capacity_band": band,
                "room_supply_count": sup,
                "request_demand_count": dem,
                "demand_to_supply_ratio": round(dem / max(1, sup), 2),
                "bottleneck_risk": "High" if (dem / max(1, sup)) > 50 else ("Moderate" if (dem / max(1, sup)) > 25 else "Low")
            })

        return {
            "bands": labels,
            "data": pd.DataFrame(comparison)
        }
