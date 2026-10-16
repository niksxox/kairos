"""
Kairos - Metrics & Key Performance Indicators Module
Aggregates executive KPIs for dashboard cards and summary reports.
"""

from typing import Dict, Any
import pandas as pd


class MetricsCalculator:
    """
    Computes high-level decision support metrics for campus operations.
    """

    @staticmethod
    def compute_summary_kpis(
        rooms_df: pd.DataFrame,
        class_schedule_df: pd.DataFrame,
        events_df: pd.DataFrame,
        requests_df: pd.DataFrame
    ) -> Dict[str, Any]:
        total_rooms = len(rooms_df)
        total_requests = len(requests_df)
        total_classes = len(class_schedule_df)
        total_events = len(events_df)

        total_seats = int(rooms_df["capacity"].sum()) if not rooms_df.empty and "capacity" in rooms_df.columns else 0
        avg_room_capacity = round(rooms_df["capacity"].mean(), 1) if not rooms_df.empty and "capacity" in rooms_df.columns else 0

        # Number of requests with at least one room having sufficient capacity
        max_room_cap = rooms_df["capacity"].max() if not rooms_df.empty and "capacity" in rooms_df.columns else 0
        if not requests_df.empty and "required_capacity" in requests_df.columns:
            feasible_cap_requests = int((requests_df["required_capacity"] <= max_room_cap).sum())
        else:
            feasible_cap_requests = 0

        return {
            "total_rooms": total_rooms,
            "total_seats": total_seats,
            "avg_room_capacity": avg_room_capacity,
            "total_incoming_requests": total_requests,
            "total_classes_scheduled": total_classes,
            "total_events_scheduled": total_events,
            "theoretically_feasible_requests": feasible_cap_requests,
            "infeasible_by_capacity_alone": total_requests - feasible_cap_requests
        }
