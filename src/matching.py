"""
Kairos - Room Matching & Recommendation Engine
Applies multi-stage constraint filtering, transparent weighted ranking,
and comprehensive natural-language explanations for room assignment.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import DEFAULT_RANKING_WEIGHTS
from src.availability import AvailabilityEngine


class RoomMatchingEngine:
    """
    Evaluates individual room requests against campus inventory.
    Implements a strict 3-step decision architecture:
      Step 1: Hard constraint pruning (Capacity, Features, Accessibility, Availability)
      Step 2: Transparent heuristic ranking (Capacity fit, Amenity bonus, Room type match)
      Step 3: Fully explainable decision breakdown (Trade-offs & rejection forensics)
    """

    def __init__(
        self,
        rooms_df: pd.DataFrame,
        availability_engine: AvailabilityEngine,
        weights: Optional[Dict[str, float]] = None
    ):
        self.rooms_df = rooms_df
        self.avail = availability_engine
        self.weights = weights or DEFAULT_RANKING_WEIGHTS

    def evaluate_request(
        self,
        request_date: str,
        start_hour: float,
        duration_hours: float,
        required_capacity: int,
        projector_required: bool = False,
        computers_required: bool = False,
        accessible_required: bool = False,
        preferred_room_type: Optional[str] = None,
        preferred_building: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates all rooms in inventory for a single room request.
        Returns ranked feasible recommendations and detailed rejection logs.
        """
        feasible_rooms: List[Dict[str, Any]] = []
        rejected_rooms: List[Dict[str, Any]] = []

        # Validate inputs
        if required_capacity <= 0:
            return {
                "success": False,
                "error": "Required attendance must be greater than 0.",
                "recommendations": [],
                "rejected": []
            }
        
        if duration_hours <= 0:
            return {
                "success": False,
                "error": "Duration must be positive.",
                "recommendations": [],
                "rejected": []
            }

        if start_hour < 0 or (start_hour + duration_hours) > 24:
            return {
                "success": False,
                "error": f"Requested interval [{start_hour}, {start_hour + duration_hours}) exceeds 24-hour day boundary.",
                "recommendations": [],
                "rejected": []
            }

        max_capacity_available = self.rooms_df["capacity"].max() if not self.rooms_df.empty else 0
        if required_capacity > max_capacity_available:
            return {
                "success": False,
                "error": f"Requested attendance ({required_capacity}) exceeds largest available room on campus ({max_capacity_available} seats).",
                "recommendations": [],
                "rejected": []
            }

        # Step 1: Filter rooms by hard constraints
        for _, room in self.rooms_df.iterrows():
            r_id = room["room_id"]
            r_cap = int(room["capacity"])
            r_proj = bool(room["projector"])
            r_comp = bool(room["computers"])
            r_acc = bool(room["accessible"])
            r_type = str(room["room_type"])
            r_bldg = str(room["building"])
            r_outlets = int(room.get("power_outlets", 0))

            rejection_reasons = []

            # Hard Constraint A: Capacity
            if r_cap < required_capacity:
                rejection_reasons.append(f"Insufficient capacity ({r_cap} seats vs {required_capacity} required)")

            # Hard Constraint B: Projector
            if projector_required and not r_proj:
                rejection_reasons.append("Missing required projector")

            # Hard Constraint C: Computers
            if computers_required and not r_comp:
                rejection_reasons.append("Missing required computers/lab hardware")

            # Hard Constraint D: Accessibility
            if accessible_required and not r_acc:
                rejection_reasons.append("Room is not certified accessible")

            # Hard Constraint E: Scheduled Availability
            is_avail, conflict_reason = self.avail.is_room_available(
                r_id, request_date, start_hour, duration_hours
            )
            if not is_avail:
                rejection_reasons.append(conflict_reason)

            # Record outcome
            room_info = {
                "room_id": r_id,
                "building": r_bldg,
                "floor": int(room["floor"]),
                "room_type": r_type,
                "capacity": r_cap,
                "projector": r_proj,
                "computers": r_comp,
                "accessible": r_acc,
                "power_outlets": r_outlets
            }

            if rejection_reasons:
                rejected_rooms.append({
                    **room_info,
                    "rejection_reasons": rejection_reasons
                })
            else:
                # Room passed all hard constraints! Calculate Step 2 Ranking Score
                score, breakdown = self._calculate_ranking_score(
                    room_info=room_info,
                    required_capacity=required_capacity,
                    preferred_room_type=preferred_room_type,
                    preferred_building=preferred_building
                )

                # Generate Step 3 Explanations & Trade-offs
                pros, cons = self._generate_explanations(
                    room_info=room_info,
                    required_capacity=required_capacity,
                    preferred_room_type=preferred_room_type,
                    preferred_building=preferred_building
                )

                feasible_rooms.append({
                    **room_info,
                    "match_score": round(score, 3),
                    "score_breakdown": breakdown,
                    "pros": pros,
                    "trade_offs": cons,
                    "excess_capacity": r_cap - required_capacity,
                    "utilization_efficiency": round((required_capacity / r_cap) * 100, 1)
                })

        # Sort feasible rooms descending by score
        ranked_rooms = sorted(feasible_rooms, key=lambda x: x["match_score"], reverse=True)

        return {
            "success": True,
            "total_rooms_evaluated": len(self.rooms_df),
            "feasible_count": len(ranked_rooms),
            "rejected_count": len(rejected_rooms),
            "recommendations": ranked_rooms,
            "rejected": rejected_rooms
        }

    def _calculate_ranking_score(
        self,
        room_info: Dict[str, Any],
        required_capacity: int,
        preferred_room_type: Optional[str] = None,
        preferred_building: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes an explainable, deterministic suitability score in [0.0, 1.0].
        Formula:
          score = w_cap * cap_fit + w_outlets * outlet_score + w_type * type_fit + w_bldg * bldg_fit
        """
        r_cap = room_info["capacity"]
        r_outlets = room_info["power_outlets"]
        r_type = room_info["room_type"]
        r_bldg = room_info["building"]

        # Capacity fit: penalize excess unused seats (ideal is 1.0 when cap == required)
        # Cap_fit in (0, 1]
        cap_fit = max(0.0, 1.0 - ((r_cap - required_capacity) / float(r_cap)))

        # Power outlets score: normalized to 30 max outlets
        outlet_score = min(1.0, r_outlets / 25.0)

        # Room type match
        type_fit = 1.0
        if preferred_room_type:
            type_fit = 1.0 if r_type.lower() == preferred_room_type.lower() else 0.4

        # Building match
        bldg_fit = 1.0
        if preferred_building:
            bldg_fit = 1.0 if r_bldg.lower() == preferred_building.lower() else 0.5

        w_cap = self.weights.get("capacity_fit", 0.50)
        w_outlets = self.weights.get("power_outlets", 0.20)
        w_type = self.weights.get("room_type_match", 0.20)
        w_bldg = 0.10

        total_weight = w_cap + w_outlets + w_type + w_bldg
        composite_score = (
            w_cap * cap_fit +
            w_outlets * outlet_score +
            w_type * type_fit +
            w_bldg * bldg_fit
        ) / total_weight

        breakdown = {
            "capacity_fit": round(cap_fit, 3),
            "power_outlets_score": round(outlet_score, 3),
            "room_type_score": round(type_fit, 3),
            "location_score": round(bldg_fit, 3)
        }

        return composite_score, breakdown

    def _generate_explanations(
        self,
        room_info: Dict[str, Any],
        required_capacity: int,
        preferred_room_type: Optional[str] = None,
        preferred_building: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        pros: List[str] = []
        cons: List[str] = []

        r_cap = room_info["capacity"]
        excess = r_cap - required_capacity

        # Capacity commentary
        if excess == 0:
            pros.append(f"Perfect capacity match ({r_cap} seats, 0 seats wasted)")
        elif excess <= 10:
            pros.append(f"Tight capacity fit: only {excess} surplus seats ({round((required_capacity/r_cap)*100)}% utilization)")
        elif excess > 50:
            cons.append(f"Substantial excess capacity: {excess} unused seats ({r_cap} capacity for {required_capacity} attendees)")

        # Features
        if room_info["projector"]:
            pros.append("Equipped with presentation projector")
        if room_info["computers"]:
            pros.append("Equipped with dedicated workstation computers")
        if room_info["accessible"]:
            pros.append("Certified universally accessible (ADA compliant)")
        if room_info["power_outlets"] >= 20:
            pros.append(f"High power outlet availability ({room_info['power_outlets']} charging ports)")

        # Preference checks
        if preferred_room_type:
            if room_info["room_type"].lower() == preferred_room_type.lower():
                pros.append(f"Matches preferred room type ({room_info['room_type']})")
            else:
                cons.append(f"Configured as {room_info['room_type']} rather than requested {preferred_room_type}")

        if preferred_building:
            if room_info["building"].lower() == preferred_building.lower():
                pros.append(f"Located in preferred building ({room_info['building']})")
            else:
                cons.append(f"Located in {room_info['building']} Building instead of {preferred_building}")

        return pros, cons
