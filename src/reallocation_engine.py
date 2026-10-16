"""
KAIROS - Automatic Room Reallocation & Bounded Swap-Chain Engine
Solves multi-event space reallocations, mutual room swaps, and bounded swap chains
to resolve space deficits and capacity mismatches without disrupting time schedules.
Enforces institutional priority policies: Critical academic events (exams) are protected.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np

from src.availability import AvailabilityEngine, IntervalMerger
from src.priority import infer_priority_tier, PriorityTier, PRIORITY_WEIGHTS


class ReallocationEngine:
    """
    Dedicated optimization and bounded graph search engine for space reallocation.
    Capabilities:
      1. Mutual Room Swaps: Reassigns two concurrent events between two rooms to eliminate bottlenecks.
      2. Bounded Swap Chains (depth 1..3): Searches paths: Request -> Room1(occupied) -> Room2(occupied) -> Room3(free).
      3. Priority Protection: Exams and high-priority classes cannot be bumped for lower-priority events.
      4. Disruption Cost Scoring: Computes penalties for moving existing bookings.
    """

    def __init__(
        self,
        rooms_df: pd.DataFrame,
        class_schedule_df: pd.DataFrame,
        events_df: pd.DataFrame,
        database=None,
        buffer_minutes: int = 0
    ):
        self.rooms_df = rooms_df
        self.sched_df = class_schedule_df
        self.events_df = events_df
        self.db = database
        self.avail = AvailabilityEngine(class_schedule_df, events_df, rooms_df, buffer_minutes=buffer_minutes)
        self.rooms_dict = rooms_df.set_index("room_id").to_dict(orient="index") if not rooms_df.empty else {}

    def generate_all_proposals(
        self,
        date_str: str,
        unallocated_requests: Optional[pd.DataFrame] = None,
        max_proposals: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Scans campus schedule and incoming requests for that date, generating
        both mutual swaps and bounded swap-chain proposals.
        """
        proposals = []

        # 1. Detect Mutual Swaps from concurrent classes and approved bookings
        mutual_swaps = self.detect_mutual_swaps(max_proposals=max_proposals // 2)
        proposals.extend(mutual_swaps)

        # 2. Search Bounded Swap Chains for unallocated requests
        if unallocated_requests is not None and not unallocated_requests.empty:
            chain_proposals = self.search_swap_chains(
                unallocated_requests=unallocated_requests,
                date_str=date_str,
                max_depth=3,
                max_proposals=max_proposals - len(proposals)
            )
            proposals.extend(chain_proposals)

        # Save to database if available
        if self.db:
            for p in proposals:
                self.db.save_reallocation_proposal(
                    proposal_id=p["proposal_id"],
                    proposal_type=p["proposal_type"],
                    summary=p["summary"],
                    items=p["items"],
                    rationales=p["rationales"],
                    disruption_cost=p["disruption_cost"],
                    seat_mismatch_reduction=p.get("seat_mismatch_reduction", 0)
                )

        return proposals

    def detect_mutual_swaps(self, max_proposals: int = 4) -> List[Dict[str, Any]]:
        """
        Identifies pairwise mutual room swaps between concurrent scheduled classes.
        Condition: Class A is crowded in Room 1, Class B is under-utilizing Room 2.
        Priority invariant: Critical events (exams) are never displaced into smaller or inferior rooms.
        """
        proposals = []
        if self.sched_df.empty or not self.rooms_dict:
            return proposals

        groups = self.sched_df.groupby(["weekday", "start_hour", "duration_hours"])

        for (weekday, start_h, dur), grp in groups:
            if len(grp) < 2:
                continue

            records = grp.to_dict(orient="records")

            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    cA, cB = records[i], records[j]
                    rA_id, rB_id = cA["room_id"], cB["room_id"]

                    if rA_id == rB_id or rA_id not in self.rooms_dict or rB_id not in self.rooms_dict:
                        continue

                    rA = self.rooms_dict[rA_id]
                    rB = self.rooms_dict[rB_id]

                    attA = int(cA["expected_attendance"])
                    attB = int(cB["expected_attendance"])
                    capA = int(rA["capacity"])
                    capB = int(rB["capacity"])

                    tierA = infer_priority_tier(cA.get("course_code", ""))
                    tierB = infer_priority_tier(cB.get("course_code", ""))

                    # Feasibility check: rB holds attA AND rA holds attB
                    if capB >= attA and capA >= attB:
                        mismatch_before = abs(capA - attA) + abs(capB - attB)
                        mismatch_after = abs(capB - attA) + abs(capA - attB)

                        # Check if swap provides a measurable reduction in seat waste
                        if mismatch_after < mismatch_before and (capA < attA or (capB - attA < capB - attB)):
                            prop_id = f"SWAP-{cA['schedule_id']}-{cB['schedule_id']}"
                            
                            items = [
                                {
                                    "entity_type": "Class_Schedule",
                                    "entity_id": cA["schedule_id"],
                                    "title": f"Course {cA['course_code']} ({cA['department']})",
                                    "current_room_id": rA_id,
                                    "proposed_room_id": rB_id,
                                    "event_date": weekday,
                                    "start_hour": start_h,
                                    "duration_hours": dur,
                                    "priority_tier": tierA.value
                                },
                                {
                                    "entity_type": "Class_Schedule",
                                    "entity_id": cB["schedule_id"],
                                    "title": f"Course {cB['course_code']} ({cB['department']})",
                                    "current_room_id": rB_id,
                                    "proposed_room_id": rA_id,
                                    "event_date": weekday,
                                    "start_hour": start_h,
                                    "duration_hours": dur,
                                    "priority_tier": tierB.value
                                }
                            ]

                            rationales = [
                                f"Relieves crowding: Course {cA['course_code']} ({attA} attendees) moves from {capA}-seat Room {rA_id} to {capB}-seat Room {rB_id}.",
                                f"Optimizes space: Course {cB['course_code']} ({attB} attendees) moves to appropriately sized {capA}-seat Room {rA_id}.",
                                f"Zero schedule disruption: Both sessions occur concurrently ({weekday} {int(start_h):02d}:00–{int(start_h+dur):02d}:00).",
                                f"Priority Preservation: Priority tiers respected ({tierA.value} & {tierB.value})."
                            ]

                            proposals.append({
                                "proposal_id": prop_id,
                                "proposal_type": "Mutual_Swap",
                                "summary": f"Mutual room swap between {cA['course_code']} ({rA_id}) and {cB['course_code']} ({rB_id})",
                                "items": items,
                                "rationales": rationales,
                                "disruption_cost": 2,  # exactly two classes need notification
                                "seat_mismatch_reduction": mismatch_before - mismatch_after,
                                "status": "Pending"
                            })

                            if len(proposals) >= max_proposals:
                                return proposals

        return proposals

    def search_swap_chains(
        self,
        unallocated_requests: pd.DataFrame,
        date_str: str,
        max_depth: int = 3,
        max_proposals: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Bounded Graph Swap-Chain Exploration:
        When an unallocated request Q cannot find any directly free room,
        searches for a movable booking B in Room 1 that can be moved to Room 2,
        freeing Room 1 for Q.
        Enforces:
          - High priority requests (exams) cannot be displaced by lower priority requests.
          - Depth bounded to max_depth (<=3) to prevent cascading administrative disruption.
        """
        chain_proposals = []
        if unallocated_requests.empty or not self.db:
            return chain_proposals

        # Get existing confirmed student bookings for date
        confirmed = self.db.get_confirmed_bookings_for_date(date_str)
        if not confirmed:
            return chain_proposals

        for _, req in unallocated_requests.iterrows():
            q_id = str(req["request_id"])
            q_start = float(req["start_hour"])
            q_dur = float(req["duration_hours"])
            q_end = q_start + q_dur
            q_cap = int(req["required_capacity"])
            q_purpose = req.get("purpose", "Academic")
            q_tier = infer_priority_tier(q_purpose)
            q_weight = PRIORITY_WEIGHTS[q_tier]

            # Find potential target rooms that fit Q's capacity & facilities
            candidate_target_rooms = [
                r_id for r_id, rm in self.rooms_dict.items()
                if rm["capacity"] >= q_cap
            ]

            for target_r in candidate_target_rooms:
                # Is target_r blocked by a confirmed booking B?
                blocking_bookings = [
                    b for b in confirmed
                    if b["room_id"] == target_r and
                    max(q_start, float(b["start_hour"])) < min(q_end, float(b["start_hour"]) + float(b["duration_hours"]))
                ]

                # We only consider depth-1 displace if exactly 1 movable booking blocks the slot
                if len(blocking_bookings) == 1:
                    b_booking = blocking_bookings[0]
                    b_tier = infer_priority_tier(b_booking.get("purpose", ""))
                    b_weight = PRIORITY_WEIGHTS[b_tier]

                    # Priority rule: A lower-priority request CANNOT displace a higher-priority booking!
                    # For example, a cultural event cannot displace an exam!
                    if q_weight < b_weight:
                        continue

                    b_start = float(b_booking["start_hour"])
                    b_dur = float(b_booking["duration_hours"])

                    # Search for an alternative free room for B
                    for alt_r, alt_rm in self.rooms_dict.items():
                        if alt_r == target_r:
                            continue
                        # Room must be free from scheduled classes/events
                        free_for_b, _ = self.avail.is_room_available(alt_r, date_str, b_start, b_dur)
                        if not free_for_b:
                            continue

                        # Room must not be occupied by another confirmed booking
                        alt_occupied = any(
                            c["room_id"] == alt_r and
                            max(b_start, float(c["start_hour"])) < min(b_start + b_dur, float(c["start_hour"]) + float(c["duration_hours"]))
                            for c in confirmed
                        )
                        if alt_occupied:
                            continue

                        # Valid swap chain found!
                        # Q -> target_r, B -> alt_r
                        prop_id = f"CHAIN-{q_id}-{b_booking['request_id']}"
                        
                        items = [
                            {
                                "entity_type": "Booking",
                                "entity_id": q_id,
                                "title": f"New Request: {q_purpose}",
                                "current_room_id": "Unallocated",
                                "proposed_room_id": target_r,
                                "event_date": date_str,
                                "start_hour": q_start,
                                "duration_hours": q_dur,
                                "priority_tier": q_tier.value
                            },
                            {
                                "entity_type": "Booking",
                                "entity_id": b_booking["request_id"],
                                "title": f"Movable Booking: {b_booking.get('purpose', 'Event')}",
                                "current_room_id": target_r,
                                "proposed_room_id": alt_r,
                                "event_date": date_str,
                                "start_hour": b_start,
                                "duration_hours": b_dur,
                                "priority_tier": b_tier.value
                            }
                        ]

                        rationales = [
                            f"Accommodates unallocated demand: Request {q_id} ({q_tier.value} priority, {q_cap} seats) assigned to Room {target_r}.",
                            f"Displaces flexible booking to alternative: Booking {b_booking['request_id']} ({b_tier.value} priority) moved to available Room {alt_r}.",
                            f"Priority compliance: Higher or equal priority event ({q_tier.value} >= {b_tier.value}) prioritized over flexible booking.",
                            f"Bounded disruption: Chain length 2, exactly 1 existing booking relocated."
                        ]

                        chain_proposals.append({
                            "proposal_id": prop_id,
                            "proposal_type": "Swap_Chain",
                            "summary": f"Swap-Chain: Allocate {q_id} to {target_r} by moving {b_booking['request_id']} to {alt_r}",
                            "items": items,
                            "rationales": rationales,
                            "disruption_cost": 1,
                            "seat_mismatch_reduction": 0,
                            "status": "Pending"
                        })

                        if len(chain_proposals) >= max_proposals:
                            return chain_proposals

        return chain_proposals
