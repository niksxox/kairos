"""
Kairos - Multi-Event Room Allocation Optimizer
Formulates and solves global space allocation using Google OR-Tools CP-SAT.
Includes deterministic Greedy First-Feasible baseline for rigorous comparative evaluation.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from src.availability import AvailabilityEngine, IntervalMerger
from src.priority import infer_priority_tier, PRIORITY_WEIGHTS, PriorityTier


class MultiEventOptimizer:
    """
    Mathematical optimization solver for concurrent campus room allocation.
    Formulation:
      Variables:
        x[i, j] in {0, 1} for request i in I, room j in J
      Hard Constraints:
        1. At most one room per request: sum_j x[i, j] <= 1
        2. No room overlap: For any two requests i1, i2 overlapping in time,
           x[i1, j] + x[i2, j] <= 1 for all shared candidate rooms j
        3. Pre-pruned candidate pairs:
           - capacity[j] >= required_capacity[i]
           - facilities matched (projector, computers, accessibility)
           - room j is free from existing classes and events during [start, end)
      Objective (Lexicographic):
        Maximize sum_{i, j} [ BIG_M * x[i, j] - (capacity[j] - required_capacity[i]) * x[i, j] + bonus[i, j] ]
    """

    BIG_M = 100000  # Ensures priority: serving a request strictly dominates saving excess seats

    def __init__(
        self,
        rooms_df: pd.DataFrame,
        class_schedule_df: pd.DataFrame,
        events_df: pd.DataFrame,
        buffer_minutes: int = 0
    ):
        self.rooms_df = rooms_df
        self.sched_df = class_schedule_df
        self.events_df = events_df
        self.avail = AvailabilityEngine(class_schedule_df, events_df, rooms_df, buffer_minutes=buffer_minutes)

    def _build_candidate_pairs(
        self,
        requests_df: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], int]]:
        """
        Filters hard constraints to produce valid candidate (request_id, room_id) pairs
        and precomputes seat surplus penalties.
        """
        candidates: Dict[str, List[str]] = {}
        excess_caps: Dict[Tuple[str, str], int] = {}

        for _, req in requests_df.iterrows():
            q_id = str(req["request_id"])
            q_date = str(req["request_date"])
            q_start = float(req["start_hour"])
            q_dur = float(req["duration_hours"])
            q_cap = int(req["required_capacity"])
            q_proj = bool(req.get("projector_required", 0))
            q_comp = bool(req.get("computers_required", 0))
            q_acc = bool(req.get("accessible_required", 0))

            valid_rooms_for_req: List[str] = []

            for _, rm in self.rooms_df.iterrows():
                r_id = str(rm["room_id"])
                r_cap = int(rm["capacity"])
                
                # Constraint 1: Capacity
                if r_cap < q_cap:
                    continue

                # Constraint 2: Projector
                if q_proj and not bool(rm["projector"]):
                    continue

                # Constraint 3: Computers
                if q_comp and not bool(rm["computers"]):
                    continue

                # Constraint 4: Accessibility
                if q_acc and not bool(rm["accessible"]):
                    continue

                # Constraint 5: Scheduled availability against class schedule & events
                is_free, _ = self.avail.is_room_available(r_id, q_date, q_start, q_dur)
                if not is_free:
                    continue

                valid_rooms_for_req.append(r_id)
                excess_caps[(q_id, r_id)] = r_cap - q_cap

            candidates[q_id] = valid_rooms_for_req

        return candidates, excess_caps

    def optimize_allocations(
        self,
        requests_df: pd.DataFrame,
        time_limit_seconds: float = 15.0
    ) -> Dict[str, Any]:
        """
        Executes Google OR-Tools CP-SAT multi-request room allocation.
        """
        start_time = time.time()
        
        if not ORTOOLS_AVAILABLE:
            # Fallback to deterministic local search baseline if OR-Tools is unavailable
            return self.run_greedy_baseline(requests_df, fallback_reason="OR-Tools package not found")

        if requests_df.empty or self.rooms_df.empty:
            return {
                "solver_status": "EMPTY_INPUT",
                "runtime_seconds": 0.0,
                "total_requests": len(requests_df),
                "allocated_count": 0,
                "unallocated_count": len(requests_df),
                "fulfillment_rate": 0.0,
                "allocations": [],
                "unallocated": [],
                "total_excess_capacity": 0,
                "avg_excess_capacity": 0.0,
                "hard_constraint_violations": 0
            }

        # Step 1: Pre-prune candidate pairs
        candidates, excess_caps = self._build_candidate_pairs(requests_df)

        # Build CP-SAT Model
        model = cp_model.CpModel()
        x: Dict[Tuple[str, str], cp_model.IntVar] = {}

        for q_id, room_list in candidates.items():
            for r_id in room_list:
                x[(q_id, r_id)] = model.NewBoolVar(f"x_{q_id}_{r_id}")

        # Constraint 1: At most one room per request
        for q_id, room_list in candidates.items():
            if room_list:
                model.Add(sum(x[(q_id, r_id)] for r_id in room_list) <= 1)

        # Constraint 2: Mutual exclusion for conflicting requests in the same room
        # Group requests by date to find pairwise temporal overlaps
        req_lookup = requests_df.set_index("request_id").to_dict(orient="index")
        req_ids = list(candidates.keys())
        
        # Build pairwise conflicts
        conflicts: List[Tuple[str, str]] = []
        for i in range(len(req_ids)):
            q1 = req_ids[i]
            r1 = req_lookup[q1]
            s1, dur1, d1 = float(r1["start_hour"]), float(r1["duration_hours"]), str(r1["request_date"])
            e1 = s1 + dur1

            for j in range(i + 1, len(req_ids)):
                q2 = req_ids[j]
                r2 = req_lookup[q2]
                s2, dur2, d2 = float(r2["start_hour"]), float(r2["duration_hours"]), str(r2["request_date"])
                e2 = s2 + dur2

                # If same date and half-open intervals overlap
                if d1 == d2 and IntervalMerger.intervals_overlap((s1, e1), (s2, e2), buffer_hours=self.avail.buffer_hours):
                    conflicts.append((q1, q2))

        # Enforce no overlapping requests in the same room
        for q1, q2 in conflicts:
            shared_rooms = set(candidates.get(q1, [])) & set(candidates.get(q2, []))
            for r_id in shared_rooms:
                model.Add(x[(q1, r_id)] + x[(q2, r_id)] <= 1)

        # Objective Function (Priority-Weighted Lexicographic Objective)
        # Maximize: sum (BIG_M * priority_weight * x - excess_capacity * x)
        # Formal exams/tests (weight 100) are strictly prioritized over cultural events (weight 5)
        objective_terms = []
        for (q_id, r_id), var in x.items():
            wasted_seats = excess_caps[(q_id, r_id)]
            q_purpose = req_lookup[q_id].get("purpose", "")
            q_tier = infer_priority_tier(q_purpose)
            p_weight = PRIORITY_WEIGHTS[q_tier]
            coeff = int(self.BIG_M * p_weight - wasted_seats)
            objective_terms.append(coeff * var)

        if objective_terms:
            model.Maximize(sum(objective_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
        solver.parameters.num_search_workers = 4  # Multithreaded CP-SAT
        status = solver.Solve(model)

        runtime = round(time.time() - start_time, 4)
        status_name = solver.StatusName(status)

        # Process Results
        allocations = []
        allocated_req_ids = set()
        total_excess = 0

        rooms_dict = self.rooms_df.set_index("room_id").to_dict(orient="index")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for (q_id, r_id), var in x.items():
                if solver.Value(var) == 1:
                    q_data = req_lookup[q_id]
                    r_data = rooms_dict[r_id]
                    req_cap = int(q_data["required_capacity"])
                    room_cap = int(r_data["capacity"])
                    surplus = room_cap - req_cap
                    total_excess += surplus
                    allocated_req_ids.add(q_id)
                    q_tier = infer_priority_tier(q_data.get("purpose", ""))

                    allocations.append({
                        "request_id": q_id,
                        "room_id": r_id,
                        "request_date": q_data["request_date"],
                        "start_hour": q_data["start_hour"],
                        "duration_hours": q_data["duration_hours"],
                        "end_hour": q_data["start_hour"] + q_data["duration_hours"],
                        "purpose": q_data.get("purpose", "N/A"),
                        "priority_tier": q_tier.value,
                        "priority_score": PRIORITY_WEIGHTS[q_tier],
                        "required_capacity": req_cap,
                        "allocated_room_capacity": room_cap,
                        "excess_capacity": surplus,
                        "building": r_data["building"],
                        "room_type": r_data["room_type"],
                        "utilization_rate": round((req_cap / room_cap) * 100, 1),
                        "method": "OR-Tools CP-SAT"
                    })

        # Identify unallocated requests with exact reasons
        unallocated = []
        for _, req in requests_df.iterrows():
            q_id = str(req["request_id"])
            if q_id not in allocated_req_ids:
                cand_rooms = candidates.get(q_id, [])
                if not cand_rooms:
                    reason = "No room met capacity/facilities or all candidate rooms occupied by existing classes/events"
                else:
                    reason = f"Candidate rooms ({len(cand_rooms)} options) assigned to higher-priority requests during conflict"
                
                unallocated.append({
                    "request_id": q_id,
                    "request_date": req["request_date"],
                    "start_hour": req["start_hour"],
                    "duration_hours": req["duration_hours"],
                    "required_capacity": req["required_capacity"],
                    "purpose": req.get("purpose", "N/A"),
                    "unallocated_reason": reason,
                    "candidate_rooms_count": len(cand_rooms)
                })

        alloc_count = len(allocations)
        total_reqs = len(requests_df)
        fulfillment = round((alloc_count / total_reqs * 100), 2) if total_reqs > 0 else 0.0
        avg_excess = round(total_excess / alloc_count, 1) if alloc_count > 0 else 0.0

        return {
            "solver_status": status_name,
            "runtime_seconds": runtime,
            "total_requests": total_reqs,
            "allocated_count": alloc_count,
            "unallocated_count": len(unallocated),
            "fulfillment_rate": fulfillment,
            "allocations": allocations,
            "unallocated": unallocated,
            "total_excess_capacity": total_excess,
            "avg_excess_capacity": avg_excess,
            "hard_constraint_violations": 0  # Formally guaranteed by CP-SAT formulation
        }

    def run_greedy_baseline(
        self,
        requests_df: pd.DataFrame,
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deterministic First-Feasible-Room Greedy Baseline.
        Orders requests chronologically, then assigns the first available feasible room
        with capacity >= required, respecting all hard constraints.
        """
        start_time = time.time()
        if requests_df.empty or self.rooms_df.empty:
            return {
                "solver_status": "EMPTY_INPUT",
                "runtime_seconds": 0.0,
                "total_requests": len(requests_df),
                "allocated_count": 0,
                "unallocated_count": len(requests_df),
                "fulfillment_rate": 0.0,
                "allocations": [],
                "unallocated": [],
                "total_excess_capacity": 0,
                "avg_excess_capacity": 0.0,
                "hard_constraint_violations": 0
            }

        # Sort requests by priority weight descending, then chronologically
        sorted_reqs = requests_df.copy()
        purposes = sorted_reqs["purpose"] if "purpose" in sorted_reqs.columns else pd.Series(["Academic"] * len(sorted_reqs), index=sorted_reqs.index)
        sorted_reqs["priority_tier"] = purposes.apply(lambda p: infer_priority_tier(str(p)).value)
        sorted_reqs["priority_weight"] = purposes.apply(lambda p: PRIORITY_WEIGHTS[infer_priority_tier(str(p))])
        sorted_reqs = sorted_reqs.sort_values(by=["priority_weight", "request_date", "start_hour", "request_id"], ascending=[False, True, True, True])
        
        # Track dynamically assigned intervals per room and date: (start, end)
        dynamic_assignments: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

        allocations = []
        unallocated = []
        total_excess = 0

        rooms_dict = self.rooms_df.set_index("room_id").to_dict(orient="index")
        # Candidate room ordering for baseline: lowest capacity room first (greedy fit)
        sorted_room_ids = self.rooms_df.sort_values(by=["capacity", "room_id"])["room_id"].tolist()

        for _, req in sorted_reqs.iterrows():
            q_id = str(req["request_id"])
            q_date = str(req["request_date"])
            q_start = float(req["start_hour"])
            q_dur = float(req["duration_hours"])
            q_end = q_start + q_dur
            q_cap = int(req["required_capacity"])
            q_proj = bool(req.get("projector_required", 0))
            q_comp = bool(req.get("computers_required", 0))
            q_acc = bool(req.get("accessible_required", 0))

            assigned_room_id = None

            for r_id in sorted_room_ids:
                rm = rooms_dict[r_id]
                r_cap = int(rm["capacity"])

                if r_cap < q_cap:
                    continue
                if q_proj and not bool(rm["projector"]):
                    continue
                if q_comp and not bool(rm["computers"]):
                    continue
                if q_acc and not bool(rm["accessible"]):
                    continue

                # Check fixed class schedule and events
                is_free, _ = self.avail.is_room_available(r_id, q_date, q_start, q_dur)
                if not is_free:
                    continue

                # Check previously assigned dynamic requests in this greedy pass
                already_taken = False
                for prev_s, prev_e in dynamic_assignments.get((r_id, q_date), []):
                    if IntervalMerger.intervals_overlap((q_start, q_end), (prev_s, prev_e), buffer_hours=self.avail.buffer_hours):
                        already_taken = True
                        break

                if not already_taken:
                    assigned_room_id = r_id
                    break

            if assigned_room_id:
                # Record assignment
                r_data = rooms_dict[assigned_room_id]
                room_cap = int(r_data["capacity"])
                surplus = room_cap - q_cap
                total_excess += surplus

                if (assigned_room_id, q_date) not in dynamic_assignments:
                    dynamic_assignments[(assigned_room_id, q_date)] = []
                dynamic_assignments[(assigned_room_id, q_date)].append((q_start, q_end))

                allocations.append({
                    "request_id": q_id,
                    "room_id": assigned_room_id,
                    "request_date": q_date,
                    "start_hour": q_start,
                    "duration_hours": q_dur,
                    "end_hour": q_end,
                    "purpose": req.get("purpose", "N/A"),
                    "priority_tier": infer_priority_tier(req.get("purpose", "")).value,
                    "required_capacity": q_cap,
                    "allocated_room_capacity": room_cap,
                    "excess_capacity": surplus,
                    "building": r_data["building"],
                    "room_type": r_data["room_type"],
                    "utilization_rate": round((q_cap / room_cap) * 100, 1),
                    "method": "Greedy First-Feasible Baseline"
                })
            else:
                unallocated.append({
                    "request_id": q_id,
                    "request_date": q_date,
                    "start_hour": q_start,
                    "duration_hours": q_dur,
                    "required_capacity": q_cap,
                    "purpose": req.get("purpose", "N/A"),
                    "unallocated_reason": "No feasible room found or all viable rooms taken by prior requests in chronological order",
                    "candidate_rooms_count": 0
                })

        runtime = round(time.time() - start_time, 4)
        alloc_count = len(allocations)
        total_reqs = len(requests_df)
        fulfillment = round((alloc_count / total_reqs * 100), 2) if total_reqs > 0 else 0.0
        avg_excess = round(total_excess / alloc_count, 1) if alloc_count > 0 else 0.0

        return {
            "solver_status": "GREEDY_COMPLETED" if not fallback_reason else f"FALLBACK ({fallback_reason})",
            "runtime_seconds": runtime,
            "total_requests": total_reqs,
            "allocated_count": alloc_count,
            "unallocated_count": len(unallocated),
            "fulfillment_rate": fulfillment,
            "allocations": allocations,
            "unallocated": unallocated,
            "total_excess_capacity": total_excess,
            "avg_excess_capacity": avg_excess,
            "hard_constraint_violations": 0
        }

    def compare_optimizer_vs_baseline(
        self,
        requests_df: pd.DataFrame,
        time_limit_seconds: float = 15.0
    ) -> Dict[str, Any]:
        """
        Conducts a rigorous benchmark comparison on identical requests
        between OR-Tools CP-SAT and the Greedy First-Feasible baseline.
        """
        opt_res = self.optimize_allocations(requests_df, time_limit_seconds=time_limit_seconds)
        base_res = self.run_greedy_baseline(requests_df)

        metrics_comparison = {
            "requests_evaluated": len(requests_df),
            "optimizer": {
                "method": "Google OR-Tools CP-SAT",
                "assigned_count": opt_res["allocated_count"],
                "fulfillment_rate": opt_res["fulfillment_rate"],
                "total_excess_seats": opt_res["total_excess_capacity"],
                "avg_excess_seats": opt_res["avg_excess_capacity"],
                "runtime_seconds": opt_res["runtime_seconds"],
                "hard_constraint_violations": opt_res["hard_constraint_violations"]
            },
            "baseline": {
                "method": "Greedy First-Feasible",
                "assigned_count": base_res["allocated_count"],
                "fulfillment_rate": base_res["fulfillment_rate"],
                "total_excess_seats": base_res["total_excess_capacity"],
                "avg_excess_seats": base_res["avg_excess_capacity"],
                "runtime_seconds": base_res["runtime_seconds"],
                "hard_constraint_violations": base_res["hard_constraint_violations"]
            },
            "deltas": {
                "additional_requests_served": opt_res["allocated_count"] - base_res["allocated_count"],
                "fulfillment_gain_pct": round(opt_res["fulfillment_rate"] - base_res["fulfillment_rate"], 2),
                "excess_seats_saved": base_res["total_excess_capacity"] - opt_res["total_excess_capacity"],
            }
        }

        return {
            "metrics": metrics_comparison,
            "optimizer_result": opt_res,
            "baseline_result": base_res
        }
