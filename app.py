"""
KAIROS — Campus Space Management Platform
A restrained, production-oriented space allocation and scheduling system.
Team: Laphing Buddha | DATATHON 2026 | Track: Education
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    VALID_BUILDINGS,
    VALID_ROOM_TYPES,
    VALID_WEEKDAYS,
    CAMPUS_DAY_START_HOUR,
    CAMPUS_DAY_END_HOUR
)
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.availability import AvailabilityEngine
from src.matching import RoomMatchingEngine
from src.optimizer import MultiEventOptimizer, ORTOOLS_AVAILABLE
from src.utilization import UtilizationEngine
from src.reallocation_engine import ReallocationEngine
from src.ingestion import DataIngestionService
from src.priority import infer_priority_tier, get_priority_info, PriorityTier, PRIORITY_WEIGHTS
from src.database import CampusDatabase
from src.ml_model import DemandForecaster


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & MINIMAL ENTERPRISE STYLING (NO AI SLOPE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KAIROS | Space Management",
    layout="wide",
    initial_sidebar_state="expanded",
)

MINIMAL_CSS = """
<style>
    /* Clean, understated typography and layout */
    body, [class*="ViewExact"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #111827;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
    }

    /* Subdued page header */
    .app-header {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }

    /* Simple stat tile */
    .stat-tile {
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        background: #ffffff;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .stat-num {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 0.2rem;
    }
    .stat-caption {
        font-size: 0.75rem;
        color: #4b5563;
        margin-top: 0.15rem;
    }

    /* Minimalist status pills */
    .pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #d1d5db;
        background: #f9fafb;
        color: #374151;
    }
    .pill-critical {
        background: #fef2f2;
        color: #991b1b;
        border-color: #fecaca;
    }
    .pill-high {
        background: #fffbeb;
        color: #92400e;
        border-color: #fde68a;
    }
    .pill-approved {
        background: #f0fdf4;
        color: #166534;
        border-color: #bbf7d0;
    }
    .pill-pending {
        background: #fffbeb;
        color: #92400e;
        border-color: #fde68a;
    }
    .pill-rejected {
        background: #fef2f2;
        color: #991b1b;
        border-color: #fecaca;
    }
</style>
"""
st.markdown(MINIMAL_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA & SERVICES
# -----------------------------------------------------------------------------
db = CampusDatabase()

if "data_version" not in st.session_state:
    st.session_state["data_version"] = 1

@st.cache_data(show_spinner=False)
def load_app_data(version: int):
    loader = DataLoader()
    raw_dict, src_type = loader.load_all()
    clean_dict = Preprocessor.process_all(raw_dict)
    return (
        clean_dict["rooms"],
        clean_dict["class_schedule"],
        clean_dict["events"],
        clean_dict["room_requests"],
        src_type
    )

rooms_df, sched_df, events_df, reqs_df, source_label = load_app_data(st.session_state["data_version"])

avail_engine = AvailabilityEngine(sched_df, events_df, rooms_df, db_instance=db)
matching_engine = RoomMatchingEngine(rooms_df, avail_engine)
optimizer_engine = MultiEventOptimizer(rooms_df, sched_df, events_df)
realloc_engine = ReallocationEngine(rooms_df, sched_df, events_df, database=db)
ingest_service = DataIngestionService(database=db, existing_rooms=set(rooms_df["room_id"]))
util_engine = UtilizationEngine(rooms_df, sched_df, events_df)

model_path = ROOT_DIR / "data" / "demand_forecast_model.json"
ml_forecaster = DemandForecaster(model_path=str(model_path) if model_path.exists() else None)


# -----------------------------------------------------------------------------
# AUTH & SIDEBAR
# -----------------------------------------------------------------------------
if "current_user" not in st.session_state:
    st.session_state["current_user"] = {
        "username": "student",
        "role": "student",
        "department": "Computer Science"
    }

user = st.session_state["current_user"]

with st.sidebar:
    st.write("**KAIROS**")
    st.caption("Campus Space Management")
    st.markdown("---")

    st.write(f"User: **{user['username']}**")
    st.write(f"Role: `{user['role']}`")
    st.caption(f"Dept: {user['department']}")

    # Plain role switcher
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("Student View", use_container_width=True, disabled=(user["role"] == "student")):
            st.session_state["current_user"] = {"username": "student", "role": "student", "department": "Computer Science"}
            st.rerun()
    with col_r2:
        if st.button("Admin View", use_container_width=True, disabled=(user["role"] == "admin")):
            st.session_state["current_user"] = {"username": "admin", "role": "admin", "department": "Campus Operations"}
            st.rerun()

    st.markdown("---")

    if user["role"] == "admin":
        nav_items = [
            "Overview",
            "Find a Room",
            "My Bookings",
            "Schedule Calendar",
            "Multi-Event Optimizer",
            "Room Reallocation",
            "Demand Forecast",
            "Space Analytics",
            "Data & Ingestion API",
            "Admin Queue"
        ]
    else:
        nav_items = [
            "Overview",
            "Find a Room",
            "My Bookings",
            "Schedule Calendar",
            "Demand Forecast"
        ]

    page = st.radio("Navigation", nav_items, index=0)

    st.markdown("---")
    st.caption(f"Source: {source_label}")
    st.caption(f"Solver: {'OR-Tools CP-SAT' if ORTOOLS_AVAILABLE else 'Deterministic Heuristic'}")
    st.caption(f"ML: {'NumPy LinReg (Active)' if ml_forecaster.model else 'Untrained'}")

    if st.button("Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state["data_version"] += 1
        st.rerun()


def render_page_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="app-header">
        <h2 class="app-title">{title}</h2>
        <div class="app-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    render_page_header("Overview", "Campus room inventory and active operational state.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">Rooms</div>
            <div class="stat-num">{len(rooms_df)}</div>
            <div class="stat-caption">{rooms_df['building'].nunique()} buildings</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">Classes</div>
            <div class="stat-num">{len(sched_df):,}</div>
            <div class="stat-caption">Recurring weekly</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">Events</div>
            <div class="stat-num">{len(events_df):,}</div>
            <div class="stat-caption">Special events</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-tile">
            <div class="stat-label">Pending Demands</div>
            <div class="stat-num">{len(reqs_df):,}</div>
            <div class="stat-caption">Unallocated requests</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.write("**Operational Policies:**")
    st.write("1. **Priority Hierarchy:** Academic examinations and tests have highest allocation priority (weight 100), strictly outranking non-academic or cultural gatherings (weight 5).")
    st.write("2. **Conflict Prevention:** Half-open time intervals [start, end) are strictly checked to prevent double-booking.")
    st.write("3. **Capacity Reallocations:** Inefficiencies between small classes in auditoriums and large classes in small rooms are detected and can be reassigned with zero time shift.")


# -----------------------------------------------------------------------------
# 2. FIND A ROOM
# -----------------------------------------------------------------------------
elif page == "Find a Room":
    render_page_header("Find a Room", "Search available rooms by capacity, equipment, and date.")

    with st.form("search_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            f_date = st.date_input("Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
            f_start = st.selectbox("Start Time", list(range(8, 21)), index=6, format_func=lambda h: f"{h:02d}:00")
            f_dur = st.selectbox("Duration (Hours)", [1.0, 1.5, 2.0, 3.0, 4.0], index=2)
        with f2:
            f_att = st.number_input("Expected Attendance", min_value=1, max_value=300, value=35, step=5)
            f_category = st.selectbox(
                "Event Category",
                [
                    "Formal Examination / Test (Critical Priority)",
                    "Academic Lecture / Lab (High Priority)",
                    "Seminar / Colloquium (Medium Priority)",
                    "Study Group / Project (Standard Priority)",
                    "Cultural / Social Event (Flexible Priority)"
                ],
                index=0
            )
            f_purpose = st.text_input("Event Description", value="Midterm Exam Session")
        with f3:
            st.write("Requirements:")
            f_proj = st.checkbox("Projector Required", value=True)
            f_comp = st.checkbox("Computer Lab Required", value=False)
            f_acc = st.checkbox("Wheelchair Accessible", value=False)
            f_bldg = st.selectbox("Building", ["Any Building"] + sorted(VALID_BUILDINGS))

        search_btn = st.form_submit_button("Search Available Rooms", type="primary")

    if search_btn:
        full_purpose = f"{f_category.split('(')[0].strip()} - {f_purpose}"
        tier, p_score, _ = get_priority_info(full_purpose)

        st.session_state["search_params"] = {
            "date": f_date.strftime("%Y-%m-%d"),
            "start": float(f_start),
            "dur": float(f_dur),
            "cap": int(f_att),
            "proj": f_proj,
            "comp": f_comp,
            "acc": f_acc,
            "bldg": None if f_bldg == "Any Building" else f_bldg,
            "purpose": full_purpose,
            "tier": tier,
            "score": p_score
        }

    if "search_params" in st.session_state:
        sp = st.session_state["search_params"]
        tier_obj = sp["tier"]
        badge_class = "pill-critical" if tier_obj == PriorityTier.CRITICAL else ("pill-high" if tier_obj == PriorityTier.HIGH else "pill")

        st.markdown(f"""
        <div style="margin: 0.8rem 0; font-size: 0.85rem;">
            Priority Classification: <span class="pill {badge_class}">{tier_obj.value} (Weight: {sp['score']})</span>
            &nbsp;|&nbsp; Target: {sp['date']} ({int(sp['start']):02d}:00 to {int(sp['start']+sp['dur']):02d}:00)
        </div>
        """, unsafe_allow_html=True)

        recs, rejs = matching_engine.find_and_rank_rooms(
            request_date=sp["date"],
            start_hour=sp["start"],
            duration_hours=sp["dur"],
            required_capacity=sp["cap"],
            projector_required=sp["proj"],
            computers_required=sp["comp"],
            accessible_required=sp["acc"],
            preferred_building=sp["bldg"]
        )

        if not recs:
            st.warning("No rooms match all capacity, facility, and availability requirements.")
            if rejs:
                st.write(f"Disqualified rooms: {len(rejs)}")
                st.dataframe(pd.DataFrame([
                    {"Room": r["room_id"], "Building": r["building"], "Capacity": r["capacity"], "Reason": " | ".join(r["rejection_reasons"])}
                    for r in rejs[:10]
                ]), use_container_width=True, hide_index=True)
        else:
            st.write(f"**Feasible Rooms Found ({len(recs)}):**")
            rec_rows = []
            for r in recs[:8]:
                rec_rows.append({
                    "Room": r["room_id"],
                    "Building": r["building"],
                    "Capacity": r["capacity"],
                    "Fit Score": f"{r['match_score']*100:.1f}%",
                    "Projector": "Yes" if r["projector"] else "No",
                    "Computers": "Yes" if r["computers"] else "No",
                    "Surplus Seats": r["excess_capacity"]
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

            selected_room = st.selectbox("Select Room to Request", [r["room_id"] for r in recs[:8]])
            if st.button(f"Submit Request for {selected_room}"):
                req_id = db.create_booking_request(
                    username=user["username"],
                    room_id=selected_room,
                    request_date=sp["date"],
                    start_hour=sp["start"],
                    duration_hours=sp["dur"],
                    required_capacity=sp["cap"],
                    projector_required=sp["proj"],
                    computers_required=sp["comp"],
                    accessible_required=sp["acc"],
                    purpose=sp["purpose"],
                    auto_approve=False
                )
                st.success(f"Request recorded: {req_id} (Status: Pending Review)")
                st.rerun()


# -----------------------------------------------------------------------------
# 3. MY BOOKINGS
# -----------------------------------------------------------------------------
elif page == "My Bookings":
    render_page_header("My Bookings", f"Booking requests submitted by {user['username']}.")

    bookings = db.get_user_bookings(user["username"])

    if bookings.empty:
        st.write("No requests found. Use 'Find a Room' to search and submit reservations.")
    else:
        st.dataframe(bookings[["request_id", "status", "room_id", "request_date", "start_hour", "duration_hours", "required_capacity", "priority_tier", "purpose"]], use_container_width=True, hide_index=True)

        # Cancel action
        active_ids = bookings[bookings["status"].isin(["Pending", "Approved"])]["request_id"].tolist()
        if active_ids:
            st.write("**Cancel a Booking:**")
            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                target_cancel = st.selectbox("Select Request ID", active_ids)
            with col_c2:
                st.write("")
                if st.button("Cancel Request"):
                    ok, msg = db.cancel_booking(target_cancel, user["username"])
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


# -----------------------------------------------------------------------------
# 4. SCHEDULE CALENDAR
# -----------------------------------------------------------------------------
elif page == "Schedule Calendar":
    render_page_header("Schedule Calendar", "Room timetable across classes, events, and bookings.")

    c1, c2 = st.columns(2)
    with c1:
        sel_date = st.date_input("Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with c2:
        sel_room = st.selectbox("Room", sorted(rooms_df["room_id"].unique()))

    date_str = sel_date.strftime("%Y-%m-%d")
    wday_str = sel_date.strftime("%a")

    room_classes = sched_df[(sched_df["room_id"] == sel_room) & (sched_df["weekday"] == wday_str)]
    room_events = events_df[(events_df["room_id"] == sel_room) & (events_df["event_date"] == date_str)]
    room_bookings = [b for b in db.get_confirmed_bookings_for_date(date_str) if b.get("room_id") == sel_room]

    items = []
    for _, c in room_classes.iterrows():
        items.append({
            "Time": f"{int(c['start_hour']):02d}:00 - {int(c['start_hour']+c['duration_hours']):02d}:00",
            "Type": "Class Lecture",
            "Details": f"{c['course_code']} ({c['department']})",
            "Attendance": c["expected_attendance"]
        })
    for _, e in room_events.iterrows():
        items.append({
            "Time": f"{int(e['start_hour']):02d}:00 - {int(e['start_hour']+e['duration_hours']):02d}:00",
            "Type": "Special Event",
            "Details": e["event_type"],
            "Attendance": e["expected_attendance"]
        })
    for b in room_bookings:
        items.append({
            "Time": f"{int(b['start_hour']):02d}:00 - {int(b['start_hour']+b['duration_hours']):02d}:00",
            "Type": "Student Booking",
            "Details": b.get("purpose", "Reservation"),
            "Attendance": "Confirmed"
        })

    if not items:
        st.write(f"No commitments scheduled for {sel_room} on {date_str}. Room is completely free.")
    else:
        st.dataframe(pd.DataFrame(items).sort_values(by="Time"), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 5. MULTI-EVENT OPTIMIZER
# -----------------------------------------------------------------------------
elif page == "Multi-Event Optimizer":
    render_page_header("Multi-Event Room Allocation", "Global mathematical assignment using Google OR-Tools CP-SAT.")

    opt_c1, opt_c2 = st.columns(2)
    with opt_c1:
        o_date = st.date_input("Date to Allocate", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with opt_c2:
        o_timeout = st.slider("Solver Timeout (Seconds)", 3, 20, 8)

    o_date_str = o_date.strftime("%Y-%m-%d")
    batch_reqs = reqs_df[reqs_df["request_date"] == o_date_str].head(15).copy()

    st.write(f"Requests under evaluation for {o_date_str}: **{len(batch_reqs)}**")

    if st.button("Run CP-SAT Allocation", type="primary"):
        if batch_reqs.empty:
            st.write("No requests found for this date.")
        else:
            res = optimizer_engine.compare_optimizer_vs_baseline(batch_reqs, time_limit_seconds=float(o_timeout))
            opt = res["metrics"]["optimizer"]
            base = res["metrics"]["baseline"]

            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                st.metric("Requests Accommodated", f"{opt['assigned_count']} / {len(batch_reqs)}", f"{opt['fulfillment_rate']}%")
            with c_res2:
                st.metric("Excess Seats Wasted", f"{opt['total_excess_seats']} seats", f"-{base['total_excess_seats'] - opt['total_excess_seats']} vs Baseline")
            with c_res3:
                st.metric("Solver Runtime", f"{opt['runtime_seconds']}s")

            allocations = res["optimizer_result"]["allocations"]
            if allocations:
                st.write("**Allocations:**")
                st.dataframe(pd.DataFrame(allocations)[["request_id", "room_id", "priority_tier", "start_hour", "end_hour", "required_capacity", "allocated_room_capacity", "purpose"]], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 6. ROOM REALLOCATION & SWAPS
# -----------------------------------------------------------------------------
elif page == "Room Reallocation":
    render_page_header("Room Reallocation & Capacity Swaps", "Identify and execute zero-disruption room swaps to resolve space bottlenecks.")

    r_c1, r_c2 = st.columns([2, 1])
    with r_c1:
        swap_date = st.date_input("Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with r_c2:
        swap_max = st.slider("Proposals Limit", 2, 8, 4)

    swap_date_str = swap_date.strftime("%Y-%m-%d")

    if st.button("Scan for Reallocation Proposals", type="primary"):
        day_unallocated = reqs_df[reqs_df["request_date"] == swap_date_str].head(8)
        proposals = realloc_engine.generate_all_proposals(
            date_str=swap_date_str,
            unallocated_requests=day_unallocated,
            max_proposals=swap_max
        )
        st.write(f"Generated {len(proposals)} proposals.")

    stored = db.get_reallocation_proposals()
    if not stored:
        st.write("No reallocation proposals in database.")
    else:
        st.write(f"**Proposals in Database ({len(stored)}):**")
        for p in stored:
            p_id = p["proposal_id"]
            stat = p["status"]

            st.write(f"**{p_id}** — {p['summary']} (Status: `{stat}`)")
            if p["items"]:
                st.dataframe(pd.DataFrame(p["items"])[["title", "current_room_id", "proposed_room_id", "event_date", "start_hour", "duration_hours"]], use_container_width=True, hide_index=True)

            if stat == "Pending":
                b1, b2, _ = st.columns([1, 1, 4])
                with b1:
                    if st.button(f"Approve {p_id}", key=f"app_{p_id}"):
                        ok, msg = db.approve_reallocation_proposal(p_id, user["username"], avail_engine)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                with b2:
                    if st.button(f"Reject {p_id}", key=f"rej_{p_id}"):
                        ok, msg = db.reject_reallocation_proposal(p_id, user["username"])
                        if ok:
                            st.warning(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            st.markdown("---")


# -----------------------------------------------------------------------------
# 7. DEMAND FORECAST
# -----------------------------------------------------------------------------
elif page == "Demand Forecast":
    render_page_header("Demand Forecast (ML)", "Hourly room request forecasting using pure NumPy Linear Regression.")

    metrics_file = ROOT_DIR / "data" / "ml_evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as mf:
            eval_metrics = json.load(mf)
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.metric("Test MAE", f"{eval_metrics['model_test_metrics']['mae']:.4f}", f"{eval_metrics['improvement_pct_mae']:+}% vs Baseline")
        with m_c2:
            st.metric("Test RMSE", f"{eval_metrics['model_test_metrics']['rmse']:.4f}", f"{eval_metrics['improvement_pct_rmse']:+}% vs Baseline")
        with m_c3:
            st.metric("Test Samples", f"{eval_metrics['test_samples']} hours", "Chronological Split")

    st.write("**Inference Tool:**")
    fc_c1, fc_c2 = st.columns(2)
    with fc_c1:
        pred_date = st.date_input("Target Date", value=date(2025, 11, 25), min_value=date(2025, 8, 1), max_value=date(2025, 12, 15))
    with fc_c2:
        pred_hour = st.slider("Hour of Day", 8, 20, 14)

    if st.button("Predict Hourly Demand"):
        if not ml_forecaster.model:
            st.write("Model weights not available. Run scripts/train_model.py.")
        else:
            ts_recent = DemandForecaster.prepare_demand_series(reqs_df)
            pred = ml_forecaster.predict_slot(pred_date.strftime("%Y-%m-%d"), pred_hour, recent_history_df=ts_recent)

            st.write(f"Predicted Requests: **{pred['predicted_requests']}** (Baseline: {pred['historical_mean_baseline']})")
            free_cnt = sum(1 for r in rooms_df["room_id"] if avail_engine.is_room_available(r, pred_date.strftime("%Y-%m-%d"), pred_hour, 1.0)[0])
            st.write(f"Available Rooms: **{free_cnt}**")
            if pred["predicted_requests"] > free_cnt:
                st.warning(f"Capacity warning: Predicted demand ({pred['predicted_requests']}) exceeds free rooms ({free_cnt}).")
            else:
                st.info(f"Sufficient capacity available.")


# -----------------------------------------------------------------------------
# 8. SPACE ANALYTICS
# -----------------------------------------------------------------------------
elif page == "Space Analytics":
    render_page_header("Space Analytics", "Scheduled utilization across rooms and time slots.")

    tab1, tab2 = st.tabs(["Scheduled Utilization", "Occupancy Heatmap"])
    with tab1:
        util_data = util_engine.get_all_room_utilization()
        st.dataframe(util_data[["room_id", "building", "room_type", "capacity", "weekly_scheduled_hours", "scheduled_utilization_rate"]], use_container_width=True, hide_index=True)

    with tab2:
        hm = util_engine.get_hourly_heatmap_matrix()
        fig = px.imshow(
            hm,
            labels=dict(x="Hour", y="Weekday", color="Rooms"),
            x=[f"{h:02d}:00" for h in hm.columns],
            y=list(hm.index),
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 9. DATA & INGESTION API
# -----------------------------------------------------------------------------
elif page == "Data & Ingestion API":
    render_page_header("Data & Ingestion API", "Transactional ingestion service and REST endpoints.")

    st.write("FastAPI REST endpoints running on Port 8000: [OpenAPI Documentation](http://localhost:8000/docs)")

    up_type = st.selectbox("Entity Type to Import", ["rooms", "class_schedule", "events", "room_requests"])
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file is not None:
        try:
            up_df = pd.read_csv(csv_file)
            st.write(f"Preview ({len(up_df)} records):")
            st.dataframe(up_df.head(4), use_container_width=True)

            if st.button("Process Import"):
                records = up_df.to_dict(orient="records")
                res = ingest_service.ingest_payload(up_type, records, source_name=csv_file.name)
                st.write(f"Import {res['import_id']}: Status `{res['status']}` ({res['total_accepted']} accepted, {res['total_rejected']} rejected)")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.write("**Recent Imports:**")
    hist = db.get_data_imports()
    if hist:
        st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
    else:
        st.write("No imports recorded.")


# -----------------------------------------------------------------------------
# 10. ADMIN QUEUE
# -----------------------------------------------------------------------------
elif page == "Admin Queue":
    render_page_header("Admin Queue", "Pending student booking requests requiring review.")

    all_b = db.get_all_bookings()
    pending = all_b[all_b["status"] == "Pending"]

    if pending.empty:
        st.write("No pending requests in queue.")
    else:
        for _, r in pending.iterrows():
            req_id = r["request_id"]
            st.write(f"**Request {req_id}** — {r['purpose']} (by `{r['username']}`)")
            st.write(f"Target Room: {r['room_id'] or 'Unassigned'} | Date: {r['request_date']} | Time: {int(r['start_hour']):02d}:00 ({r['duration_hours']}h) | Attendees: {r['required_capacity']}")
            
            c_a, c_r, _ = st.columns([1, 1, 4])
            with c_a:
                if st.button("Approve", key=f"q_app_{req_id}"):
                    ok, msg = db.approve_booking(req_id, user["username"], avail_engine)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with c_r:
                if st.button("Reject", key=f"q_rej_{req_id}"):
                    ok, msg = db.reject_booking(req_id, user["username"], reason="Administrative prioritization")
                    if ok:
                        st.warning(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            st.markdown("---")
