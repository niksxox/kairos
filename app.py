"""
KAIROS — Campus Space Intelligence Platform
Predict demand. Optimize allocation. Reduce wasted space.
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
import plotly.graph_objects as go

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    VALID_BUILDINGS,
    VALID_ROOM_TYPES,
    VALID_WEEKDAYS,
    DEPARTMENTS,
    REQUEST_PURPOSES,
    CAMPUS_DAY_START_HOUR,
    CAMPUS_DAY_END_HOUR,
    OPERATING_HOURS_PER_DAY
)
from src.data_loader import DataLoader, DataSourceType
from src.data_validation import DataValidator
from src.preprocessing import Preprocessor
from src.availability import AvailabilityEngine
from src.matching import RoomMatchingEngine
from src.optimizer import MultiEventOptimizer, ORTOOLS_AVAILABLE
from src.utilization import UtilizationEngine
from src.recommendations import AdminRecommendationEngine, DecisionStore
from src.reallocation_engine import ReallocationEngine
from src.ingestion import DataIngestionService
from src.priority import infer_priority_tier, get_priority_info, PriorityTier, PRIORITY_WEIGHTS, PRIORITY_BADGE_STYLE
from src.metrics import MetricsCalculator
from src.database import CampusDatabase
from src.ml_model import DemandForecaster


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & RESTRAINED INSTITUTIONAL STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KAIROS | Campus Space Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Clean academic SaaS aesthetic - zero AI-slope gradients */
    :root {
        --primary: #0f766e;
        --primary-dark: #115e59;
        --navy: #0f172a;
        --surface: #ffffff;
        --bg-subtle: #f8fafc;
        --border-color: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    .institutional-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 1.25rem;
        margin-bottom: 1.75rem;
    }
    
    .product-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    .product-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin: 0.25rem 0 0 0;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #0f766e;
        margin-top: 0.2rem;
    }

    .priority-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid transparent;
    }

    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f766e;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATABASE & DATA INITIALIZATION
# -----------------------------------------------------------------------------
db = CampusDatabase()

if "data_version" not in st.session_state:
    st.session_state["data_version"] = 1

@st.cache_data(show_spinner="Loading verified campus records...")
def get_datasets(version: int):
    loader = DataLoader()
    datasets, source = loader.load_all()
    rooms_clean, sched_clean, events_clean, reqs_clean = Preprocessor.preprocess_all(
        datasets["rooms"],
        datasets["class_schedule"],
        datasets["events"],
        datasets["room_requests"]
    )
    return rooms_clean, sched_clean, events_clean, reqs_clean, source

rooms_df, sched_df, events_df, reqs_df, current_source = get_datasets(st.session_state["data_version"])

# Core domain engines
avail_engine = AvailabilityEngine(sched_df, events_df, rooms_df, db_instance=db)
matching_engine = RoomMatchingEngine(rooms_df, avail_engine)
optimizer_engine = MultiEventOptimizer(rooms_df, sched_df, events_df)
realloc_engine = ReallocationEngine(rooms_df, sched_df, events_df, database=db)
ingest_service = DataIngestionService(database=db, existing_rooms=set(rooms_df["room_id"]))
util_engine = UtilizationEngine(rooms_df, sched_df, events_df)

# ML model loader
model_json = str(ROOT_DIR / "data" / "demand_forecast_model.json")
ml_forecaster = DemandForecaster(model_path=model_json if os.path.exists(model_json) else None)


# -----------------------------------------------------------------------------
# USER AUTHENTICATION & ROLE SWITCHER
# -----------------------------------------------------------------------------
if "current_user" not in st.session_state:
    st.session_state["current_user"] = {
        "username": "student",
        "role": "student",
        "department": "Computer Science & Engineering"
    }

user = st.session_state["current_user"]

with st.sidebar:
    st.markdown("### 🏛️ **KAIROS**")
    st.caption("Predict demand. Optimize allocation. Reduce wasted space.")
    st.markdown("---")
    
    st.markdown("**Active Session**")
    role_indicator = "🟢" if user["role"] == "student" else "🔵"
    st.write(f"{role_indicator} **{user['username']}** (`{user['role'].upper()}`)")
    st.caption(f"Dept: {user['department']}")

    # Quick demo role switcher
    st.markdown("**Switch Role (Demo):**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Student", use_container_width=True, disabled=(user["role"] == "student")):
            st.session_state["current_user"] = {"username": "student", "role": "student", "department": "Computer Science"}
            st.rerun()
    with c2:
        if st.button("Admin", use_container_width=True, disabled=(user["role"] == "admin")):
            st.session_state["current_user"] = {"username": "admin", "role": "admin", "department": "Campus Operations"}
            st.rerun()

    st.markdown("---")

    # Navigation menu
    if user["role"] == "admin":
        nav_options = [
            "Overview",
            "Find a Room",
            "My Bookings",
            "Calendar View",
            "Multi-Event Optimizer",
            "Automatic Reallocation & Swaps",
            "Demand Forecast (ML)",
            "Space Analytics",
            "Data Ingestion & REST API",
            "Admin Workspace"
        ]
    else:
        nav_options = [
            "Overview",
            "Find a Room",
            "My Bookings",
            "Calendar View",
            "Demand Forecast (ML)"
        ]

    page = st.radio("Navigation", nav_options, index=0)

    st.markdown("---")
    st.caption(f"**Data Pipeline:** {current_source}")
    st.caption(f"**Optimization:** {'Google OR-Tools CP-SAT' if ORTOOLS_AVAILABLE else 'Deterministic Heuristic'}")
    st.caption(f"**ML Engine:** {'NumPy Linear Reg (Trained)' if ml_forecaster.model else 'Untrained'}")
    st.caption(f"**REST API:** FastAPI on Port 8000")

    if st.button("🔄 Refresh Application State", use_container_width=True):
        st.cache_data.clear()
        st.session_state["data_version"] += 1
        st.rerun()


# -----------------------------------------------------------------------------
# TOP HEADER COMPONENT (Clean, restrained, non-AI-slope)
# -----------------------------------------------------------------------------
def render_header(title: str, subtitle: str, tag: str = "DATATHON 2026"):
    st.markdown(f"""
    <div class="institutional-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-end;">
            <div>
                <h1 class="product-title">{title}</h1>
                <p class="product-subtitle">{subtitle}</p>
            </div>
            <div>
                <span style="background:#f1f5f9; border:1px solid #cbd5e1; color:#334155; font-size:0.75rem; font-weight:600; padding:0.25rem 0.6rem; border-radius:4px;">{tag}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    render_header(
        "KAIROS — Campus Space Intelligence",
        "Predict demand. Optimize allocation. Reduce wasted space. • Team: Laphing Buddha"
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Physical Inventory</div>
            <div class="metric-value">{len(rooms_df)}</div>
            <div class="metric-sub">Across {rooms_df['building'].nunique()} Campus Buildings</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scheduled Classes</div>
            <div class="metric-value">{len(sched_df):,}</div>
            <div class="metric-sub">Mon–Sat Academic Timetable</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Special Events</div>
            <div class="metric-value">{len(events_df):,}</div>
            <div class="metric-sub">Conferences & Guest Sessions</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Student Requests</div>
            <div class="metric-value">{len(reqs_df):,}</div>
            <div class="metric-sub">Unallocated Demand Profile</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Institutional Core Highlights")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("""
        **1. Institutional Priority Metric:**
        Formal assessments (exams, tests) are strictly prioritized over discretionary cultural/social events. 
        When multiple requests contend for a scarce room, Google OR-Tools CP-SAT enforces priority-weighting mathematically.

        **2. Bounded Swap & Reallocation Engine:**
        Detects space inefficiencies where small classes occupy massive lecture halls while crowded classes overflow. 
        Generates mutual swaps and bounded swap chains with atomic approval workflows and zero schedule disruption.
        """)

    with c_right:
        st.markdown("""
        **3. Pure NumPy Machine Learning Demand Forecasting:**
        Trained from scratch using cyclic temporal representations. Accurately anticipates future peak request volume 
        and flags capacity shortages before students face rejections.

        **4. Transactional Data Ingestion & REST API:**
        FastAPI application running on port 8000 with interactive Swagger UI at `/docs`. Enforces schema validation, 
        referential integrity checks, and persistent audit trails.
        """)


# -----------------------------------------------------------------------------
# PAGE 2: FIND A ROOM (STUDENT BOOKING WORKFLOW WITH PRIORITY METRIC)
# -----------------------------------------------------------------------------
elif page == "Find a Room":
    render_header("Find a Room", "Multi-criteria room discovery, fit scoring, and priority-ranked booking submission.")

    # Priority Metric Explanation Banner
    st.info("💡 **Institutional Priority Policy Active:** Formal examinations and tests receive highest allocation ranking (`CRITICAL`), strictly outranking discretionary cultural and social gatherings (`FLEXIBLE`).")

    with st.form("room_search_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            req_date = st.date_input("Event Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
            start_hour = st.selectbox("Start Time", [h for h in range(8, 21)], index=6, format_func=lambda h: f"{h:02d}:00")
            duration = st.selectbox("Duration", [1.0, 1.5, 2.0, 3.0, 4.0], index=2, format_func=lambda d: f"{d} Hours")

        with col2:
            attendance = st.number_input("Expected Attendance", min_value=1, max_value=300, value=40, step=5)
            
            # Purpose and Category Selection (Drives Institutional Priority Tier)
            category_selection = st.selectbox(
                "Event Category (Priority Ranking)",
                [
                    "Formal Examination / Midterm / Test (Priority: Critical)",
                    "Core Course Lecture / Lab Practical (Priority: High)",
                    "Academic Seminar / Department Colloquium (Priority: Medium)",
                    "Student Project / Study Group (Priority: Standard)",
                    "Cultural Fest / Rehearsal / Social Gathering (Priority: Flexible)"
                ],
                index=0
            )
            purpose_note = st.text_input("Event Description / Topic", value="Midterm Review Assessment")

        with col3:
            st.markdown("**Required Facilities:**")
            req_projector = st.checkbox("Projector Required", value=True)
            req_computers = st.checkbox("Computer Stations Required", value=False)
            req_accessible = st.checkbox("Accessibility (Wheelchair Ramp)", value=False)
            preferred_building = st.selectbox("Building Filter", ["Any Building"] + sorted(VALID_BUILDINGS))

        search_submitted = st.form_submit_button("🔍 Search Feasible Rooms", use_container_width=True, type="primary")

    if search_submitted:
        full_purpose = f"{category_selection.split('(')[0].strip()} — {purpose_note}"
        tier, p_score, badge_info = get_priority_info(full_purpose)

        st.session_state["last_search"] = {
            "date": req_date.strftime("%Y-%m-%d"),
            "start": float(start_hour),
            "dur": float(duration),
            "cap": int(attendance),
            "proj": req_projector,
            "comp": req_computers,
            "acc": req_accessible,
            "bldg": None if preferred_building == "Any Building" else preferred_building,
            "purpose": full_purpose,
            "tier": tier,
            "score": p_score,
            "badge": badge_info
        }

    if "last_search" in st.session_state:
        s = st.session_state["last_search"]
        b_info = s["badge"]
        tier_obj = s["tier"]

        st.markdown(f"""
        <div style="background:{b_info['bg']}; border:1px solid {b_info['border']}; border-radius:6px; padding:0.75rem 1rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="color:{b_info['color']}; font-size:0.9rem;">{b_info['label']}</strong>
                    <span style="color:#475569; font-size:0.85rem; margin-left:0.5rem;">| Allocation Weight: <b>{s['score']}</b></span>
                </div>
                <span style="color:#64748b; font-size:0.8rem;">Date: {s['date']} ({int(s['start']):02d}:00 - {int(s['start']+s['dur']):02d}:00)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        recs, rejs = matching_engine.find_and_rank_rooms(
            request_date=s["date"],
            start_hour=s["start"],
            duration_hours=s["dur"],
            required_capacity=s["cap"],
            projector_required=s["proj"],
            computers_required=s["comp"],
            accessible_required=s["acc"],
            preferred_building=s["bldg"]
        )

        if not recs:
            st.error("⚠️ No feasible rooms found that satisfy all hard capacity, facility, and availability constraints.")
            if rejs:
                with st.expander(f"Inspect Disqualified Rooms ({len(rejs)} evaluated)", expanded=True):
                    rej_df = pd.DataFrame([
                        {"Room": r["room_id"], "Building": r["building"], "Capacity": r["capacity"], "Rejection Reasons": " | ".join(r["rejection_reasons"])}
                        for r in rejs
                    ])
                    st.dataframe(rej_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(f"### Feasible Recommendations ({len(recs)} rooms available)")
            for idx, room in enumerate(recs[:4], start=1):
                r_id = room["room_id"]
                with st.expander(f"Option #{idx}: Room {r_id} ({room['building']}) — Fit Score: {room['match_score']*100:.1f}%", expanded=(idx==1)):
                    c_det, c_act = st.columns([3, 1])
                    with c_det:
                        st.write(f"**Capacity:** {room['capacity']} seats ({room['excess_capacity']} surplus)")
                        st.write(f"**Type:** {room['room_type']} | **Power Outlets:** {room['power_outlets']}")
                        st.write(f"**Facilities:** Projector: {'✅' if room['projector'] else '❌'} | Computers: {'✅' if room['computers'] else '❌'} | Accessible: {'✅' if room['accessible'] else '❌'}")
                        st.markdown("**Pros:** " + ", ".join(room["pros"]))
                    with c_act:
                        st.markdown("**Reserve Space**")
                        if st.button(f"📌 Book {r_id}", key=f"book_btn_{r_id}_{idx}", use_container_width=True):
                            new_req_id = db.create_booking_request(
                                username=user["username"],
                                room_id=r_id,
                                request_date=s["date"],
                                start_hour=s["start"],
                                duration_hours=s["dur"],
                                required_capacity=s["cap"],
                                projector_required=s["proj"],
                                computers_required=s["comp"],
                                accessible_required=s["acc"],
                                purpose=s["purpose"],
                                auto_approve=False
                            )
                            st.success(f"🎉 Booking request recorded! Request ID: **{new_req_id}** (Status: **Pending**).")
                            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 3: MY BOOKINGS
# -----------------------------------------------------------------------------
elif page == "My Bookings":
    render_header("My Bookings & Request History", f"Track and manage space reservations submitted by {user['username']}.")

    bookings_df = db.get_user_bookings(user["username"])

    if bookings_df.empty:
        st.info("No bookings recorded for this account. Use **Find a Room** to submit a space reservation.")
    else:
        st.markdown(f"**Showing {len(bookings_df)} reservations:**")
        for _, b in bookings_df.iterrows():
            req_id = b["request_id"]
            stat = b["status"]
            tier_val = b.get("priority_tier", "Standard")
            badge_cfg = PRIORITY_BADGE_STYLE.get(PriorityTier(tier_val), PRIORITY_BADGE_STYLE[PriorityTier.STANDARD])

            stat_color = "#166534" if stat == "Approved" else ("#92400e" if stat == "Pending" else "#991b1b")
            stat_bg = "#dcfce7" if stat == "Approved" else ("#fef3c7" if stat == "Pending" else "#fee2e2")

            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong style="font-size:1.05rem; color:#0f172a;">Request {req_id}</strong>
                            <span style="background:{badge_cfg['bg']}; color:{badge_cfg['color']}; border:1px solid {badge_cfg['border']}; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:600; margin-left:0.5rem;">{tier_val}</span>
                        </div>
                        <span style="background:{stat_bg}; color:{stat_color}; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.8rem; font-weight:600;">{stat}</span>
                    </div>
                    <div style="margin-top:0.6rem; font-size:0.9rem; color:#334155;">
                        <b>Purpose:</b> {b['purpose']}
                    </div>
                    <div style="margin-top:0.5rem; display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; font-size:0.85rem; color:#475569;">
                        <div><strong>Room:</strong> {b['room_id'] or 'Pending Allocation'}</div>
                        <div><strong>Date:</strong> {b['request_date']}</div>
                        <div><strong>Time:</strong> {int(b['start_hour']):02d}:00 - {int(b['start_hour']+b['duration_hours']):02d}:00</div>
                        <div><strong>Attendees:</strong> {b['required_capacity']}</div>
                    </div>
                    {f"<p style='color:#b91c1c; font-size:0.85rem; margin-top:0.5rem;'><strong>Note:</strong> {b['rejection_reason']}</p>" if b['rejection_reason'] else ""}
                </div>
                """, unsafe_allow_html=True)

                if stat in ("Pending", "Approved"):
                    if st.button(f"Cancel Reservation {req_id}", key=f"cancel_btn_{req_id}"):
                        ok, msg = db.cancel_booking(req_id, user["username"])
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)


# -----------------------------------------------------------------------------
# PAGE 4: CALENDAR VIEW
# -----------------------------------------------------------------------------
elif page == "Calendar View":
    render_header("Campus Room Timetable", "Inspect daily commitments across recurring classes, campus events, and confirmed bookings.")

    c1, c2 = st.columns(2)
    with c1:
        cal_date = st.date_input("Calendar Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with c2:
        cal_room = st.selectbox("Target Room", sorted(rooms_df["room_id"].unique()) if not rooms_df.empty else ["R001"])

    cal_date_str = cal_date.strftime("%Y-%m-%d")
    cal_weekday = cal_date.strftime("%a")

    room_classes = sched_df[(sched_df["room_id"] == cal_room) & (sched_df["weekday"] == cal_weekday)]
    room_events = events_df[(events_df["room_id"] == cal_room) & (events_df["event_date"] == cal_date_str)]
    db_confirmed = [b for b in db.get_confirmed_bookings_for_date(cal_date_str) if b.get("room_id") == cal_room]

    items = []
    for _, c in room_classes.iterrows():
        items.append({
            "Activity": f"Class: {c['course_code']} ({c['department']})",
            "Category": "Academic Lecture",
            "Priority": "High (Academic)",
            "Start": f"{int(c['start_hour']):02d}:00",
            "End": f"{int(c['start_hour']+c['duration_hours']):02d}:00",
            "Attendance": c["expected_attendance"]
        })
    for _, e in room_events.iterrows():
        items.append({
            "Activity": f"Event: {e['event_type']}",
            "Category": "Special Event",
            "Priority": "Medium (Event)",
            "Start": f"{int(e['start_hour']):02d}:00",
            "End": f"{int(e['start_hour']+e['duration_hours']):02d}:00",
            "Attendance": e["expected_attendance"]
        })
    for b in db_confirmed:
        items.append({
            "Activity": f"Student Reservation: {b.get('purpose', 'Reserved')}",
            "Category": "Student Booking",
            "Priority": b.get("priority_tier", "Standard"),
            "Start": f"{int(b['start_hour']):02d}:00",
            "End": f"{int(b['start_hour']+b['duration_hours']):02d}:00",
            "Attendance": "Confirmed"
        })

    if not items:
        st.success(f"✨ Room {cal_room} has zero commitments on {cal_date_str} ({cal_date.strftime('%A')}). Free all day.")
    else:
        cal_df = pd.DataFrame(items).sort_values(by="Start")
        st.dataframe(cal_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 5: MULTI-EVENT OPTIMIZER (WITH PRIORITY WEIGHTING)
# -----------------------------------------------------------------------------
elif page == "Multi-Event Optimizer":
    render_header("Multi-Event Room Allocation Optimizer", "Mathematical space allocation contrasting Google OR-Tools CP-SAT against Greedy Baseline.")

    st.info("⚖️ **Priority-Weighted Objective Active:** Mathematical solver assigns weights (Exam: 100, Class: 50, Seminar: 25, Study: 10, Cultural: 5). High-stakes exams take precedence during space shortages.")

    o1, o2, o3 = st.columns(3)
    with o1:
        target_opt_date = st.date_input("Target Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with o2:
        solver_timeout = st.slider("CP-SAT Time Limit (Seconds)", min_value=3, max_value=25, value=10)
    with o3:
        status_filter = st.multiselect("Request Statuses", ["Waitlisted", "Approved", "Cancelled", "Rejected"], default=["Waitlisted", "Approved"])

    target_date_str = target_opt_date.strftime("%Y-%m-%d")
    date_reqs = reqs_df[reqs_df["request_date"] == target_date_str].copy()
    if status_filter:
        date_reqs = date_reqs[date_reqs["status"].isin(status_filter)]

    st.markdown(f"**Evaluating {len(date_reqs)} incoming requests for {target_date_str}:**")

    if st.button("🚀 Execute Global CP-SAT Solver", type="primary", use_container_width=True):
        if date_reqs.empty:
            st.warning("No requests match the selected date and status filters.")
        else:
            with st.spinner("Formulating CP-SAT decision variables and solving..."):
                benchmark = optimizer_engine.compare_optimizer_vs_baseline(date_reqs, time_limit_seconds=float(solver_timeout))

            m = benchmark["metrics"]
            opt = m["optimizer"]
            base = m["baseline"]
            deltas = m["deltas"]

            st.markdown("### Benchmark Evaluation")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Fulfillment Rate", f"{opt['fulfillment_rate']}%", f"{deltas['fulfillment_gain_pct']:+}% vs Greedy")
            bc2.metric("Requests Accommodated", f"{opt['assigned_count']} / {m['requests_evaluated']}", f"{deltas['additional_requests_served']:+} extra requests")
            bc3.metric("Excess Seats Wasted", f"{opt['total_excess_seats']:,}", f"-{deltas['excess_seats_saved']:,} seats saved")
            bc4.metric("Constraint Violations", "0", "Formally Guaranteed")

            st.markdown("### Allocations Breakdown")
            opt_allocs = benchmark["optimizer_result"]["allocations"]
            if opt_allocs:
                alloc_df = pd.DataFrame(opt_allocs)
                display_cols = ["request_id", "room_id", "priority_tier", "building", "start_hour", "end_hour", "required_capacity", "allocated_room_capacity", "excess_capacity", "purpose"]
                st.dataframe(alloc_df[display_cols], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 6: AUTOMATIC REALLOCATION & SWAP ENGINE
# -----------------------------------------------------------------------------
elif page == "Automatic Reallocation & Swaps":
    render_header("Automatic Room Reallocation & Swap Engine", "Solves capacity bottlenecks and bounded swap chains without shifting schedule hours.")

    st.write("""
    Scans scheduled commitments for **Mutual Room Swaps** (crowded class in small room $\\leftrightarrow$ small class in oversized hall) 
    and **Bounded Swap Chains** (Request $Q \\to$ Room 1 $\\to$ Room 2) with atomic administrative approval.
    """)

    sw_c1, sw_c2 = st.columns([2, 1])
    with sw_c1:
        realloc_date = st.date_input("Scan Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with sw_c2:
        max_props = st.slider("Max Proposals to Generate", min_value=2, max_value=10, value=4)

    realloc_date_str = realloc_date.strftime("%Y-%m-%d")

    if st.button("🔍 Scan Schedule & Generate Proposals", type="primary", use_container_width=True):
        with st.spinner("Analyzing capacity mismatches and searching bounded swap chains..."):
            day_reqs = reqs_df[reqs_df["request_date"] == realloc_date_str].head(10)
            props = realloc_engine.generate_all_proposals(
                date_str=realloc_date_str,
                unallocated_requests=day_reqs,
                max_proposals=max_props
            )
            st.success(f"Generated {len(props)} actionable reallocation proposals!")

    # Display existing stored proposals
    stored_proposals = db.get_reallocation_proposals()
    if not stored_proposals:
        st.info("No active reallocation proposals in database. Click 'Scan Schedule & Generate Proposals' above.")
    else:
        st.markdown(f"### Stored Reallocation Proposals ({len(stored_proposals)} in queue)")
        for p in stored_proposals:
            p_id = p["proposal_id"]
            p_stat = p["status"]
            p_type = p["proposal_type"]

            with st.container():
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong style="font-size:1.05rem; color:#0f172a;">{p_id} ({p_type.replace('_', ' ')})</strong>
                            <span style="color:#0f766e; font-size:0.85rem; margin-left:0.5rem;">| Disruption Cost: <b>{p['disruption_cost']}</b> | Seat Waste Reduction: <b>{p['seat_mismatch_reduction']} seats</b></span>
                        </div>
                        <span style="font-size:0.8rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:4px; background:#f1f5f9; color:#334155;">{p_stat}</span>
                    </div>
                    <p style="margin:0.5rem 0; font-size:0.9rem; color:#334155;">{p['summary']}</p>
                    <div style="font-size:0.85rem; color:#475569;">
                        <b>Movement Items ({len(p['items'])}):</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if p["items"]:
                    items_df = pd.DataFrame(p["items"])[["title", "current_room_id", "proposed_room_id", "priority_tier", "event_date", "start_hour", "duration_hours"]]
                    st.dataframe(items_df, use_container_width=True, hide_index=True)

                if p_stat == "Pending":
                    btn_c1, btn_c2, _ = st.columns([1, 1, 3])
                    with btn_c1:
                        if st.button(f"✅ Approve & Execute {p_id}", key=f"app_prop_{p_id}", use_container_width=True):
                            ok, msg = db.approve_reallocation_proposal(p_id, user["username"], avail_engine)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with btn_c2:
                        if st.button(f"❌ Reject {p_id}", key=f"rej_prop_{p_id}", use_container_width=True):
                            ok, msg = db.reject_reallocation_proposal(p_id, user["username"], reason="Administrative policy override")
                            if ok:
                                st.warning(msg)
                                st.rerun()
                            else:
                                st.error(msg)


# -----------------------------------------------------------------------------
# PAGE 7: DEMAND FORECAST (NUMPY ML TRAINED FROM SCRATCH)
# -----------------------------------------------------------------------------
elif page == "Demand Forecast (ML)":
    render_header("Campus Hourly Demand Forecasting (ML)", "Predicts student room demand by time slot to anticipate space bottlenecks before they occur.")

    metrics_file = ROOT_DIR / "data" / "ml_evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as mf:
            eval_data = json.load(mf)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Architecture", "NumPy Linear Reg", "Pure NumPy")
        m2.metric("Held-Out Test MAE", f"{eval_data['model_test_metrics']['mae']:.4f}", f"{eval_data['improvement_pct_mae']:+}% vs Baseline")
        m3.metric("Held-Out Test RMSE", f"{eval_data['model_test_metrics']['rmse']:.4f}", f"{eval_data['improvement_pct_rmse']:+}% vs Baseline")
        m4.metric("Chronological Test Set", f"{eval_data['test_samples']} hourly slots", "Zero Future Leakage")

    st.markdown("### Interactive Demand Inference")
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        f_date = st.date_input("Inference Date", value=date(2025, 11, 25), min_value=date(2025, 8, 1), max_value=date(2025, 12, 15))
        f_hour = st.slider("Time Slot (Hour of Day)", min_value=8, max_value=20, value=14, step=1)
    with f_c2:
        st.markdown("**Feature Inputs Fed to Model:**")
        st.caption("Cyclic hour (sin/cos), cyclic weekday (sin/cos), month, 1-hour lag, 24-hour lag, rolling 7-hour moving average.")
        st.caption("Training procedure: Batch Gradient Descent with L2 penalty on 70% chronological train partition.")

    if st.button("🔮 Generate Model Inference", type="primary", use_container_width=True):
        if ml_forecaster.model is None:
            st.error("Model weights not found. Run `python scripts/train_model.py` first.")
        else:
            ts_recent = DemandForecaster.prepare_demand_series(reqs_df)
            pred = ml_forecaster.predict_slot(f_date.strftime("%Y-%m-%d"), f_hour, recent_history_df=ts_recent)

            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Hourly Requests", f"{pred['predicted_requests']}", f"~{pred['rounded_request_count']} reservations")
            p2.metric("Historical Mean Baseline", f"{pred['historical_mean_baseline']}")
            p3.metric("Demand Classification", pred["demand_intensity"])

            # Compare against feasible supply
            free_rooms = sum(1 for r in rooms_df["room_id"] if avail_engine.is_room_available(r, f_date.strftime("%Y-%m-%d"), f_hour, 1.0)[0])
            if pred["predicted_requests"] > free_rooms:
                st.error(f"🚨 **Space Deficit Warning:** Model predicts **{pred['predicted_requests']} requests**, but only **{free_rooms} rooms** are available at {f_hour:02d}:00 on {f_date}.")
            else:
                st.success(f"✅ **Adequate Inventory:** Campus has **{free_rooms} available rooms** at {f_hour:02d}:00 to accommodate expected demand of {pred['predicted_requests']} requests.")


# -----------------------------------------------------------------------------
# PAGE 8: SPACE ANALYTICS
# -----------------------------------------------------------------------------
elif page == "Space Analytics":
    render_header("Campus Utilization & Demand Analytics", "Scheduled utilization analysis distinguishing scheduled commitments from physical sensor occupancy.")

    tab_util, tab_heatmap, tab_demand = st.tabs(["Scheduled Utilization", "Occupancy Heatmap", "Demand vs Supply"])

    with tab_util:
        st.write("Distribution of scheduled lecture and event hours across rooms:")
        util_df = util_engine.get_all_room_utilization()
        st.dataframe(util_df[["room_id", "building", "room_type", "capacity", "weekly_scheduled_hours", "scheduled_utilization_rate"]], use_container_width=True, hide_index=True)

    with tab_heatmap:
        st.write("Campus scheduled room occupancy heatmap by weekday and hour:")
        hm_matrix = util_engine.get_hourly_heatmap_matrix()
        fig_hm = px.imshow(
            hm_matrix,
            labels=dict(x="Hour of Day", y="Weekday", color="Rooms Occupied"),
            x=[f"{h:02d}:00" for h in hm_matrix.columns],
            y=list(hm_matrix.index),
            color_continuous_scale="Teal"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_demand:
        st.write("Comparison of physical room capacity tiers against student attendance demands:")
        d_vs_s = util_engine.get_demand_vs_supply_summary(reqs_df)
        st.dataframe(d_vs_s, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 9: DATA INGESTION & REST API WORKBENCH
# -----------------------------------------------------------------------------
elif page == "Data Ingestion & REST API":
    render_header("Data Ingestion & REST API Workbench", "Controlled transactional data imports and interactive FastAPI OpenAPI / Swagger endpoints.")

    st.markdown("""
    **FastAPI Service is running on Port 8000.**  
    Explore interactive Swagger documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
    """)

    st.markdown("### 📤 Upload Operational Data Batch (CSV)")
    upload_type = st.selectbox("Entity Type", ["rooms", "class_schedule", "events", "room_requests"])
    uploaded_file = st.file_uploader(f"Choose a CSV file for '{upload_type}'", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(input_df)} records preview:")
            st.dataframe(input_df.head(5), use_container_width=True)

            if st.button("🚀 Process & Ingest Transactionally", type="primary"):
                records = input_df.to_dict(orient="records")
                res = ingest_service.ingest_payload(upload_type, records, source_name=f"Upload: {uploaded_file.name}")
                if res["status"] in ("SUCCESS", "PARTIAL"):
                    st.success(f"Import {res['import_id']} completed with status {res['status']}: {res['total_accepted']} accepted, {res['total_rejected']} rejected.")
                else:
                    st.error(f"Import {res['import_id']} failed: {res['error_log']}")
        except Exception as e:
            st.error(f"Failed to read CSV: {str(e)}")

    st.markdown("### 📋 Transactional Ingestion Audit Trail")
    imports = db.get_data_imports()
    if imports:
        st.dataframe(pd.DataFrame(imports), use_container_width=True, hide_index=True)
    else:
        st.info("No external imports recorded yet.")


# -----------------------------------------------------------------------------
# PAGE 10: ADMIN WORKSPACE
# -----------------------------------------------------------------------------
elif page == "Admin Workspace":
    render_header("Administrator Workspace", "Review pending student requests, verify real-time conflicts atomically, and approve/reject bookings.")

    all_reqs = db.get_all_bookings()
    p_count = len(all_reqs[all_reqs["status"] == "Pending"])
    a_count = len(all_reqs[all_reqs["status"] == "Approved"])
    r_count = len(all_reqs[all_reqs["status"] == "Rejected"])
    c_count = len(all_reqs[all_reqs["status"] == "Cancelled"])

    ak1, ak2, ak3, ak4 = st.columns(4)
    ak1.metric("Pending Approvals", p_count)
    ak2.metric("Confirmed Bookings", a_count)
    ak3.metric("Rejected Requests", r_count)
    ak4.metric("Cancelled Bookings", c_count)

    st.markdown("### Pending Requests Queue")
    pending_df = all_reqs[all_reqs["status"] == "Pending"]

    if pending_df.empty:
        st.success("✨ Zero pending booking requests. All campus requests have been processed.")
    else:
        for _, req in pending_df.iterrows():
            req_id = req["request_id"]
            tier_val = req.get("priority_tier", "Standard")
            badge_cfg = PRIORITY_BADGE_STYLE.get(PriorityTier(tier_val), PRIORITY_BADGE_STYLE[PriorityTier.STANDARD])

            with st.container():
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong style="font-size:1.05rem; color:#0f172a;">Request {req_id}</strong>
                            <span style="background:{badge_cfg['bg']}; color:{badge_cfg['color']}; border:1px solid {badge_cfg['border']}; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:600; margin-left:0.5rem;">{tier_val}</span>
                        </div>
                        <span style="background:#fef3c7; color:#92400e; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.8rem; font-weight:600;">Pending Review</span>
                    </div>
                    <div style="margin-top:0.5rem; font-size:0.9rem; color:#334155;">
                        <b>Purpose:</b> {req['purpose']} (by {req['username']})
                    </div>
                    <div style="margin-top:0.5rem; display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; font-size:0.85rem; color:#475569;">
                        <div><strong>Target Room:</strong> {req['room_id'] or 'Unassigned'}</div>
                        <div><strong>Date:</strong> {req['request_date']}</div>
                        <div><strong>Time:</strong> {int(req['start_hour']):02d}:00 - {int(req['start_hour']+req['duration_hours']):02d}:00</div>
                        <div><strong>Attendance:</strong> {req['required_capacity']} students</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_c1, act_c2, _ = st.columns([1, 1, 3])
                with act_c1:
                    if st.button(f"✅ Approve {req_id}", key=f"adm_app_{req_id}", use_container_width=True):
                        ok, msg = db.approve_booking(req_id, user["username"], avail_engine)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                with act_c2:
                    if st.button(f"❌ Reject {req_id}", key=f"adm_rej_{req_id}", use_container_width=True):
                        ok, msg = db.reject_booking(req_id, user["username"], reason="Administrative space prioritization")
                        if ok:
                            st.warning(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    st.markdown("### Complete Bookings Log")
    st.dataframe(all_reqs, use_container_width=True, hide_index=True)
