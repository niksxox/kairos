"""
Kairos - CampusSpace Intelligence
Interactive Decision-Support Prototype for Educational Space Optimization
Product: Kairos | Team: Laphing Buddha | DATATHON 2026
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import (
    VALID_BUILDINGS,
    VALID_ROOM_TYPES,
    DEPARTMENTS,
    REQUEST_PURPOSES,
    CAMPUS_DAY_START_HOUR,
    CAMPUS_DAY_END_HOUR,
    OPERATING_HOURS_PER_DAY
)
from src.data_loader import DataLoader, DataSourceType, generate_demo_datasets
from src.data_validation import DataValidator
from src.preprocessing import Preprocessor
from src.availability import AvailabilityEngine
from src.matching import RoomMatchingEngine
from src.optimizer import MultiEventOptimizer, ORTOOLS_AVAILABLE
from src.utilization import UtilizationEngine
from src.recommendations import AdminRecommendationEngine, DecisionStore, CapacitySwapEngine
from src.metrics import MetricsCalculator


# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & CUSTOM STYLING (Teal/Navy Campus-Tech Palette)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kairos | CampusSpace Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Theme Accents */
    :root {
        --primary: #0f766e;       /* Deep Teal */
        --primary-dark: #0f172a;  /* Navy Slate */
        --primary-light: #f0fdfa; /* Ice Teal */
        --accent: #0284c7;        /* Sky Blue */
        --text-main: #1e293b;
        --border-color: #e2e8f0;
    }
    
    /* Top Header Bar */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        color: #ffffff;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        color: #99f6e4;
        margin-top: 0.25rem;
        font-weight: 400;
    }
    .badge-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #ccfbf1;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #0f766e;
        margin-top: 0.25rem;
    }

    /* Recommendation Pill & Cards */
    .rec-card {
        background: #ffffff;
        border-left: 4px solid #0f766e;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 0.375rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .rec-card-warn {
        border-left-color: #f59e0b;
    }
    .rec-card-swap {
        border-left-color: #6366f1;
    }

    /* Status indicators */
    .badge-approved {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-rejected {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-pending {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* System Disclaimer Notice */
    .system-disclaimer {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 0.375rem;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #475569;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA CACHING & PIPELINE WRAPPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading campus datasets...")
def load_and_preprocess_data(source_override=None):
    loader = DataLoader()
    raw_datasets, source_name = loader.load_all()
    cleaned = Preprocessor.process_all(raw_datasets)
    validator = DataValidator(raw_datasets)
    report = validator.validate_all()
    return cleaned, source_name, report


# Initialize session state for persistent data reloads and decision store
if "data_version" not in st.session_state:
    st.session_state["data_version"] = 1

cleaned_data, current_source, dq_report = load_and_preprocess_data(st.session_state["data_version"])
rooms_df = cleaned_data["rooms"]
sched_df = cleaned_data["class_schedule"]
events_df = cleaned_data["events"]
reqs_df = cleaned_data["room_requests"]

# Core domain engines
avail_engine = AvailabilityEngine(sched_df, events_df, rooms_df)
matching_engine = RoomMatchingEngine(rooms_df, avail_engine)
optimizer_engine = MultiEventOptimizer(rooms_df, sched_df, events_df)
util_engine = UtilizationEngine(rooms_df, sched_df, events_df)
decision_store = DecisionStore()
admin_engine = AdminRecommendationEngine(rooms_df, sched_df, events_df, reqs_df, decision_store)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ **Kairos**")
    st.caption("CampusSpace Decision-Support | Laphing Buddha")
    
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Find a Room",
            "Multi-Event Optimizer",
            "Capacity Swaps",
            "Utilization Analytics",
            "Admin Decision Support",
            "Data Quality & Sources"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("**System Snapshot**")
    st.caption(f"**Data Source:** {current_source}")
    st.caption(f"**Inventory:** {len(rooms_df)} Rooms | {int(rooms_df['capacity'].sum() if not rooms_df.empty else 0):,} Seats")
    st.caption(f"**OR-Tools Solver:** {'✅ Active (CP-SAT)' if ORTOOLS_AVAILABLE else '⚠️ Greedy Fallback'}")

    if st.button("🔄 Reload Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state["data_version"] += 1
        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    st.markdown("""
    <div class="brand-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1 class="brand-title">Kairos <span class="badge-tag">DATATHON 2026</span></h1>
                <p class="brand-subtitle">The right space, at the right time. Intelligent campus room matching & schedule optimization.</p>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.9rem; color:#ccfbf1; font-weight:600;">Team: Laphing Buddha</span><br/>
                <span style="font-size:0.8rem; color:#99f6e4;">Track: Education Space Optimization</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Executive KPIs
    kpi = MetricsCalculator.compute_summary_kpis(rooms_df, sched_df, events_df, reqs_df)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Rooms</div>
            <div class="metric-value">{kpi['total_rooms']}</div>
            <div class="metric-sub">{kpi['total_seats']:,} campus seats</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scheduled Classes</div>
            <div class="metric-value">{kpi['total_classes_scheduled']:,}</div>
            <div class="metric-sub">Mon–Sat recurring</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Special Events</div>
            <div class="metric-value">{kpi['total_events_scheduled']:,}</div>
            <div class="metric-sub">Aug–Nov calendar</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unallocated Demand</div>
            <div class="metric-value">{kpi['total_incoming_requests']:,}</div>
            <div class="metric-sub">{kpi['theoretically_feasible_requests']:,} capacity-viable</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 🎯 Problem & Innovation")
        st.write("""
        **The Campus Has Space. Students Don't.**
        Traditional campus booking systems suffer from fragmented schedules, locked rooms, and capacity mismatches. 
        Small classes occupy massive lecture halls while 100+ student clubs and study sessions are waitlisted or rejected.
        
        **Kairos** delivers technical decision-support to solve this paradox:
        1. **Explainable Real-Time Matching:** Eliminates undersized or conflicting recommendations through strict multi-constraint pruning.
        2. **Global Multi-Event Optimization:** Leverages Google OR-Tools CP-SAT to maximize fulfilled requests across the entire student body without schedule collisions.
        3. **Capacity-Aware Swaps:** Identifies zero-disruption room exchanges between crowded and oversized concurrent sessions.
        4. **True Scheduled Utilization:** Merges overlapping bookings to prevent double-counting and visualizes genuine demand-supply imbalances.
        """)
        
        st.info("💡 **Ready to find space?** Navigate to **Find a Room** in the sidebar, or run the global optimizer on **Multi-Event Optimizer**.")

    with col_right:
        st.markdown("### 🏢 Physical Space Inventory")
        if not rooms_df.empty:
            fig_types = px.pie(
                rooms_df,
                names="room_type",
                title="Room Inventory by Room Type",
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig_types.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
            st.plotly_chart(fig_types, use_container_width=True)

    st.markdown("""
    <div class="system-disclaimer">
        ℹ️ <strong>Prototype & Transparency Notice:</strong> All data displayed is derived from the official DATATHON 2026 synthetic dataset. 
        Kairos operates as an analytical decision-support system proposing optimized allocations. Real university room locks require authorized campus administrative approval.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: FIND A ROOM (REQUEST FORM & MATCHING)
# -----------------------------------------------------------------------------
elif page == "Find a Room":
    st.markdown("## 🔍 Find a Room")
    st.write("Submit student or faculty requirements to search and rank available rooms in real-time.")

    with st.form("room_request_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            req_date = st.date_input("Event Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
            req_start = st.slider("Start Hour (24h)", min_value=8, max_value=21, value=14, step=1)
        with col2:
            req_duration = st.selectbox("Duration (Hours)", [1, 2, 3, 4], index=1)
            req_capacity = st.number_input("Expected Attendance", min_value=1, max_value=250, value=45)
        with col3:
            req_purpose = st.selectbox("Purpose", REQUEST_PURPOSES, index=0)
            req_dept = st.selectbox("Department / Faculty", ["All / None"] + DEPARTMENTS, index=0)

        st.markdown("**Equipment & Accessibility Constraints**")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            req_proj = st.checkbox("📽️ Projector Required", value=False)
        with f2:
            req_comp = st.checkbox("💻 Computers Required", value=False)
        with f3:
            req_acc = st.checkbox("♿ Accessibility Certified", value=False)
        with f4:
            req_room_type = st.selectbox("Preferred Room Type", ["Any"] + VALID_ROOM_TYPES, index=0)

        submit_search = st.form_submit_button("🔍 Find Available Matching Rooms", use_container_width=True)

    if submit_search:
        pref_type = None if req_room_type == "Any" else req_room_type
        eval_date_str = req_date.strftime("%Y-%m-%d")

        result = matching_engine.evaluate_request(
            request_date=eval_date_str,
            start_hour=float(req_start),
            duration_hours=float(req_duration),
            required_capacity=int(req_capacity),
            projector_required=req_proj,
            computers_required=req_comp,
            accessible_required=req_acc,
            preferred_room_type=pref_type
        )

        if not result["success"]:
            st.error(f"❌ {result['error']}")
        else:
            recs = result["recommendations"]
            rejs = result["rejected"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Feasible Rooms Found", len(recs))
            m2.metric("Disqualified Rooms", len(rejs))
            m3.metric("Evaluated Inventory", result["total_rooms_evaluated"])

            if not recs:
                st.warning(f"⚠️ No feasible rooms match all hard constraints on **{eval_date_str}** from **{req_start:02d}:00 to {req_start+req_duration:02d}:00**.")
                st.write("Inspect the disqualified room breakdown below to see the specific constraint bottlenecks.")
            else:
                st.markdown("### 🏆 Top Ranked Recommendations")
                for idx, room in enumerate(recs[:5], start=1):
                    with st.expander(f"#{idx} — Room {room['room_id']} ({room['building']} Building, Floor {room['floor']}) — Score: {room['match_score']*100:.1f}%", expanded=(idx==1)):
                        r_c1, r_c2, r_c3 = st.columns(3)
                        with r_c1:
                            st.write(f"**Capacity:** {room['capacity']} seats ({room['excess_capacity']} surplus)")
                            st.write(f"**Room Type:** {room['room_type']}")
                        with r_c2:
                            st.write(f"**Utilization Efficiency:** {room['utilization_efficiency']}%")
                            st.write(f"**Power Outlets:** {room['power_outlets']} outlets")
                        with r_c3:
                            st.write(f"**Projector:** {'Yes' if room['projector'] else 'No'}")
                            st.write(f"**Computers:** {'Yes' if room['computers'] else 'No'}")
                            st.write(f"**Accessible:** {'Yes' if room['accessible'] else 'No'}")

                        st.markdown("**Why Recommended:**")
                        for p in room["pros"]:
                            st.markdown(f"- ✅ {p}")
                        if room["trade_offs"]:
                            st.markdown("**Trade-offs:**")
                            for c in room["trade_offs"]:
                                st.markdown(f"- ⚠️ {c}")

            # Disqualified Rooms Forensics
            with st.expander(f"🔍 Disqualified Rooms Audit ({len(rejs)} rooms)", expanded=False):
                st.write("Detailed failure analysis for rejected rooms (Hard Constraint Violations):")
                rej_data = []
                for r in rejs:
                    rej_data.append({
                        "Room": r["room_id"],
                        "Building": r["building"],
                        "Type": r["room_type"],
                        "Capacity": r["capacity"],
                        "Disqualification Reasons": " | ".join(r["rejection_reasons"])
                    })
                st.dataframe(pd.DataFrame(rej_data), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 3: MULTI-EVENT OPTIMIZER
# -----------------------------------------------------------------------------
elif page == "Multi-Event Optimizer":
    st.markdown("## ⚡ Multi-Event Room Allocation Optimizer")
    st.write("""
    Solve campus-wide space allocation across multiple competing requests simultaneously.
    Contrasts mathematical **Google OR-Tools CP-SAT** optimization against a **Greedy First-Feasible** baseline under identical hard constraints.
    """)

    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        target_date = st.date_input("Target Allocation Date", value=date(2025, 9, 15), min_value=date(2025, 8, 1), max_value=date(2025, 11, 30))
    with opt_col2:
        solver_time_limit = st.slider("Solver Time Limit (Seconds)", min_value=3, max_value=30, value=10)
    with opt_col3:
        filter_status = st.multiselect("Request Status Filter", ["Waitlisted", "Approved", "Cancelled", "Rejected"], default=["Waitlisted", "Approved"])

    # Filter requests for the selected date
    target_date_str = target_date.strftime("%Y-%m-%d")
    date_reqs = reqs_df[reqs_df["request_date"] == target_date_str].copy()
    if filter_status:
        date_reqs = date_reqs[date_reqs["status"].isin(filter_status)]

    st.markdown(f"**Active Demands for {target_date_str}:** {len(date_reqs)} incoming requests")

    if st.button("🚀 Run Global Optimization Engine", type="primary", use_container_width=True):
        if date_reqs.empty:
            st.warning(f"No requests found for {target_date_str} with the selected status filters. Try another date like 2025-09-15.")
        else:
            with st.spinner("Solving global multi-commodity allocation via Google OR-Tools CP-SAT..."):
                benchmark = optimizer_engine.compare_optimizer_vs_baseline(date_reqs, time_limit_seconds=float(solver_time_limit))

            m = benchmark["metrics"]
            opt = m["optimizer"]
            base = m["baseline"]
            deltas = m["deltas"]

            # High-level comparison cards
            st.markdown("### 📊 Benchmark: Optimizer vs. Greedy Baseline")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric(
                "Fulfillment Rate",
                f"{opt['fulfillment_rate']}%",
                delta=f"{deltas['fulfillment_gain_pct']:+}% vs Greedy"
            )
            bc2.metric(
                "Requests Accommodated",
                f"{opt['assigned_count']} / {m['requests_evaluated']}",
                delta=f"{deltas['additional_requests_served']:+} extra requests"
            )
            bc3.metric(
                "Avg Excess Seats Wasted",
                f"{opt['avg_excess_seats']} seats",
                delta=f"-{base['avg_excess_seats'] - opt['avg_excess_seats']:.1f} seat waste"
            )
            bc4.metric(
                "Hard Constraint Violations",
                f"{opt['hard_constraint_violations']}",
                delta="0 (Guaranteed)"
            )

            # Detailed comparison table
            comp_df = pd.DataFrame([
                {
                    "Metric": "Algorithm / Solver",
                    "OR-Tools CP-SAT (Kairos)": opt["method"],
                    "Greedy First-Feasible Baseline": base["method"]
                },
                {
                    "Metric": "Requests Accommodated",
                    "OR-Tools CP-SAT (Kairos)": f"{opt['assigned_count']} ({opt['fulfillment_rate']}%)",
                    "Greedy First-Feasible Baseline": f"{base['assigned_count']} ({base['fulfillment_rate']}%)"
                },
                {
                    "Metric": "Total Excess Seats Wasted",
                    "OR-Tools CP-SAT (Kairos)": f"{opt['total_excess_seats']:,} seats",
                    "Greedy First-Feasible Baseline": f"{base['total_excess_seats']:,} seats"
                },
                {
                    "Metric": "Average Excess per Room",
                    "OR-Tools CP-SAT (Kairos)": f"{opt['avg_excess_seats']} seats",
                    "Greedy First-Feasible Baseline": f"{base['avg_excess_seats']} seats"
                },
                {
                    "Metric": "Execution Runtime",
                    "OR-Tools CP-SAT (Kairos)": f"{opt['runtime_seconds']} sec",
                    "Greedy First-Feasible Baseline": f"{base['runtime_seconds']} sec"
                },
                {
                    "Metric": "Hard Constraint Violations",
                    "OR-Tools CP-SAT (Kairos)": "0 (Strictly Enforced)",
                    "Greedy First-Feasible Baseline": "0 (Strictly Enforced)"
                }
            ])
            st.table(comp_df.set_index("Metric"))

            # Allocation Schedule Table
            opt_allocs = benchmark["optimizer_result"]["allocations"]
            if opt_allocs:
                st.markdown("### 📋 Proposed Room Allocations")
                alloc_df = pd.DataFrame(opt_allocs)
                display_cols = ["request_id", "room_id", "building", "room_type", "start_hour", "end_hour", "required_capacity", "allocated_room_capacity", "excess_capacity", "utilization_rate", "purpose"]
                st.dataframe(alloc_df[display_cols], use_container_width=True, hide_index=True)

                # Export allocation CSV
                csv_data = alloc_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Allocation Plan (CSV)",
                    data=csv_data,
                    file_name=f"kairos_allocations_{target_date_str}.csv",
                    mime="text/csv",
                )


# -----------------------------------------------------------------------------
# PAGE 4: CAPACITY SWAPS
# -----------------------------------------------------------------------------
elif page == "Capacity Swaps":
    st.markdown("## 🔄 Capacity-Aware Room Swaps")
    st.write("""
    Identifies zero-disruption room exchanges between concurrent scheduled classes.
    By swapping rooms between an overflowing class and an underutilized lecture hall at the **exact same time**, 
    Kairos solves severe overcrowding without altering class start times.
    """)

    swap_engine = CapacitySwapEngine(rooms_df, sched_df, events_df)
    proposals = swap_engine.detect_swaps(max_proposals=6)

    if not proposals:
        st.info("No capacity swap bottlenecks detected in the current filter.")
    else:
        st.markdown(f"**Identified {len(proposals)} Beneficial Swap Proposals:**")

        for idx, swap in enumerate(proposals, start=1):
            cA = swap["class_A"]
            cB = swap["class_B"]
            with st.container():
                st.markdown(f"""
                <div class="rec-card rec-card-swap">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#1e1b4b;">Swap Proposal #{idx}: {cA['course']} & {cB['course']} ({swap['weekday']} {swap['time_slot']})</h4>
                        <span class="badge-tag" style="background:#4338ca;">Reduces {swap['seat_mismatch_reduction']} Seats of Waste</span>
                    </div>
                    <hr style="margin:0.75rem 0;"/>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1.5rem;">
                        <div>
                            <strong>Class 1: {cA['course']} ({cA['department']})</strong><br/>
                            • Expected Attendance: <b>{cA['attendance']} students</b><br/>
                            • Current Room: <b>{cA['current_room']}</b> ({cA['current_capacity']} seats) — <span style="color:#b91c1c;">{cA['current_fit']}</span><br/>
                            • Proposed Room: <b>{cA['proposed_room']}</b> ({cA['proposed_capacity']} seats) — <span style="color:#15803d;">{cA['proposed_fit']}</span>
                        </div>
                        <div>
                            <strong>Class 2: {cB['course']} ({cB['department']})</strong><br/>
                            • Expected Attendance: <b>{cB['attendance']} students</b><br/>
                            • Current Room: <b>{cB['current_room']}</b> ({cB['current_capacity']} seats) — <span style="color:#b45309;">{cB['current_fit']}</span><br/>
                            • Proposed Room: <b>{cB['proposed_room']}</b> ({cB['proposed_capacity']} seats) — <span style="color:#15803d;">{cB['proposed_fit']}</span>
                        </div>
                    </div>
                    <div style="margin-top:0.75rem; font-size:0.85rem; color:#475569;">
                        <strong>Key Benefits:</strong> {', '.join(swap['rationales'])}<br/>
                        <strong>Trade-off:</strong> {swap['trade_offs']}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 5: UTILIZATION ANALYTICS
# -----------------------------------------------------------------------------
elif page == "Utilization Analytics":
    st.markdown("## 📈 Campus Utilization & Demand Analytics")
    st.write("""
    All metrics reflect **scheduled utilization** based on recurring class schedules and scheduled events.
    Overlapping intervals are strictly merged to avoid double-counting.
    """)

    tab_overview, tab_heatmap, tab_demand = st.tabs(["Room Utilization", "Time-Slot Heatmap", "Demand vs. Supply"])

    with tab_overview:
        weekly_util = util_engine.compute_weekly_schedule_utilization()
        if not weekly_util.empty:
            u1, u2, u3 = st.columns(3)
            u1.metric("Average Scheduled Utilization", f"{weekly_util['scheduled_utilization_rate'].mean():.1f}%")
            u2.metric("Highest Utilized Room", f"{weekly_util.sort_values(by='scheduled_utilization_rate', ascending=False).iloc[0]['room_id']} ({weekly_util['scheduled_utilization_rate'].max():.1f}%)")
            u3.metric("Campus Operating Denominator", f"{OPERATING_HOURS_PER_DAY * 6} hrs/week (Mon–Sat)")

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                top_rooms = weekly_util.sort_values(by="scheduled_utilization_rate", ascending=False).head(15)
                fig_top = px.bar(
                    top_rooms,
                    x="room_id",
                    y="scheduled_utilization_rate",
                    color="scheduled_utilization_rate",
                    title="Top 15 Most Utilized Rooms (% Weekly Scheduled Hours)",
                    color_continuous_scale="Teal",
                    labels={"scheduled_utilization_rate": "Utilization %", "room_id": "Room"}
                )
                fig_top.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_top, use_container_width=True)

            with col_chart2:
                fig_bldg = px.box(
                    weekly_util,
                    x="building",
                    y="scheduled_utilization_rate",
                    color="building",
                    title="Scheduled Utilization Variance by Building",
                    labels={"scheduled_utilization_rate": "Utilization %", "building": "Building"}
                )
                fig_bldg.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bldg, use_container_width=True)

            st.markdown("### 📋 Full Room Schedule Audit")
            st.dataframe(weekly_util[["room_id", "building", "room_type", "capacity", "weekly_scheduled_hours", "scheduled_utilization_rate"]], use_container_width=True, hide_index=True)

    with tab_heatmap:
        heatmap_data = util_engine.compute_hourly_heatmap_data()
        if not heatmap_data.empty:
            pivot_df = heatmap_data.pivot(index="hour", columns="weekday", values="occupancy_rate_pct")
            # Order columns Mon-Sat
            ordered_cols = [c for c in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] if c in pivot_df.columns]
            pivot_df = pivot_df[ordered_cols]

            fig_hm = px.imshow(
                pivot_df,
                labels=dict(x="Day of Week", y="Hour of Day", color="Occupied Rooms %"),
                x=ordered_cols,
                y=[f"{h:02d}:00" for h in pivot_df.index],
                color_continuous_scale="YlGnBu",
                title="Campus Scheduled Occupancy Heatmap (% Rooms Booked by Hour)"
            )
            fig_hm.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hm, use_container_width=True)

    with tab_demand:
        dvs = util_engine.compute_demand_vs_supply(reqs_df)
        if "data" in dvs and not dvs["data"].empty:
            dvs_df = dvs["data"]
            fig_dvs = go.Figure(data=[
                go.Bar(name='Physical Room Supply', x=dvs_df['capacity_band'], y=dvs_df['room_supply_count'], marker_color='#0f766e'),
                go.Bar(name='Incoming Student Requests', x=dvs_df['capacity_band'], y=dvs_df['request_demand_count'], marker_color='#0284c7')
            ])
            fig_dvs.update_layout(
                barmode='group',
                title="Demand (Student Requests) vs. Supply (Physical Rooms) by Capacity Band",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_dvs, use_container_width=True)

            st.dataframe(dvs_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 6: ADMIN DECISION SUPPORT (RECOMMENDATIONS & PERSISTENCE)
# -----------------------------------------------------------------------------
elif page == "Admin Decision Support":
    st.markdown("## 📋 Administrator Decision Support")
    st.write("""
    Actionable recommendations derived from schedule bottlenecks and space mismatch analysis.
    Decisions persist across application reloads using local SQLite database storage.
    """)

    admin_recs = admin_engine.generate_recommendations()

    # Reset button for demo
    col_t1, col_t2 = st.columns([4, 1])
    with col_t2:
        if st.button("🗑️ Reset All Decisions", use_container_width=True):
            decision_store.reset_all_decisions()
            st.success("Admin decisions reset.")
            st.rerun()

    for rec in admin_recs:
        rec_id = rec["id"]
        current_status = rec["status"]

        with st.container():
            st.markdown(f"""
            <div class="rec-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#0f172a;">{rec['title']}</h4>
                    <span class="badge-{current_status.lower()}">{current_status}</span>
                </div>
                <p style="margin:0.5rem 0 0.25rem 0; color:#334155;"><strong>Issue:</strong> {rec['issue_detected']}</p>
                <p style="margin:0.25rem 0; color:#475569; font-size:0.9rem;"><strong>Evidence:</strong> {rec['evidence']}</p>
                <p style="margin:0.25rem 0; color:#0f766e; font-size:0.9rem;"><strong>Expected Benefit:</strong> {rec['expected_benefit']}</p>
                <p style="margin:0.25rem 0; color:#991b1b; font-size:0.9rem;"><strong>Potential Downside:</strong> {rec['potential_downside']}</p>
                <p style="margin:0.25rem 0; color:#1e293b; font-size:0.9rem;"><strong>Action:</strong> {rec['suggested_action']}</p>
            </div>
            """, unsafe_allow_html=True)

            btn_c1, btn_c2, _ = st.columns([1, 1, 4])
            with btn_c1:
                if st.button(f"✅ Approve", key=f"app_{rec_id}", disabled=(current_status=="Approved")):
                    decision_store.record_decision(rec_id, rec["title"], rec["category"], "Approved")
                    st.rerun()
            with btn_c2:
                if st.button(f"❌ Reject", key=f"rej_{rec_id}", disabled=(current_status=="Rejected")):
                    decision_store.record_decision(rec_id, rec["title"], rec["category"], "Rejected")
                    st.rerun()


# -----------------------------------------------------------------------------
# PAGE 7: DATA QUALITY & SOURCES
# -----------------------------------------------------------------------------
elif page == "Data Quality & Sources":
    st.markdown("## 🔍 Data Quality & Source Auditing")
    st.write("Inspect active data provenance, schema invariants, missing values, and referential integrity.")

    st.markdown(f"**Active Source:** `{current_source}`")

    # Upload replacement CSVs
    with st.expander("📤 Upload Custom / Replacement CSV Datasets", expanded=False):
        st.write("Upload replacement CSVs to audit or test custom campus layouts:")
        u_rooms = st.file_uploader("Upload rooms.csv", type=["csv"], key="u_rooms")
        u_sched = st.file_uploader("Upload class_schedule.csv", type=["csv"], key="u_sched")
        u_events = st.file_uploader("Upload events.csv", type=["csv"], key="u_events")
        u_reqs = st.file_uploader("Upload room_requests.csv", type=["csv"], key="u_reqs")

        if st.button("Apply Uploaded CSVs", use_container_width=True):
            buffers = {}
            if u_rooms: buffers["rooms"] = u_rooms
            if u_sched: buffers["class_schedule"] = u_sched
            if u_events: buffers["events"] = u_events
            if u_reqs: buffers["room_requests"] = u_reqs

            if buffers:
                loader = DataLoader()
                custom_dfs, s_name = loader.load_all(custom_buffers=buffers)
                st.session_state["custom_data"] = custom_dfs
                st.success(f"Loaded custom datasets from buffer. Source: {s_name}")
                st.rerun()

    # Data Quality Report
    st.markdown("### 📋 Automated Data Quality Audit")
    q_s1, q_s2, q_s3, q_s4 = st.columns(4)
    q_s1.metric("Ingested Records", f"{dq_report['summary']['total_records_ingested']:,}")
    q_s2.metric("Missing Fields", f"{dq_report['summary']['total_missing_fields']}")
    q_s3.metric("Duplicate Identifiers", f"{dq_report['summary']['total_duplicate_ids']}")
    q_s4.metric("Referential Integrity", "Passed ✅" if dq_report['referential_integrity']['integrity_passed'] else "Orphans ❌")

    # Table per dataset
    st.markdown("#### Dataset Invariant Summary")
    audit_summary = []
    for k, v in dq_report["dataset_audits"].items():
        audit_summary.append({
            "Dataset": k,
            "Row Count": f"{v['row_count']:,}",
            "Column Count": v['column_count'],
            "Null Values": v['null_values_count'],
            "Duplicate IDs": v['duplicate_identifiers_count'],
            "Missing Required Columns": ", ".join(v['missing_expected_columns']) or "None"
        })
    st.dataframe(pd.DataFrame(audit_summary), use_container_width=True, hide_index=True)

    with st.expander("🔍 Referential Integrity & Temporal Checks Details"):
        st.json(dq_report["referential_integrity"])
        st.json(dq_report["temporal_audits"])
        st.json(dq_report["capacity_audits"])
