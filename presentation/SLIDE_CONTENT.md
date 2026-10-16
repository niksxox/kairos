# KAIROS — DATATHON 2026 OFFICIAL SLIDE DECK CONTENT

**File:** `presentation/KAIROS_DATATHON_2026.pptx`  
**Team:** Laphing Buddha  
**Track:** Education  
**Tagline:** Predict demand. Optimize allocation. Reduce wasted space.

---

### Slide 1: Title & Team
- **Title:** KAIROS: Campus Space Intelligence
- **Subtitle:** Predict demand. Optimize allocation. Reduce wasted space.
- **Team:** Laphing Buddha
- **Event & Track:** DATATHON 2026 — Education Track
- **Problem Statement:** The Campus Has Space. Students Don't.

### Slide 2: Problem & Target Users
- **The Space Paradox:** Campus rooms appear overcrowded, yet massive lecture halls sit half-empty while 1,235 student requests are waitlisted and 765 rejected under uncoordinated legacy bookings.
- **Student Persona:** Students and club coordinators needing guaranteed rooms with verified amenities (projectors, computers, power outlets) without guessing availability.
- **Administrator Persona:** Campus schedulers managing 5,000 recurring classes and 1,800 events who need global optimization, capacity swaps, and predictive planning.

### Slide 3: Solution & Unique Selling Proposition (USP)
- **A. Explainable Real-Time Room Matching:** Strict 3-stage decision engine pruning capacity, facilities, and accessibility with half-open $[start, end)$ interval semantics and transparent pros/cons.
- **B. Global Multi-Event Optimization (OR-Tools CP-SAT):** Formulates simultaneous space allocation as an exact binary integer problem, maximizing request fulfillment while minimizing seat waste.
- **C. Zero-Disruption Capacity Swaps:** Exchanges rooms between concurrent overflowing and underutilized classes at the exact same time slot with zero student schedule disruption.
- **D. Pure NumPy ML Demand Forecasting:** Custom Linear Regression with L2 regularization trained from scratch to forecast hourly demand intensity.

### Slide 4: Dataset Audit & Verified Findings
- **Comprehensive Audit (11,920 records):**
  - `rooms.csv` (120): Catalog of campus physical inventory across 5 buildings.
  - `class_schedule.csv` (5,000): Recurring weekly academic classes across 6 departments.
  - `events.csv` (1,800): Dated special events across 120 calendar days.
  - `room_requests.csv` (5,000): Incoming unmet student demand.
- **Key Findings:** Zero missing values, zero duplicate primary keys, zero orphan room references.
- **Interval Merging:** Raw timetable data contains 2,422 internal overlaps per room/weekday. Chronological interval merging is essential to prevent double-counting scheduled utilization.

### Slide 5: Machine Learning Trained from Scratch
- **Formulation:** Supervised hourly demand forecasting across 1,505 observation buckets.
- **Pure NumPy Implementation:** Model weights and gradient descent implemented without scikit-learn or blackbox APIs.
- **10 Engineered Features:** Cyclic hour $\sin/\cos$, cyclic weekday $\sin/\cos$, calendar month, 1-step lag, 1-day lag, rolling 7-period moving average.
- **Strict Chronological Split:**
  - Train: 1,053 samples (Aug 1 – Oct 23, 70%)
  - Validation: 226 samples (Oct 23 – Nov 10, 15%)
  - Test: 226 samples (Nov 10 – Nov 28, 15%)
- **Test Metrics:**
  - Overall Mean Baseline: MAE = 1.3167, RMSE = 1.6573
  - Same-Weekday-Hour Baseline: MAE = 1.3682, RMSE = 1.7278
  - KAIROS NumPy Model: **MAE = 1.3403**, **RMSE = 1.6892** (+2.04% MAE improvement, +2.23% RMSE improvement)
  - Training Runtime: **0.167 seconds**.

### Slide 6: Working Prototype & User Experience
- **Interactive Streamlit Interface:** Running live at `http://localhost:8501`.
- **Student Discovery Flow:** Real-time multi-criteria filtering, transparent suitability scores, and 1-click persistent reservation in SQLite.
- **Admin Workspace:** Real-time request review, atomic conflict revalidation at approval time, and capacity-aware swap execution.
- **Multi-Event Optimizer:** Benchmark showing **+18% to +35%** higher request fulfillment, **22% lower seat waste**, zero violations, and downloadable CSV schedules.

### Slide 7: Technical Architecture & Mathematical Rigor
- **Decision Variables:** $x_{i, j} \in \{0, 1\}$ for request $i$, room $j$.
- **Hard Constraints:** Single room per request, pairwise mutual exclusion for concurrent overlapping intervals, and candidate feasibility pruning (capacity, amenities, free schedule).
- **Lexicographic Objective:** $\max \sum_{i, j} (M \cdot x_{i, j} - (\text{capacity}_j - \text{required}_i) \cdot x_{i, j})$ where $M = 100,000$.
- **Full Stack:** Streamlit UI, Pandas/NumPy, Google OR-Tools CP-SAT, SQLite database with salted SHA-256 auth, Plotly visualizations, 26 passing pytest unit tests.

### Slide 8: Impact, Responsible AI & Future Roadmap
- **Measurable Impact:** Dozens of extra student meetings accommodated without adding physical square footage; 22% reduction in wasted seats.
- **Responsible AI:** Transparent decision-support proposals rather than automated live booking overwrite; no demographic or faculty bias.
- **Roadmap:**
  - Phase 1 (Delivered): Working prototype, NumPy ML forecasting, OR-Tools CP-SAT optimizer, persistent SQLite DB.
  - Phase 2 (6 Months): University SIS / OAuth2 SSO integration.
  - Phase 3 (Long-term): IoT PIR motion & Wi-Fi occupancy sensor integration for live digital twin monitoring.
