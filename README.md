# KAIROS 2.0 — Campus Space Intelligence Platform

**Predict demand. Optimize allocation. Reduce wasted space.**  
An explainable campus space decision-support system, priority-weighted multi-event room allocation engine, bounded swap-chain reallocation service, pure NumPy machine learning demand forecasting pipeline, and FastAPI REST engine.

- **Product Name:** KAIROS
- **Team Name:** Laphing Buddha
- **Hackathon:** DATATHON 2026
- **Track:** Education — The Campus Has Space. Students Don't.
- **GitHub Repository:** [https://github.com/niksxox/kairos](https://github.com/niksxox/kairos)
- **Live Public URL:** [https://immune-subsidiary-clips-draw.trycloudflare.com](https://immune-subsidiary-clips-draw.trycloudflare.com)
- **1-Click Cloud Deploy (Streamlit Community Cloud):** [![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=niksxox/kairos&branch=main&mainModule=app.py)

### Demo Credentials
- **Student Portal:** Username: `student` | Password: `student123`
- **Administrator Portal:** Username: `admin` | Password: `admin123`

---

## 1. What's New in KAIROS 2.0

1. **Institutional Priority & Importance Metric (`src/priority.py`):**
   - Formal assessments (Exams, Midterms, Tests: Weight 100) are strictly prioritized over discretionary social gatherings (Cultural Fests: Weight 5).
   - In Google OR-Tools CP-SAT, priority weighting mathematically guarantees that high-stakes academic events are satisfied first under space constraints.
2. **Automatic Reallocation & Bounded Swap-Chain Engine (`src/reallocation_engine.py`):**
   - Bounded graph search detecting mutual swaps between concurrent classes and swap chains ($Q \to \text{Room}_1 \to \text{Room}_2$) with atomic all-or-nothing database transactions.
3. **Transactional Data Ingestion & REST API (`src/ingestion.py`, `src/api.py`):**
   - High-performance FastAPI application with interactive OpenAPI/Swagger UI at `http://localhost:8000/docs`.
   - Enforces schema validation, referential integrity verification against physical rooms, value bounds checking, and audit logging.
4. **Pure NumPy Machine Learning Evaluator (`scripts/evaluate_model.py`):**
   - Standalone CLI evaluator verifying the custom NumPy Linear Regression model on a held-out chronological test set (Test MAE: 1.3403 vs Stratified Baseline: 1.3682).
5. **Clean Academic SaaS Design System (Zero AI-Slope):**
   - Redesigned with restrained typography, clean borders, crisp cards, functional day/week timetable, and priority classification badges.
6. **35/35 Passing Automated Tests (`tests/`):**
   - Comprehensive unit and integration coverage for booking lifecycle, priority ordering, swap chains, ingestion schema verification, and REST endpoints.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                              PRESENTATION LAYER                         |
|  Streamlit Multipage Web App (Clean Academic UI / No AI-Slope Gradients)|
|  • Student Discovery & Booking Portal     • Calendar Timetable          |
|  • Admin Workspace & Collision Approvals  • Reallocation Workbench      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                             API & SERVICES                              |
|  FastAPI REST Engine (OpenAPI /docs) & Ingestion Validation Service     |
|  • /api/health                            • /api/reallocation/*         |
|  • /api/data/import                       • /api/bookings/*             |
+------------------+-----------------+------------------+-----------------+
                   |                 |                  |
                   v                 v                  v
+----------------------+  +---------------------+  +----------------------+
|  OPTIMIZATION LAYER  |  |   PERSISTENCE LAYER |  |     ML PIPELINE      |
| • OR-Tools CP-SAT    |  | • SQLite DB         |  | • Pure NumPy LinReg  |
| • Priority Objective |  | • Passwords (SHA256)|  | • Chrono Split (70%) |
| • Bounded Swap Chain |  | • Realloc Proposals |  | • Cyclic Time Feats  |
| • Disruption Penalty |  | • Audit Logs        |  | • Inference Service  |
+----------------------+  +---------------------+  +----------------------+
```

---

## 3. Dataset Specifications

| Dataset | Verified Rows | Columns | Primary Key / Join Key |
| :--- | :---: | :--- | :--- |
| `rooms.csv` | **120** | `room_id`, `building`, `floor`, `room_type`, `capacity`, `projector`, `computers`, `accessible`, `power_outlets` | `room_id` |
| `class_schedule.csv` | **5,000** | `schedule_id`, `room_id`, `course_code`, `weekday`, `start_hour`, `duration_hours`, `expected_attendance`, `department` | `schedule_id` (joins on `room_id`) |
| `events.csv` | **1,800** | `event_id`, `room_id`, `event_type`, `start_time`, `duration_hours`, `expected_attendance`, `actual_attendance` | `event_id` (joins on `room_id`) |
| `room_requests.csv` | **5,000** | `request_id`, `request_date`, `start_hour`, `duration_hours`, `required_capacity`, `projector_required`, `computers_required`, `purpose`, `status` | `request_id` (Unallocated demand) |

---

## 4. Quick Start & Execution

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run the FastAPI REST Engine (Port 8000)
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation: `http://localhost:8000/docs`

### 3. Run the Streamlit Web Application (Port 8501)
```powershell
streamlit run app.py --server.headless true --server.port 8501
```

### 4. Run Machine Learning Training & Evaluation CLI
```powershell
python scripts/train_model.py
python scripts/evaluate_model.py
```

### 5. Run Automated Test Suite
```powershell
python -m pytest -v
```

---

## 5. Documentation Directory

- [Implementation Audit & Repair Plan](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/docs/IMPLEMENTATION_AUDIT.md)
- [System Architecture](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/docs/ARCHITECTURE.md)
- [REST API Specification](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/docs/API.md)
- [Reallocation & Swap Engine](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/docs/SWAP_ENGINE.md)
- [Machine Learning Methodology](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/docs/ML_METHODOLOGY.md)
- [End-to-End Evaluation & Demo Guide](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/docs/DEMO_GUIDE.md)
