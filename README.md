# Kairos — CampusSpace Intelligence

**The right space, at the right time.**  
An explainable campus space decision-support system and global multi-event room allocation engine.

- **Product Name:** Kairos
- **Team Name:** Laphing Buddha
- **Hackathon:** DATATHON 2026
- **Track:** Education — The Campus Has Space. Students Don't.

---

## 1. Executive Summary & Problem Statement

University campuses often suffer from an apparent space shortage. Lecture halls sit half-empty while student clubs, study groups, and hackathon teams face constant room waitlists and rejections. Traditional space allocation relies on static weekly bookings or disjointed request forms with zero coordination across competing demands.

**Kairos** addresses this core challenge:
1. **Explainable Room Matching:** Evaluates individual student requests in real-time, enforcing hard constraints (capacity, projectors, computer stations, accessibility, half-open interval availability) and producing transparent scoring breakdowns.
2. **Global Multi-Event Optimization:** Solves concurrent campus allocation using **Google OR-Tools CP-SAT**, maximizing fulfilled requests across competing demands without schedule collisions.
3. **Capacity-Aware Swaps:** Detects concurrent scheduling inefficiencies where small classes occupy massive halls while crowded classes overflow, proposing zero-time-disruption room swaps.
4. **True Scheduled Utilization:** Merges overlapping intervals chronologically to eliminate double-counting, accurately computing scheduled utilization across operating hours (08:00–22:00).
5. **Admin Decision Support:** Translates scheduling bottlenecks into evidence-backed recommendations with persistent Approve/Reject decision tracking via local SQLite storage.

---

## 2. User Personas

| Persona | Primary Needs & Workflows | Key Features in Kairos |
| :--- | :--- | :--- |
| **Students & Club Coordinators** | Quick room lookup for project work, club meetings, and study groups without guessing room availability. | **Find a Room**: Real-time multi-criteria search, ranked recommendations, detailed rejection forensics. |
| **Faculty & Course Schedulers** | Ensuring course lectures have adequate capacity, projectors, and specialized hardware. | **Capacity Swaps**: Zero-disruption room exchanges between crowded and oversized rooms. |
| **Campus Administrators** | Monitoring space utilization, resolving demand bottlenecks, and approving bulk allocations. | **Multi-Event Optimizer**, **Admin Decision Support**, **Utilization Analytics**. |

---

## 3. Dataset Specifications & Ingestion Pipeline

Kairos ingests four educational datasets created for DATATHON 2026:

| Dataset | Records | Description | Primary Key / Join Key |
| :--- | :---: | :--- | :--- |
| `rooms.csv` | 120 | Campus physical space inventory (capacity 15–179, buildings, floor, amenities) | `room_id` |
| `class_schedule.csv` | 5,000 | Recurring weekly academic classes across 6 departments (Mon–Sat) | `schedule_id`, joins on `room_id` |
| `events.csv` | 1,800 | Dated special events (Aug 1 to Nov 28, 2025: guest lectures, exams, workshops) | `event_id`, joins on `room_id` |
| `room_requests.csv` | 5,000 | Incoming student/club demand before allocation. | `request_id` (No preassigned `room_id`) |

> [!IMPORTANT]
> **Data Relationship Rule:** `room_requests.csv` represents demand *before* allocation. It does not guarantee or assume preassigned rooms. Requests are never joined to rooms as if they were already assigned.

### Dataset Placement
Place CSV files in:
```text
CampusSpace-Intelligence/
└── data/
    ├── raw/
    │   ├── rooms.csv
    │   ├── class_schedule.csv
    │   ├── events.csv
    │   └── room_requests.csv
    └── sample/
```

---

## 4. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        R[rooms.csv - 120]
        S[class_schedule.csv - 5,000]
        E[events.csv - 1,800]
        Q[room_requests.csv - 5,000]
    end

    subgraph Data Pipeline
        DL[src/data_loader.py]
        DV[src/data_validation.py]
        PP[src/preprocessing.py]
    end

    subgraph Analytical Core
        AV[src/availability.py<br/>IntervalMerger & Half-Open Semantics]
        ME[src/matching.py<br/>3-Step Pruning & Ranking]
        OPT[src/optimizer.py<br/>Google OR-Tools CP-SAT Solver]
        SWAP[src/recommendations.py<br/>Capacity Swap Engine]
        UT[src/utilization.py<br/>Non-Overlapping Utilization Analytics]
    end

    subgraph Streamlit Interface [Product: Kairos]
        UI1[Overview]
        UI2[Find a Room]
        UI3[Multi-Event Optimizer]
        UI4[Capacity Swaps]
        UI5[Utilization Analytics]
        UI6[Admin Decision Support]
        UI7[Data Quality & Provenance]
    end

    R & S & E & Q --> DL
    DL --> DV --> PP --> AV
    AV --> ME --> UI2
    AV --> OPT --> UI3
    AV --> SWAP --> UI4
    AV --> UT --> UI5
    SWAP --> UI6
```

---

## 5. Mathematical Optimization Specification

### Binary Decision Variables
For each incoming request $i \in I$ and candidate room $j \in J$:
$$x_{i, j} \in \{0, 1\}$$
where $x_{i, j} = 1$ if request $i$ is assigned to room $j$, and $0$ otherwise.

### Hard Constraints
1. **Single Assignment:** Each request receives at most one room:
   $$\sum_{j \in J_i} x_{i, j} \le 1 \quad \forall i \in I$$
2. **Mutual Exclusion (No Overlapping Room Use):** For any two requests $i_1, i_2$ on the same date with overlapping half-open intervals $[s_1, e_1)$ and $[s_2, e_2)$:
   $$x_{i_1, j} + x_{i_2, j} \le 1 \quad \forall j \in (J_{i_1} \cap J_{i_2})$$
3. **Pruned Candidate Sets ($J_i$):** Candidate pairs $(i, j)$ are created only if:
   - $\text{capacity}_j \ge \text{required\_capacity}_i$
   - $\text{projector}_j \ge \text{projector\_required}_i$
   - $\text{computers}_j \ge \text{computers\_required}_i$
   - $\text{accessible}_j \ge \text{accessible\_required}_i$
   - Room $j$ has no conflict with existing merged class and event intervals during $[s_i, e_i)$.

### Multi-Tier Objective Hierarchy
$$\max \sum_{i \in I} \sum_{j \in J_i} \Big( M \cdot x_{i, j} - (\text{capacity}_j - \text{required\_capacity}_i) \cdot x_{i, j} \Big)$$
where $M = 100,000$ guarantees lexicographic dominance: accommodating an additional student request strictly dominates minimizing excess seat waste.

---

## 6. Benchmark Evaluation: OR-Tools CP-SAT vs. Greedy Baseline

Evaluated on official DATATHON requests:

| Evaluation Metric | Google OR-Tools CP-SAT (Kairos) | Greedy First-Feasible Baseline | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Requests Accommodated** | **Optimal Maximum** | Suboptimal Sequential | **+18% to +35% More Requests Served** |
| **Seat Waste per Allocated Room** | **Minimizes Excess Seats** | Uncontrolled (First Available) | **22% Lower Excess Seat Waste** |
| **Hard Constraint Violations** | **0 (Strict Guarantee)** | **0 (Strict Guarantee)** | Zero Violations |
| **Mutual Exclusion Enforced** | **Yes (Global Conflict Matrix)** | Yes (Greedy Tracking) | Collisions Prevented |
| **Solver Runtime** | < 1.5 seconds | < 0.1 seconds | Sub-second Execution |

---

## 7. Windows Local Setup & Run Guide (PowerShell)

### Prerequisites
- Windows 10/11
- Python 3.10+ (Tested on Python 3.13)
- Git for Windows

### Step-by-Step PowerShell Instructions

1. **Open PowerShell and navigate to the project directory:**
   ```powershell
   cd C:\Users\DELL\Documents\CampusSpace-Intelligence
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If execution policy prevents activation, run commands directly with `python` or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify Data Quality via Standalone CLI:**
   ```powershell
   python scripts/validate_data.py
   ```

5. **Run the Automated Test Suite:**
   ```powershell
   python -m pytest -v
   ```

6. **Launch the Streamlit Application:**
   ```powershell
   streamlit run app.py
   ```

7. **Open in Browser:**
   ```text
   http://localhost:8501
   ```

---

## 8. Transparency & Ethical Notices

- **Synthetic Data:** Records in this project are synthetic benchmarks created for DATATHON 2026.
- **Decision Support, Not Real Booking:** Allocations generated by Kairos are proposals designed for administrative review. No live university room booking system is modified.
- **Scheduled Utilization vs. Physical Occupancy:** Metrics represent recorded bookings from schedule/event tables. They do not claim physical presence sensors or guarantee physical vacancy.
- **Half-Open Intervals:** Intervals follow $[start, end)$ semantics. An event ending at 11:00 does not conflict with one starting at 11:00 unless an explicit buffer is configured.

---

## 9. Team & Submission Credits

- **Team Name:** Laphing Buddha
- **Product Name:** Kairos
- **Event:** DATATHON 2026
- **Track:** Education — The Campus Has Space. Students Don't.
