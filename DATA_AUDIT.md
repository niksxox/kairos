# KAIROS — DATA AUDIT & SPECIFICATION REPORT
**Event:** DATATHON 2026 | **Track:** Education — The Campus Has Space. Students Don't.  
**Team:** Laphing Buddha | **Product:** Kairos

---

## 1. Dataset Inventory & Discovered Schema

All four synthetic campus datasets were thoroughly ingested and audited using our automated validation pipeline (`scripts/validate_data.py`).

| Dataset Name | Filename | Records | Columns | Null Values | Duplicate IDs | Orphan Keys |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rooms Catalog** | `rooms.csv` | 120 | 9 | 0 | 0 | Master |
| **Class Schedule** | `class_schedule.csv` | 5,000 | 8 | 0 | 0 | 0 (All room_ids match) |
| **Special Events** | `events.csv` | 1,800 | 7 | 0 | 0 | 0 (All room_ids match) |
| **Room Requests** | `room_requests.csv` | 5,000 | 9 | 0 | 0 | None (Demand table) |
| **Total Ingested** | — | **11,920** | — | **0** | **0** | **0** |

---

## 2. In-Depth Dataset Profiling & Column Descriptions

### A. Rooms Catalog (`rooms.csv` — 120 rows)
- `room_id` (string, e.g. R001–R120): Unique room identifier.
- `building` (categorical): 5 buildings across campus (`Central`: 24, `West`: 24, `North`: 24, `East`: 24, `South`: 24).
- `floor` (integer): 0 (ground) through 5.
- `room_type` (categorical): `Classroom` (45), `Lab` (25), `Seminar` (20), `Study room` (20), `Auditorium` (10).
- `capacity` (integer): Range 15 to 179 seats. Mean: 89.2 seats. Total campus capacity: 10,704 seats.
- `projector` (binary 0/1): 72 rooms equipped (60.0%).
- `computers` (binary 0/1): 38 rooms equipped (31.7%).
- `accessible` (binary 0/1): 96 rooms certified ADA compliant (80.0%).
- `power_outlets` (integer): Range 2 to 30 charging outlets.

### B. Class Schedule (`class_schedule.csv` — 5,000 rows)
- `schedule_id` (string, e.g. S00001–S05000): Unique class timetable record.
- `room_id` (string): Maps to `rooms.room_id`. Zero orphan references found.
- `course_code` (string): Academic course identifier (CS001–CS060).
- `weekday` (categorical): `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`.
- `start_hour` (integer): Range 8 (08:00) to 17 (17:00).
- `duration_hours` (integer): 1, 2, or 3 hours.
- `expected_attendance` (integer): Range 8 to 149 students. Mean: 78.4 students.
- `department` (categorical): `CSE`, `ECE`, `EEE`, `Management`, `Mechanical`, `Sciences`.

### C. Special Events (`events.csv` — 1,800 rows)
- `event_id` (string, e.g. EV00001–EV01800): Unique event identifier.
- `room_id` (string): Maps to `rooms.room_id`. Zero orphan references found.
- `event_type` (categorical): `Guest lecture`, `Seminar`, `Club meeting`, `Exam`, `Workshop`.
- `start_time` (timestamp): Range `2025-08-01 00:00:00` to `2025-11-28 23:00:00` (120 days).
- `duration_hours` (integer): 1, 2, 3, or 4 hours.
- `expected_attendance` (integer): Range 5 to 149 attendees.
- `actual_attendance` (integer): Recorded post-event attendance (range 5 to 144).

### D. Room Requests (`room_requests.csv` — 5,000 rows)
- `request_id` (string, e.g. Q00001–Q05000): Unique student/club room request.
- `request_date` (date string): Range `2025-08-01` to `2025-11-28`. 120 unique calendar dates.
- `start_hour` (integer): 8 (08:00) to 20 (20:00). 13 distinct operating hours.
- `duration_hours` (integer): 1, 2, 3, or 4 hours.
- `required_capacity` (integer): Range 5 to 119 attendees. Mean: 62.4 attendees.
- `projector_required` (binary 0/1): 2,520 requests (50.4%).
- `computers_required` (binary 0/1): 1,012 requests (20.2%).
- `purpose` (categorical): `Study group`, `Class`, `Workshop`, `Club meeting`, `Project work`.
- `status` (categorical): `Approved` (2,185), `Waitlisted` (1,235), `Cancelled` (815), `Rejected` (765).

---

## 3. Relational Joins & Integrity Verification

1. **Room References:** All `room_id` values in `class_schedule.csv` (5,000) and `events.csv` (1,800) cleanly resolve against `rooms.csv` (120). There are **0 orphan records**.
2. **Independent Demand Invariant:** `room_requests.csv` does not contain `room_id`. It represents unallocated student demand. We maintain strict separation: requests are never joined to rooms as if they were preassigned.
3. **Temporal Mapping:**
   - A request on date $D$ at hour $H$ conflicts with:
     - All `class_schedule` rows where `room_id == r`, `weekday == D.weekday()`, and $[H, H+\text{dur})$ overlaps $[\text{start\_hour}, \text{start\_hour}+\text{dur})$.
     - All `events` rows where `room_id == r`, `start_time.date() == D`, and intervals overlap.

---

## 4. Machine Learning Viability & Formulation Justification

### Is Demand Forecasting Viable?
**Yes.** The dataset provides:
- **120 continuous calendar days** (August 1, 2025 to November 28, 2025).
- **1,505 distinct hourly time buckets** across operating hours (08:00 to 20:00).
- Consistent recurring weekly and hourly demand variations (mean: 3.32 requests/hour, std: 1.72, range: 1 to 12).
- Zero future data leakage: aggregated strictly by chronological order.

### Selected Task & Objective
- **Supervised Task:** Hourly student room demand forecasting.
- **Target Variable ($y$):** Number of room requests submitted for a given campus time slot (Date, Hour).
- **Model Architecture:** **Linear Regression with Gradient Descent trained from scratch using pure NumPy** (no scikit-learn, no pre-trained weights).
- **Features ($X$):**
  1. `start_hour` (normalized in $[0, 1]$).
  2. Cyclic hour encoding: $\sin(2\pi H/24)$ and $\cos(2\pi H/24)$.
  3. Cyclic weekday encoding: $\sin(2\pi W/7)$ and $\cos(2\pi W/7)$.
  4. Day of week one-hot encoding (Mon–Sun).
  5. Month feature (normalized August–November).
  6. Historical 7-day lag demand (same hour 1 week prior).
  7. Rolling 3-day average hourly demand.
- **Train/Val/Test Split:** Strict chronological split:
  - **Train:** August 1, 2025 – October 15, 2025 (70%)
  - **Validation:** October 16, 2025 – November 5, 2025 (15%)
  - **Test:** November 6, 2025 – November 28, 2025 (15%)
- **Baseline Comparison:** Historical Same-Weekday-Hour Mean Baseline.
- **Evaluation Metric:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
