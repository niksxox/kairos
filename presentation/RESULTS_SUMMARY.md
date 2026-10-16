# KAIROS — OFFICIAL BENCHMARK & EXPERIMENTAL RESULTS SUMMARY
**Team:** Laphing Buddha | **Product:** Kairos | **DATATHON 2026**

---

## 1. Multi-Event Optimization Benchmark Results

Evaluated on official DATATHON 2026 test subsets under identical constraints:

| Evaluation Metric | Google OR-Tools CP-SAT (KAIROS) | Greedy First-Feasible Baseline | Improvement / Impact |
| :--- | :---: | :---: | :---: |
| **Requests Accommodated** | **Optimal Maximum** | Suboptimal Sequential | **+18% to +35% More Requests Served** |
| **Average Seat Waste per Room** | **Minimizes Excess Seats** | Uncontrolled Fit | **22.4% Lower Excess Seat Waste** |
| **Hard Constraint Violations** | **0 (Strict Guarantee)** | **0 (Strict Guarantee)** | Zero Collisions |
| **Capacity Constraints Satisfied** | **100%** | 100% | Zero Overcrowding |
| **Specialist Facilities Matched** | **100%** | 100% | Verified Hardware |
| **Mutual Room Exclusions** | **Enforced Globally** | Enforced Greedily | Complete Collisions Prevention |
| **Solver Runtime** | **< 1.5 seconds** | < 0.1 seconds | Sub-second Execution |

---

## 2. Machine Learning Training & Evaluation Metrics

- **Model Type:** Linear Regression with Batch Gradient Descent & $L_2$ Regularization
- **Implementation:** **100% Pure NumPy trained from scratch** (Zero sklearn or external ML libraries)
- **Dataset Granularity:** 1,505 hourly observation buckets across 120 days (Aug 1 – Nov 28, 2025)
- **Split Protocol:** Chronological 70% Train (1,053), 15% Validation (226), 15% Test (226)

| Model / Baseline | Test MAE | Test RMSE | Training Runtime | Parameter Count |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Mean Baseline** | 1.3167 reqs | 1.6573 reqs | — | 1 parameter (mean) |
| **Same-Weekday-Hour Baseline** | 1.3682 reqs | 1.7278 reqs | — | 91 cell lookup |
| **KAIROS NumPy Linear Model** | **1.3403 reqs** | **1.6892 reqs** | **0.167 seconds** | **11 parameters (10 weights + bias)** |

### Top Predictive Feature Weights
1. `sin_hour` (Cyclic time of day): $-0.2418$ (Peak afternoon demand concentration)
2. `lag_1` (Immediate preceding demand): $+0.1852$ (Temporal continuity of student booking activity)
3. `rolling_mean_7` (7-step moving average): $+0.1420$ (Medium-term daily momentum)
4. `weekday_num` (Day of week index): $-0.0982$ (Monday–Thursday heavier than Friday/Saturday)

---

## 3. Data Integrity & Validation Verification

- **Total Records Ingested:** 11,920 records
- **Total Missing Values:** 0
- **Duplicate Primary Keys:** 0
- **Orphan Room References:** 0 (100% of class schedules and events resolve to existing rooms)
- **Automated Tests:** 26 passing pytest tests in 1.47 seconds
