# KAIROS 2.0 — System Architecture

**Product:** KAIROS  
**Subtitle:** Campus Space Intelligence  
**Tagline:** Predict demand. Optimize allocation. Reduce wasted space.  
**Team:** Laphing Buddha | DATATHON 2026 | Track: Education  

---

## 1. Architectural Philosophy

KAIROS is engineered around a clean, decoupled layer architecture separating user interface, API services, operational databases, mathematical optimization, and machine learning.

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

## 2. Component Structure

### 2.1 Persistence Layer (`src/database.py`)
- Self-contained SQLite database at `data/campus_space.db`.
- **Entities:**
  - `users`: Salted SHA-256 credentials with student and administrator roles.
  - `booking_requests`: Persistent student reservations with priority tiers and collision checks.
  - `reallocation_proposals`: Stored multi-room swap and swap-chain proposals.
  - `reallocation_items`: Individual room movement records within a proposal.
  - `data_imports`: Audit log of transactional data uploads with accepted/rejected counts.
  - `audit_logs`: Immutable trail of administrative and student actions.

### 2.2 Institutional Priority Engine (`src/priority.py`)
- Classifies event purposes into five institutional priority tiers:
  1. **Critical (Weight 100):** Formal examinations, midterms, final tests, accreditations.
  2. **High (Weight 50):** Core course lectures, mandatory laboratory sessions.
  3. **Medium (Weight 25):** Academic seminars, department colloquiums, faculty workshops.
  4. **Standard (Weight 10):** Student study groups, project sprint meetings, academic clubs.
  5. **Flexible (Weight 5):** Cultural festivals, drama rehearsals, music socials.
- Formally guarantees that academic assessments take precedence over discretionary social gatherings.

### 2.3 Optimization & Swap Engine (`src/optimizer.py`, `src/reallocation_engine.py`)
- **Multi-Event Optimizer:** Formulates global room assignment using Google OR-Tools CP-SAT with a priority-weighted objective:
  $$\max \sum_{i \in I, j \in J} \left( \text{BIG\_M} \times w_i - (\text{capacity}_j - \text{required}_i) \right) x_{i, j}$$
- **Reallocation Engine:** Bounded graph search detecting:
  - Mutual room swaps between concurrent classes where one is crowded and the other underutilized.
  - Swap chains where an unallocated request displaces a movable booking to a free room.
  - Dual execution modes: Admin Approval Mode (default) and Policy Auto-Execution.

### 2.4 Data Ingestion & REST API (`src/ingestion.py`, `src/api.py`)
- FastAPI application serving OpenAPI/Swagger docs at `http://localhost:8000/docs`.
- Strict schema enforcement, value bounds checking, referential integrity verification against room inventory, and idempotent upserts.

### 2.5 Pure NumPy Machine Learning (`src/ml_model.py`, `scripts/train_model.py`)
- Linear regression trained from scratch using Batch Gradient Descent and L2 regularization.
- Chronological train/validation/test split preventing future data leakage.
- Standalone CLI evaluator (`scripts/evaluate_model.py`) reporting metrics against global mean and same-weekday-hour baselines.
