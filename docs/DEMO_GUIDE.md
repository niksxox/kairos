# KAIROS — End-to-End Evaluation & Demo Guide

**Team:** Laphing Buddha | DATATHON 2026 | Track: Education  
**Local App URL:** `http://localhost:8501`  
**REST API & Swagger Docs:** `http://localhost:8000/docs`  

---

## 1. Quick Startup Commands

```powershell
# 1. Start the FastAPI REST Engine (Port 8000)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 2. Start the Streamlit Web Application (Port 8501)
streamlit run app.py --server.headless true --server.port 8501
```

---

## 2. Step-by-Step 16-Stage Hackathon Judging Workflow

| Step | Action | Expected Application Behavior |
| :---: | :--- | :--- |
| **1** | Open `http://localhost:8501` | Landing page loads with KAIROS clean academic dashboard (zero AI-slope gradients). |
| **2** | Check Sidebar Pipeline Status | Shows active datasets: `data/raw/` (120 rooms, 5,000 classes, 1,800 events, 5,000 requests). |
| **3** | Switch to **Student** Role | Active session switches to `student` (CSE). Navigation updates to student tools. |
| **4** | Open **Find a Room** | Search form renders with date, start hour, attendance, projector, and category selector. |
| **5** | Select **Formal Examination** | Notice dynamic Institutional Priority Banner updating to `CRITICAL (Weight 100)`. |
| **6** | Submit Room Search | Feasible recommendations render with fit scores, excess seat calculation, pros, and trade-offs. |
| **7** | Click **Book Room** (e.g. R002) | Booking is recorded in SQLite database with a unique request ID (e.g. `K-REQ-1001`, Status: `Pending`). |
| **8** | Open **My Bookings** | The new reservation appears in the student log with a color-coded `Critical: Exam` badge. |
| **9** | Switch to **Admin** Role | Active session switches to `admin` (Campus Operations). Full management suite appears. |
| **10** | Open **Admin Workspace** | The pending `K-REQ-1001` appears in the approval queue. Click **Approve**. Atomic collision validation passes and confirms room. |
| **11** | Open **Multi-Event Optimizer** | Set date to `2025-09-15` and run solver. Benchmarks Google OR-Tools CP-SAT against Greedy Baseline, proving exams receive priority. |
| **12** | Open **Automatic Reallocation & Swaps** | Scan `2025-09-15`. Mutual swaps and swap chains are detected. Click **Approve & Execute** on a proposal to commit changes atomically. |
| **13** | Open **Data Ingestion & REST API** | Links to OpenAPI Swagger UI (`http://localhost:8000/docs`). Demonstrates transactional CSV uploader and audit trail. |
| **14** | Open **Demand Forecast (ML)** | Displays pure NumPy Linear Regression metrics on held-out test data (Test MAE: 1.3403 vs Baseline 1.3682). Interactive hour inference flags capacity deficits. |
| **15** | Open **Calendar View** | Timetable shows colored timeline blocks for Academic Lectures (slate), Events (purple), and Confirmed Student Bookings (teal). |
| **16** | Restart Application / Check Persistence | Data remains persistent in `data/campus_space.db` across reloads and server restarts. |
