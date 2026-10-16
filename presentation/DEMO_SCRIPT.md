# KAIROS — 3-TO-5 MINUTE HACKATHON DEMO WALKTHROUGH SCRIPT
**Team:** Laphing Buddha | **Product:** Kairos | **DATATHON 2026**

---

### Step 1: The Problem Hook (30 seconds)
> *"Judges, university campuses face a massive paradox: The campus has space, but students don't. In our 11,920-record dataset, over 2,000 student requests were waitlisted or rejected, while lecture halls sat half-empty. Current campus systems are reactive and fragmented. We built **KAIROS**—an intelligent space coordination platform that predicts demand with machine learning, optimizes allocation with integer programming, and eliminates wasted space."*

---

### Step 2: Student Discovery & Booking Flow (1 minute)
1. **Show Overview:**
   > *"We start on the KAIROS Overview dashboard at `localhost:8501`. As a student logged in, I immediately see campus capacity: 120 rooms, 5,000 recurring classes, and 1,800 events."*
2. **Navigate to 'Find a Room':**
   > *"Let's book a room for our student AI study group on September 15 at 14:00 for 30 attendees needing a projector and computers. As we click 'Search', KAIROS applies strict hard constraints in real-time."*
3. **Show Results & Explanations:**
   > *"Notice the recommendations: Room R002 in West Building is ranked #1 with an 87.5% suitability score. KAIROS explains exactly why it was chosen—tight capacity fit, computer workstations, and verified availability. Notice also the Disqualified Rooms tab: smaller rooms like R001 are excluded for insufficient capacity, while others are excluded due to schedule conflicts. No guessing, no double-booking."*
4. **Submit Booking:**
   > *"We click 'Book R002'. The request is instantly recorded in our persistent SQLite database with Request ID `K-REQ-0001` in 'Pending' status."*

---

### Step 3: Administrator Workspace & Conflict Revalidation (1 minute)
1. **Switch to Administrator Role:**
   > *"Now, using our 1-click role switcher in the sidebar, let's switch to the Administrator portal."*
2. **Review Pending Requests:**
   > *"Under 'Admin Workspace', the administrator sees our new request. If another administrator attempts to approve a conflicting request at the same time, KAIROS's atomic database transaction revalidates availability and blocks collisions."*
3. **Approve Request:**
   > *"We click 'Approve'. The booking is confirmed. If we check the Room Timetable on the Calendar view, Room R002 is now dynamically blocked for September 15 from 14:00 to 16:00!"*

---

### Step 4: Multi-Event Optimization & Capacity Swaps (1 minute)
1. **Open 'Multi-Event Optimizer':**
   > *"When dozens of competing requests arrive simultaneously, human schedulers cannot solve the combinatorial explosion. We implemented an exact mathematical solver using **Google OR-Tools CP-SAT**."*
2. **Run Solver vs. Baseline:**
   > *"Watch what happens when we run global allocation on September 15. In under 1.5 seconds, OR-Tools achieves **+18% to +35% higher fulfillment** than the sequential greedy baseline while cutting excess seat waste by **22%**, with a mathematical guarantee of zero hard-constraint violations."*
3. **Show Capacity Swaps:**
   > *"On 'Capacity Swaps', KAIROS scans concurrent classes and detects where swapping an overflowing class with an oversized room saves dozens of wasted seats at the exact same hour with zero timetable disruption."*

---

### Step 5: Machine Learning Demand Forecasting (1 minute)
1. **Open 'Demand Forecast':**
   > *"Finally, to anticipate space shortages before they occur, we built and trained a demand forecasting model from scratch using pure NumPy—no scikit-learn or blackbox libraries."*
2. **Explain Model & Metrics:**
   > *"Trained on a strict chronological 70/15/15 split across 1,505 hourly buckets, our model achieved a test MAE of **1.3403 requests**, outperforming the stratified baseline. When forecast demand exceeds available supply, KAIROS automatically triggers an administrative bottleneck alert."*

---

### Step 6: Closing Summary (20 seconds)
> *"KAIROS is not a mock-up: it has 26 passing automated unit tests, clean SQLite persistence, mathematical CP-SAT optimization, and a verified ML model trained from scratch. We give the campus the space students need, when they need it. Thank you!"*
