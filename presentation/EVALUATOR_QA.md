# KAIROS — HACKATHON EVALUATOR DEFENSE & Q&A GUIDE
**Team:** Laphing Buddha | **Product:** Kairos | **DATATHON 2026**

---

### Q1: Why did you choose this problem and track?
> **Answer:**  
> Campus real estate is one of the highest capital expenditures for universities, yet room scheduling is notoriously fragmented. In the DATATHON dataset, over 2,000 requests were waitlisted or rejected while scheduled room utilization was only 38.4%. By transforming room allocation from reactive, sequential booking into an intelligent decision-support system, KAIROS increases space efficiency and democratizes room access for student clubs and academic projects.

---

### Q2: Why did you choose this specific ML model, and what does it ingest and predict?
> **Answer:**  
> We framed the task as **hourly campus room demand forecasting**.  
> - **Target ($y$):** Number of student room requests submitted for a given campus time slot (Date, Hour).  
> - **Input Features ($X$):** 10 features including cyclic hour $\sin/\cos(2\pi H/24)$, cyclic weekday $\sin/\cos(2\pi W/7)$, calendar month, 1-step lag, 1-day lag, and 7-period rolling moving averages.  
> - **Model Choice:** Linear Regression with L2 regularization implemented from scratch using pure NumPy. Linear regression with cyclic and lag features provides high interpretability, exact parameter inspection, and zero risk of neural network overfitting on moderate temporal datasets.

---

### Q3: What does "trained from scratch" mean in your implementation?
> **Answer:**  
> We did not import `scikit-learn`, `statsmodels`, PyTorch, or use any pre-trained weights or LLM APIs for the model.  
> In [`src/ml_model.py`](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/src/ml_model.py), we wrote:
> 1. Xavier parameter initialization.
> 2. Analytical forward prediction: $\hat{y} = X \cdot w + b$.
> 3. Loss calculation: Mean Squared Error + $L_2$ penalty: $\frac{1}{2N} \sum (y - \hat{y})^2 + \frac{\lambda}{2} \|w\|^2$.
> 4. Gradient descent updates: $\nabla_w = \frac{1}{N} X^T (\hat{y} - y) + \frac{\lambda}{N} w$ and $\nabla_b = \frac{1}{N} \sum (\hat{y} - y)$.
> 5. Z-score feature standardization fit strictly on the training set.
> 6. JSON serialization and deserialization of weights and parameters.

---

### Q4: How did you prevent data leakage during ML training?
> **Answer:**  
> 1. **Chronological Splitting:** We rejected random cross-validation. The 1,505 hourly buckets were split strictly chronologically:
>    - Training: Aug 1 to Oct 23 (70%, 1,053 samples)
>    - Validation: Oct 23 to Nov 10 (15%, 226 samples)
>    - Test: Nov 10 to Nov 28 (15%, 226 samples)
> 2. **Feature Scaling Isolation:** Feature means and standard deviations were computed exclusively on the training set and applied forward to validation and test sets.
> 3. **Backward-Looking Lags:** Lag features (1-step, 1-day, 7-period rolling mean) were calculated using `.shift(1)` so no current or future target information was visible during feature calculation.

---

### Q5: What is your baseline and what are your measured test metrics?
> **Answer:**  
> We evaluated against two baselines on the held-out test set:
> 1. **Overall Mean Baseline:** MAE = 1.3167 requests, RMSE = 1.6573 requests.
> 2. **Same-Weekday-Hour Stratified Baseline:** MAE = 1.3682 requests, RMSE = 1.7278 requests.
> - **KAIROS NumPy Model:** **MAE = 1.3403 requests** (+2.04% improvement over same-weekday-hour baseline), **RMSE = 1.6892 requests** (+2.23% improvement).
> - **Runtime:** Training completed in **0.167 seconds**.

---

### Q6: How does optimization differ from machine learning in KAIROS?
> **Answer:**  
> - **Machine Learning (Predictive):** Predicts aggregate future demand to answer: *"How many requests will arrive on Thursday at 14:00?"* This gives administrators early warning of upcoming bottlenecks.
> - **Optimization (Prescriptive):** Uses **Google OR-Tools CP-SAT** to solve actual allocations to answer: *"Given 40 real student requests on September 15, which exact room should each request receive?"* Optimization guarantees that every single hard constraint (capacity, equipment, no overlap) is strictly satisfied.

---

### Q7: How do you mathematically guarantee that no double-bookings occur?
> **Answer:**  
> We implement a three-layer defense:
> 1. **Half-Open Interval Semantics:** Intervals are treated as $[start, end)$. An event ending at 11:00 does not conflict with one starting at 11:00.
> 2. **Pre-Pruning in Availability Engine:** Any room with an overlapping class or event during $[s, e)$ is excluded before candidate pair creation.
> 3. **Pairwise Mutual Exclusion Constraints:** In OR-Tools CP-SAT, for any two requests $i_1, i_2$ overlapping on date $D$:
>    $$x_{i_1, j} + x_{i_2, j} \le 1 \quad \forall j$$
> 4. **Atomic SQLite Revalidation:** In [`src/database.py`](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/src/database.py), when an administrator clicks "Approve", the backend re-queries existing confirmed bookings inside an atomic transaction. If another administrator or user approved a conflicting request a second earlier, the collision is detected and blocked.

---

### Q8: How is scheduled utilization calculated without double-counting?
> **Answer:**  
> In [`src/availability.py`](file:///C:/Users/DELL/Documents/CampusSpace-Intelligence/src/availability.py), the `IntervalMerger` sorts all class and event intervals chronologically and merges contiguous/overlapping blocks:
> - Class: 09:00–11:00, Event: 10:00–12:00 $\rightarrow$ Merged: 09:00–12:00.
> - Occupied duration is **3 hours, not 4 hours**.
> - Denominator is strictly defined as the campus operating window: 6 days (Mon–Sat) $\times$ 14 hours (08:00–22:00) = 84 operating hours per week.

---

### Q9: What is genuinely implemented versus future scope?
> **Answer:**  
> - **Genuinely Implemented:** Data ingestion/audit CLI, interval merging availability engine, explainable matching engine, Google OR-Tools CP-SAT multi-event optimizer, greedy benchmark, capacity swap detector, NumPy ML demand forecasting model trained from scratch, SQLite persistence with hashed password auth, Plotly analytics, and 26 passing automated tests.
> - **Future Scope:** Live university student information system (SIS) sync via OAuth2 SSO, and hardware IoT occupancy sensor integration (PIR / Wi-Fi probe sniffing).

---

### Q10: How would KAIROS integrate with an existing university infrastructure?
> **Answer:**  
> KAIROS uses a decoupled architecture. Its ingestion layer supports any standard tabular format or REST API. In production, the CSV loader would be replaced by webhooks connecting to Banner, Canvas, or PeopleSoft timetables. Decisions are persisted in standard SQL tables, enabling two-way calendar sync (iCal / Google Calendar) for confirmed bookings.
