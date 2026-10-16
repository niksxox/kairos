# KAIROS — Automatic Room Reallocation & Swap Engine

## 1. Core Problem & Concept

Space allocation is rarely static. Lecture halls sit half-empty with 15 students while an overcrowded engineering class overflows into the hallway. 

Traditional room management systems treat allocations as immutable silos. **KAIROS Reallocation Engine** resolves space deficits by optimizing room allocations dynamically without altering class or event start times.

---

## 2. Distinction: Request Allocation vs. Room Swaps

- **Request Allocation:** Solved via **Google OR-Tools CP-SAT**. Takes unassigned incoming requests ($Q \in \text{Requests}$) and matches them against free room inventory.
- **Mutual Room Swaps:** Operates on records that **genuinely have existing room assignments**. Swaps Room 1 and Room 2 concurrently when both classes take place at the same hour, eliminating capacity waste.
- **Bounded Swap Chains:** When an unallocated high-priority request $Q$ finds no directly available room, the engine searches for a movable lower-priority booking $B$ in Room 1 that can be relocated to a free Room 2, liberating Room 1 for $Q$.

---

## 3. Institutional Priority Protection

Critical academic events are strictly protected from displacement:
- **Exams and Formal Tests (Weight 100):** Can **never** be displaced by cultural gatherings, club meetings, or informal study groups.
- **Core Lectures (Weight 50):** Protected from displacement; can only participate in mutual swaps if the proposed room has equal or greater capacity and identical mandatory facilities.
- **Cultural & Social Events (Weight 5):** Marked as flexible and eligible for bounded reassignment to alternative rooms when higher-priority academic needs arise.

---

## 4. Hard Constraints

1. **Capacity Invariant:** $\text{capacity}(\text{Room}) \ge \text{attendance}(\text{Event})$.
2. **Mandatory Facilities:** Projectors, computers, and accessibility ramps must be present if required.
3. **Half-Open Temporal Disjointness:** Overlapping intervals $[s_1, e_1) \cap [s_2, e_2) = \emptyset$.
4. **Zero Time Shift:** Start hours and durations remain strictly unchanged.
5. **Atomic All-or-Nothing Commit:** Reallocation proposals update all affected rooms inside a single SQLite transaction with automatic rollback upon collision.

---

## 5. Execution Policies

### Mode A: Administrator Approval Mode (Default)
1. Proposals are generated and displayed with before/after comparisons, seat mismatch reductions, and rationales.
2. An authorized administrator inspects the proposal.
3. On approval, the backend atomically executes room updates, changes proposal status to `Approved`, and records the decision in `audit_logs`.

### Mode B: Policy-Driven Auto-Execution
- Applicable only to flexible student bookings explicitly tagged as movable.
- Re-verifies all availability constraints in the backend immediately before committing changes.
