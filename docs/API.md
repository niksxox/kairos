# KAIROS — REST API Specification & Endpoints

**Interactive OpenAPI Documentation:** Available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

---

## 1. Overview & Base URL

The KAIROS backend exposes a high-performance REST API powered by **FastAPI**.
- Base URL: `http://localhost:8000`
- Content Type: `application/json`
- CORS: Enabled for localhost and web clients

---

## 2. Endpoints Reference

### 2.1 Health & Verification
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns operational health, version, and inventory counts. |
| `GET` | `/api/data/sources` | Lists active dataset sources, record counts, and verification status. |

**Sample Response (`GET /api/health`):**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "product": "KAIROS",
  "team": "Laphing Buddha",
  "timestamp": "2026-10-16T16:20:13.526641",
  "rooms_in_inventory": 120,
  "scheduled_classes": 5000,
  "campus_events": 1800
}
```

---

### 2.2 Data Ingestion Pipeline
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/data/import` | Generic batch ingestion endpoint for any entity type. |
| `POST` | `/api/data/rooms` | Ingest physical room records. |
| `POST` | `/api/data/schedules` | Ingest academic class schedules. |
| `POST` | `/api/data/events` | Ingest campus special events. |
| `POST` | `/api/data/requests` | Ingest incoming demand / room requests. |
| `GET` | `/api/data/imports` | Returns recent transactional ingestion audit records. |

**Sample Request (`POST /api/data/import`):**
```json
{
  "entity_type": "rooms",
  "source_name": "Facilities_Export_Q3",
  "records": [
    {
      "room_id": "R121",
      "building": "North",
      "capacity": 60,
      "room_type": "Classroom",
      "projector": 1,
      "computers": 0,
      "accessible": 1
    }
  ]
}
```

---

### 2.3 Automatic Reallocation & Swap Engine
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/reallocation/plan` | Scans schedules and unallocated demand to generate swap proposals. |
| `GET` | `/api/reallocation/proposals` | Lists stored reallocation proposals (filter by `status=Pending`). |
| `GET` | `/api/reallocation/proposals/{proposal_id}` | Retrieves detailed movement items for a specific proposal. |
| `POST` | `/api/reallocation/proposals/{proposal_id}/approve` | Atomically executes proposal movements in the database. |
| `POST` | `/api/reallocation/proposals/{proposal_id}/reject` | Records administrative rejection with justification. |

---

### 2.4 Bookings Lifecycle
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/bookings` | Lists booking requests (query by `username` or `status_filter`). |
| `POST` | `/api/bookings` | Submits a new booking request with priority tiering. |
| `POST` | `/api/bookings/{request_id}/cancel` | Cancels an existing reservation and restores room availability. |

**Sample Request (`POST /api/bookings`):**
```json
{
  "username": "student",
  "room_id": "R005",
  "request_date": "2025-08-04",
  "start_hour": 14.0,
  "duration_hours": 2.0,
  "required_capacity": 40,
  "purpose": "Final Calculus Examination",
  "projector_required": true,
  "computers_required": false,
  "accessible_required": true,
  "auto_approve": false
}
```
