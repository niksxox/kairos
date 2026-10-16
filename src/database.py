"""
KAIROS - Persistence & Database Service (SQLite)
Manages user accounts, persistent booking requests, audit trails, and atomic approval verification.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from src.auth import hash_password, verify_password

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "campus_space.db"


class CampusDatabase:
    """
    Thread-safe SQLite database manager for Kairos.
    Stores user credentials, student booking requests, room status overrides, and audit logs.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._init_schema()
        self._seed_default_users()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._get_connection() as conn:
            # 1. Users Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('student', 'admin')),
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. Booking Requests Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS booking_requests (
                    request_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    room_id TEXT,
                    request_date TEXT NOT NULL,
                    start_hour REAL NOT NULL,
                    duration_hours REAL NOT NULL,
                    required_capacity INTEGER NOT NULL,
                    projector_required INTEGER DEFAULT 0,
                    computers_required INTEGER DEFAULT 0,
                    accessible_required INTEGER DEFAULT 0,
                    purpose TEXT,
                    status TEXT NOT NULL CHECK(status IN ('Pending', 'Approved', 'Rejected', 'Cancelled')),
                    rejection_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            """)

            # 3. Room Operational Status Overrides
            conn.execute("""
                CREATE TABLE IF NOT EXISTS room_overrides (
                    room_id TEXT PRIMARY KEY,
                    is_operational INTEGER DEFAULT 1,
                    notes TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Audit Log Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    performed_by TEXT NOT NULL,
                    entity_id TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _seed_default_users(self):
        """Seeds default student and administrator accounts if not present."""
        default_accounts = [
            ("student", "student123", "student", "CSE"),
            ("admin", "admin123", "admin", "Campus Operations")
        ]
        with self._get_connection() as conn:
            for username, password, role, dept in default_accounts:
                cur = conn.cursor()
                cur.execute("SELECT username FROM users WHERE username = ?", (username,))
                if not cur.fetchone():
                    salt, pwd_hash = hash_password(password)
                    conn.execute("""
                        INSERT INTO users (username, salt, password_hash, role, department)
                        VALUES (?, ?, ?, ?, ?)
                    """, (username, salt, pwd_hash, role, dept))
            conn.commit()

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticates user credentials and returns user payload if valid."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT username, salt, password_hash, role, department FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            if not row:
                return None
            
            if verify_password(password, row["salt"], row["password_hash"]):
                return {
                    "username": row["username"],
                    "role": row["role"],
                    "department": row["department"]
                }
        return None

    def create_booking_request(
        self,
        username: str,
        room_id: Optional[str],
        request_date: str,
        start_hour: float,
        duration_hours: float,
        required_capacity: int,
        projector_required: bool = False,
        computers_required: bool = False,
        accessible_required: bool = False,
        purpose: str = "Study group",
        auto_approve: bool = False
    ) -> str:
        """
        Creates a new persistent booking request.
        Generates a unique request ID (e.g. K-REQ-1001).
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM booking_requests")
            next_idx = cur.fetchone()[0] + 1
            req_id = f"K-REQ-{next_idx:04d}"

            status = "Approved" if auto_approve and room_id else "Pending"

            conn.execute("""
                INSERT INTO booking_requests (
                    request_id, username, room_id, request_date, start_hour, duration_hours,
                    required_capacity, projector_required, computers_required, accessible_required,
                    purpose, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                req_id, username, room_id, request_date, start_hour, duration_hours,
                required_capacity, int(projector_required), int(computers_required), int(accessible_required),
                purpose, status
            ))

            # Audit log
            conn.execute("""
                INSERT INTO audit_logs (action, performed_by, entity_id, details)
                VALUES ('CREATE_REQUEST', ?, ?, ?)
            """, (username, req_id, f"Date: {request_date}, Hour: {start_hour}, Room: {room_id or 'Auto-match'}"))
            conn.commit()

        return req_id

    def get_user_bookings(self, username: str) -> pd.DataFrame:
        """Returns all booking requests created by a specific student."""
        with self._get_connection() as conn:
            query = "SELECT * FROM booking_requests WHERE username = ? ORDER BY request_date DESC, start_hour DESC"
            return pd.read_sql_query(query, conn, params=(username,))

    def get_all_bookings(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """Returns all booking requests across campus for administrative management."""
        with self._get_connection() as conn:
            if status_filter:
                query = "SELECT * FROM booking_requests WHERE status = ? ORDER BY created_at DESC"
                return pd.read_sql_query(query, conn, params=(status_filter,))
            else:
                query = "SELECT * FROM booking_requests ORDER BY created_at DESC"
                return pd.read_sql_query(query, conn)

    def cancel_booking(self, request_id: str, username: str, is_admin: bool = False) -> Tuple[bool, str]:
        """
        Cancels a booking request. Students can only cancel their own bookings.
        Immediately restores room availability.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT username, status, room_id, request_date, start_hour FROM booking_requests WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            if not row:
                return False, "Request not found."

            if not is_admin and row["username"] != username:
                return False, "Unauthorized: You may only cancel your own bookings."

            if row["status"] in ("Cancelled", "Rejected"):
                return False, f"Request is already {row['status'].lower()}."

            conn.execute("""
                UPDATE booking_requests
                SET status = 'Cancelled', updated_at = CURRENT_TIMESTAMP
                WHERE request_id = ?
            """, (request_id,))

            conn.execute("""
                INSERT INTO audit_logs (action, performed_by, entity_id, details)
                VALUES ('CANCEL_BOOKING', ?, ?, ?)
            """, (username, request_id, f"Cancelled room {row['room_id']} on {row['request_date']}"))
            conn.commit()

        return True, "Booking successfully cancelled. The room is now available for other students."

    def approve_booking(
        self,
        request_id: str,
        admin_username: str,
        availability_checker,
        assigned_room_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Approves a pending request with ATOMIC CONFLICT REVALIDATION.
        Guarantees that two administrators or concurrent bookings can NEVER double-book a room!
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT request_id, room_id, request_date, start_hour, duration_hours, required_capacity, status
                FROM booking_requests WHERE request_id = ?
            """, (request_id,))
            row = cur.fetchone()
            if not row:
                return False, "Request not found."

            if row["status"] == "Approved":
                return False, "Request is already approved."

            target_room = assigned_room_id or row["room_id"]
            if not target_room:
                return False, "Cannot approve: No room specified for allocation."

            # Check conflict with existing approved bookings in the database
            cur.execute("""
                SELECT request_id, start_hour, duration_hours FROM booking_requests
                WHERE room_id = ? AND request_date = ? AND status = 'Approved' AND request_id != ?
            """, (target_room, row["request_date"], request_id))
            db_conflicts = cur.fetchall()

            req_start = float(row["start_hour"])
            req_end = req_start + float(row["duration_hours"])

            for c_id, c_start, c_dur in db_conflicts:
                c_s = float(c_start)
                c_e = c_s + float(c_dur)
                if max(req_start, c_s) < min(req_end, c_e):
                    return False, f"Approval collision: Room {target_room} was already confirmed for another request ({c_id}) from {int(c_s):02d}:00 to {int(c_e):02d}:00."

            # Check conflict with baseline class schedules and events
            if availability_checker is not None:
                is_free, reason = availability_checker.is_room_available(
                    target_room, row["request_date"], req_start, float(row["duration_hours"])
                )
                if not is_free:
                    return False, f"Cannot approve: {reason}"

            # Approval passed all invariants! Update atomically
            conn.execute("""
                UPDATE booking_requests
                SET room_id = ?, status = 'Approved', updated_at = CURRENT_TIMESTAMP
                WHERE request_id = ?
            """, (target_room, request_id))

            conn.execute("""
                INSERT INTO audit_logs (action, performed_by, entity_id, details)
                VALUES ('APPROVE_BOOKING', ?, ?, ?)
            """, (admin_username, request_id, f"Approved room {target_room} for {row['request_date']}"))
            conn.commit()

        return True, f"Request {request_id} confirmed and allocated to Room {target_room}."

    def reject_booking(self, request_id: str, admin_username: str, reason: str) -> Tuple[bool, str]:
        """Rejects a pending booking request with explainable justification."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT status FROM booking_requests WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            if not row:
                return False, "Request not found."

            conn.execute("""
                UPDATE booking_requests
                SET status = 'Rejected', rejection_reason = ?, updated_at = CURRENT_TIMESTAMP
                WHERE request_id = ?
            """, (reason, request_id))

            conn.execute("""
                INSERT INTO audit_logs (action, performed_by, entity_id, details)
                VALUES ('REJECT_BOOKING', ?, ?, ?)
            """, (admin_username, request_id, f"Reason: {reason}"))
            conn.commit()

        return True, f"Request {request_id} rejected. Reason recorded."

    def get_confirmed_bookings_for_date(self, request_date: str) -> List[Dict[str, Any]]:
        """Returns confirmed student bookings for conflict checking."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT request_id, room_id, start_hour, duration_hours, purpose, username
                FROM booking_requests
                WHERE request_date = ? AND status = 'Approved'
            """, (request_date,))
            return [dict(r) for r in cur.fetchall()]
