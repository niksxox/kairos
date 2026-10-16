"""
KAIROS - Transactional Data Ingestion & Validation Pipeline
Handles controlled data ingestion from CSV uploads or JSON payloads.
Validates schemas, referential integrity, formats, detects duplicates,
and transactionally records imports into the persistent audit trail.
"""

import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np

from config.settings import VALID_BUILDINGS, VALID_ROOM_TYPES, VALID_WEEKDAYS


class DataIngestionService:
    """
    Validates, normalizes, and stages incoming campus datasets.
    Supports rooms, class schedules, events, and room requests.
    """

    SCHEMAS = {
        "rooms": {
            "required": ["room_id", "building", "capacity"],
            "optional": ["floor", "room_type", "projector", "computers", "accessible", "power_outlets"],
            "types": {"capacity": int}
        },
        "class_schedule": {
            "required": ["schedule_id", "room_id", "course_code", "weekday", "start_hour", "duration_hours"],
            "optional": ["expected_attendance", "department"],
            "types": {"start_hour": float, "duration_hours": float}
        },
        "events": {
            "required": ["event_id", "room_id", "event_type", "start_time", "duration_hours"],
            "optional": ["expected_attendance", "actual_attendance"],
            "types": {"duration_hours": float}
        },
        "room_requests": {
            "required": ["request_id", "request_date", "start_hour", "duration_hours", "required_capacity"],
            "optional": ["projector_required", "computers_required", "purpose", "status"],
            "types": {"start_hour": float, "duration_hours": float, "required_capacity": int}
        }
    }

    def __init__(self, database=None, existing_rooms: Optional[Set[str]] = None):
        self.db = database
        self.existing_rooms = existing_rooms or set()

    def ingest_payload(
        self,
        entity_type: str,
        records: List[Dict[str, Any]],
        source_name: str = "API_Upload"
    ) -> Dict[str, Any]:
        """
        Processes and validates an ingestion batch.
        Enforces schema, types, range checks, and referential integrity.
        """
        import_id = f"IMP-{uuid.uuid4().hex[:8].upper()}"
        total_received = len(records)

        if entity_type not in self.SCHEMAS:
            return {
                "import_id": import_id,
                "entity_type": entity_type,
                "source": source_name,
                "total_received": total_received,
                "total_accepted": 0,
                "total_rejected": total_received,
                "status": "FAILED",
                "error_log": [f"Unsupported entity type: '{entity_type}'. Must be one of {list(self.SCHEMAS.keys())}"]
            }

        schema = self.SCHEMAS[entity_type]
        accepted = []
        errors = []
        seen_ids = set()
        seen_req_signatures = set()

        id_key = f"{entity_type[:-1]}_id" if entity_type != "class_schedule" else "schedule_id"
        if entity_type == "rooms":
            id_key = "room_id"

        for idx, rec in enumerate(records):
            rec_id = str(rec.get(id_key, f"row_{idx}"))

            # Check 1: Required columns
            missing = [col for col in schema["required"] if col not in rec or rec[col] is None or rec[col] == ""]
            if missing:
                errors.append(f"Record {rec_id}: Missing required fields: {missing}")
                continue

            # Check 2: Duplicate ID within batch
            if rec_id in seen_ids:
                errors.append(f"Record {rec_id}: Duplicate identifier within batch")
                continue
            seen_ids.add(rec_id)

            # Check 2b: Duplicate request payload signature within batch
            if entity_type == "room_requests":
                try:
                    sig = (
                        str(rec.get("request_date", "")),
                        float(rec.get("start_hour", 0)),
                        float(rec.get("duration_hours", 0)),
                        int(rec.get("required_capacity", 0)),
                        str(rec.get("purpose", "")).strip().lower()
                    )
                    if sig in seen_req_signatures:
                        errors.append(f"Record {rec_id}: Duplicate request payload — an identical request already exists in this import batch.")
                        continue
                    seen_req_signatures.add(sig)
                except (ValueError, TypeError):
                    pass

            # Check 3: Referential integrity for room_id
            if entity_type in ("class_schedule", "events") and self.existing_rooms:
                rm_id = str(rec.get("room_id", ""))
                if rm_id not in self.existing_rooms:
                    errors.append(f"Record {rec_id}: Referential integrity violation — room_id '{rm_id}' does not exist in inventory.")
                    continue

            # Check 4: Value bounds
            try:
                if "start_hour" in rec:
                    sh = float(rec["start_hour"])
                    if sh < 0 or sh > 24:
                        errors.append(f"Record {rec_id}: Invalid start_hour {sh} (must be between 0 and 24).")
                        continue

                if "duration_hours" in rec:
                    dh = float(rec["duration_hours"])
                    if dh <= 0 or dh > 12:
                        errors.append(f"Record {rec_id}: Invalid duration_hours {dh} (must be between 0.5 and 12).")
                        continue

                if "capacity" in rec:
                    cap = int(rec["capacity"])
                    if cap <= 0:
                        errors.append(f"Record {rec_id}: Capacity must be positive (got {cap}).")
                        continue

                if "required_capacity" in rec:
                    req_cap = int(rec["required_capacity"])
                    if req_cap <= 0:
                        errors.append(f"Record {rec_id}: Required capacity must be positive (got {req_cap}).")
                        continue

            except (ValueError, TypeError) as e:
                errors.append(f"Record {rec_id}: Type conversion failed: {str(e)}")
                continue

            # Valid record!
            accepted.append(rec)

        total_accepted = len(accepted)
        total_rejected = len(errors)
        status = "SUCCESS" if total_rejected == 0 else ("PARTIAL" if total_accepted > 0 else "FAILED")

        # Record in persistent audit database
        if self.db:
            err_summary = "; ".join(errors[:5]) if errors else None
            self.db.record_data_import(
                import_id=import_id,
                source=source_name,
                entity_type=entity_type,
                total_received=total_received,
                total_accepted=total_accepted,
                total_rejected=total_rejected,
                status=status,
                error_log=err_summary
            )

        return {
            "import_id": import_id,
            "entity_type": entity_type,
            "source": source_name,
            "total_received": total_received,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "status": status,
            "accepted_records": accepted,
            "errors": errors,
            "error_log": errors
        }
