"""
Kairos - Standalone Data Quality Auditor CLI
Usage: python scripts/validate_data.py [--sample]
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser(description="Audit Kairos campus space datasets.")
    parser.add_argument("--sample", action="store_true", help="Audit sample dataset instead of raw")
    args = parser.parse_args()

    loader = DataLoader()
    datasets, source_name = loader.load_all(use_sample=args.sample)

    print("======================================================================")
    print(" KAIROS - CAMPUS SPACE INTELLIGENCE: DATA QUALITY AUDIT REPORT")
    print(f" Team: Laphing Buddha | DATATHON 2026")
    print(f" Active Source: {source_name}")
    print("======================================================================\n")

    validator = DataValidator(datasets)
    report = validator.validate_all()

    print(f"Overall Status: {report['summary']['status']}")
    print(f"Total Ingested Records: {report['summary']['total_records_ingested']:,}")
    print(f"Total Missing Values:   {report['summary']['total_missing_fields']:,}")
    print(f"Total Duplicate IDs:    {report['summary']['total_duplicate_ids']:,}")
    print("----------------------------------------------------------------------")

    print("\n[DATASET BREAKDOWN]")
    for name, audit in report["dataset_audits"].items():
        print(f"  * {name.upper()}:")
        print(f"      Rows: {audit['row_count']:,} | Columns: {audit['column_count']}")
        print(f"      Missing Required Columns: {audit['missing_expected_columns'] or 'None'}")
        print(f"      Duplicate Primary Keys:   {audit['duplicate_identifiers_count']}")
        print(f"      Null Values:              {audit['null_values_count']}")

    print("\n[REFERENTIAL INTEGRITY (ROOM IDs)]")
    ref = report["referential_integrity"]
    print(f"  Cataloged Rooms:        {ref['total_rooms_cataloged']}")
    print(f"  Orphan Schedule Rooms:  {len(ref['orphan_schedule_rooms'])} ({ref['orphan_schedule_rooms'] or 'None'})")
    print(f"  Orphan Event Rooms:     {len(ref['orphan_event_rooms'])} ({ref['orphan_event_rooms'] or 'None'})")
    print(f"  Referential Passed:     {ref['integrity_passed']}")

    print("\n[TEMPORAL CONSISTENCY]")
    temp = report["temporal_audits"]
    print(f"  Invalid Schedule Hours (outside 0-23):  {temp['invalid_schedule_hours']}")
    print(f"  Invalid Schedule Durations (<=0 hours): {temp['invalid_schedule_durations']}")
    print(f"  Invalid Event Timestamps:               {temp['invalid_event_timestamps']}")
    print(f"  Invalid Request Hours (outside 0-23):   {temp['invalid_request_hours']}")

    print("\n[CAPACITY AUDIT]")
    cap = report["capacity_audits"]
    print(f"  Rooms with Capacity <= 0:               {cap['rooms_with_invalid_capacity']}")
    print(f"  Schedule with Attendance < 0:           {cap['schedule_with_invalid_attendance']}")
    print(f"  Requests with Required Capacity <= 0:   {cap['requests_with_invalid_capacity']}")
    print(f"  Requests Exceeding Campus Max Capacity: {cap['requests_exceeding_max_campus_capacity']}")

    print("\n======================================================================")
    print(" AUDIT COMPLETED SUCCESSFULLY. ZERO SILENT DROPS.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
