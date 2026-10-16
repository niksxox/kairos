import json
import re
import os

transcript_path = r"C:\Users\DELL\.gemini\antigravity\brain\40217b74-e4be-4346-a2a1-7d66e00bd59f\.system_generated\logs\transcript_full.jsonl"
output_dir = r"C:\Users\DELL\Documents\CampusSpace-Intelligence\data\raw"
os.makedirs(output_dir, exist_ok=True)

with open(transcript_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for idx, line in enumerate(lines):
    if "schedule_id,room_id" in line:
        d = json.loads(line)
        print(f"Index {idx}: keys={list(d.keys())}")
        for k in d:
            val = str(d[k])
            if "schedule_id,room_id" in val:
                print(f"  Field '{k}' contains dataset! Length: {len(val)}")
                raw_text = val
                
                # Check for datasets in this field
                if "event_id,room_id" in raw_text and "request_id,request_date" in raw_text:
                    # 1. class_schedule.csv
                    m_sched = re.search(r"(schedule_id,room_id,course_code,weekday,start_hour,duration_hours,expected_attendance,department.*?)(?=event_id,room_id|\Z)", raw_text, re.DOTALL)
                    if m_sched:
                        sched_lines = [l.strip() for l in m_sched.group(1).splitlines() if l.strip()]
                        out_p = os.path.join(output_dir, "class_schedule.csv")
                        with open(out_p, "w", encoding="utf-8", newline="") as out:
                            out.write("\n".join(sched_lines) + "\n")
                        print(f"Wrote class_schedule.csv with {len(sched_lines)-1} rows")

                    # 2. events.csv
                    m_ev = re.search(r"(event_id,room_id,event_type,start_time,duration_hours,expected_attendance,actual_attendance.*?)(?=EDUCATION|request_id,request_date|\Z)", raw_text, re.DOTALL)
                    if m_ev:
                        ev_lines = [l.strip() for l in m_ev.group(1).splitlines() if l.strip()]
                        out_p = os.path.join(output_dir, "events.csv")
                        with open(out_p, "w", encoding="utf-8", newline="") as out:
                            out.write("\n".join(ev_lines) + "\n")
                        print(f"Wrote events.csv with {len(ev_lines)-1} rows")

                    # 3. room_requests.csv
                    m_rq = re.search(r"(request_id,request_date,start_hour,duration_hours,required_capacity,projector_required,computers_required,purpose,status.*?)(?=room_id,building,floor,room_type|\Z)", raw_text, re.DOTALL)
                    if m_rq:
                        rq_lines = [l.strip() for l in m_rq.group(1).splitlines() if l.strip()]
                        out_p = os.path.join(output_dir, "room_requests.csv")
                        with open(out_p, "w", encoding="utf-8", newline="") as out:
                            out.write("\n".join(rq_lines) + "\n")
                        print(f"Wrote room_requests.csv with {len(rq_lines)-1} rows")

                    # 4. rooms.csv
                    m_rm = re.search(r"(room_id,building,floor,room_type,capacity,projector,computers,accessible,power_outlets.*?)(?=\n[A-Z_]+|\Z)", raw_text, re.DOTALL)
                    if m_rm:
                        rm_lines = [l.strip() for l in m_rm.group(1).splitlines() if l.strip() and (l.strip().startswith("room_id") or l.strip().startswith("R"))]
                        out_p = os.path.join(output_dir, "rooms.csv")
                        with open(out_p, "w", encoding="utf-8", newline="") as out:
                            out.write("\n".join(rm_lines) + "\n")
                        print(f"Wrote rooms.csv with {len(rm_lines)-1} rows")
                    
                    break
