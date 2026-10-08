#!/usr/bin/env python3
"""
trackB_lamp_refresh.py -- The Bell Study: daily refresh of the agency reference (MBTA LAMP bus events) + instrument check.

1. Downloads LAMP_RECENT_Bus_Events.parquet (MBTA LAMP public export, ~180 MB, last ~7 service days) only if it changed
   (Last-Modified), to C:\\dev\\rel_trackB\\lamp\\ (local, not OneDrive).
2. Keeps a permanent per-day history of the Route 1/52 rows and the columns the instrument check needs:
   C:\\dev\\rel_trackB\\lamp\\history\\lamp_bus_events_<service_date>.parquet  (only completed service days; rewritten if newer).
3. Runs pilot_instrument_check.py on every Track B day that has both stop events and agency history.
Data: MBTA LAMP public export (performancedata.mbta.com), used under the MassDOT Developers License; see README.
Scheduled as task TrackB_LampCheck (daily, after LAMP's morning refresh).
"""
import datetime as dt, os, subprocess, sys
import requests
import pyarrow.parquet as pq
import pyarrow.compute as pc
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://performancedata.mbta.com/lamp/tableau/bus/LAMP_RECENT_Bus_Events.parquet"
DIR = r"C:\dev\rel_trackB\lamp"; HIST = os.path.join(DIR, "history"); FILE = os.path.join(DIR, "LAMP_RECENT_Bus_Events.parquet")
STAMP = os.path.join(DIR, "last_modified.txt"); LOG = os.path.join(DIR, "refresh.log")
COLS = ["service_date", "trip_id", "stop_id", "route_id", "checkpoint_id", "plan_stop_departure_dt", "tm_actual_arrival_dt",
        "tm_actual_departure_dt", "gtfs_arrival_dt", "vehicle_label", "stop_sequence"]
ET = ZoneInfo("America/New_York")

def log(m):
    os.makedirs(DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {m}\n")

def main():
    os.makedirs(HIST, exist_ok=True)
    head = requests.head(URL, timeout=60); lm = head.headers.get("last-modified", "")
    prev = open(STAMP).read().strip() if os.path.exists(STAMP) else ""
    if lm and lm == prev and os.path.exists(FILE):
        log(f"unchanged ({lm}) — no download")
    else:
        tmp = FILE + ".part"
        with requests.get(URL, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20): f.write(chunk)
        os.replace(tmp, FILE); open(STAMP, "w").write(lm); log(f"downloaded {os.path.getsize(FILE):,} bytes ({lm})")
    t = pq.read_table(FILE, columns=COLS, filters=[("route_id", "in", ["1", "52"])])
    today = (dt.datetime.now(ET) - dt.timedelta(hours=3)).date()
    dates = sorted(set(t.column("service_date").to_pylist()))
    for d in dates:
        dd = d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)[:10])
        if dd >= today: continue                      # only completed service days go into the history
        part = t.filter(pc.equal(t.column("service_date"), d))
        pq.write_table(part, os.path.join(HIST, f"lamp_bus_events_{dd.isoformat()}.parquet"))
    log(f"history updated for {len([d for d in dates])} dates in file")
    events = os.path.join(r"C:\dev\rel_trackB\events")
    have = sorted(f[12:22] for f in os.listdir(events) if f.startswith("stop_events_")) if os.path.isdir(events) else []
    hist = {f[16:26] for f in os.listdir(HIST)}
    run = [d for d in have if d in hist]
    if run:
        subprocess.run([sys.executable, os.path.join(HERE, "pilot_instrument_check.py"), "--lamp", HIST, "--date", *run],
                       cwd=HERE, check=False)
        log(f"instrument check run for {run}")
    else:
        log("no Track B day with both stop events and agency history yet")

if __name__ == "__main__":
    main()
