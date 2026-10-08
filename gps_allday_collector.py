#!/usr/bin/env python3
"""
gps_allday_collector.py -- Track B (clean-sheet study) instrument: MBTA vehicle positions, ALL DAY, every service day.

Council_Clean_Sheet_Study_Design_2026-10-08.md: "Measure position, not prediction ... collect VehiclePositions for the study
routes continuously, every service day, every ~15-30 s ... reconstruct stop arrival and departure times." No windows, no
fidelity filter. Arrival/departure reconstruction is a separate step (positions -> stop events); this script only records.

Data use: MassDOT Developers License (2009, current) -- see Data_License_Check_2026-10-08.md. "Respect resources":
  * ONE request per poll for all study routes (filter[route]=1,52), sparse fieldsets, If-Modified-Since (304s don't count);
  * ~20 s cadence (~3 requests/min) -- with Track A's windowed collectors the peak stays < 20 req/min (the no-key limit);
  * uses a free MBTA API key if MBTA_API_KEY is set in the environment (recommended; Claude can't create accounts).

Output:
  raw (local, not OneDrive-synced):  C:\\dev\\rel_trackB\\raw\\gps_<service_date>.csv   (a row only when a vehicle's report changes)
  daily archive (synced):            <project>\\trackB_gps_daily\\gps_<service_date>.csv.gz  (written at day rollover)
  heartbeat:                          <project>\\trackB_status.json   (every ~5 min: last poll, rows today, errors)
Service date = Eastern-time date of (now - 3 h), so post-midnight trips stay on the day they belong to.
Runs indefinitely; the scheduled task re-launches it if it stops (multiple instances ignored).
Keeps the PC from idle-sleeping while it runs (SetThreadExecutionState); pass --allow-sleep to disable.

USAGE:  python gps_allday_collector.py [--routes 1,52] [--poll 20] [--allow-sleep] [--once]
"""
import argparse, csv, ctypes, datetime as dt, gzip, json, os, shutil, sys, time
from zoneinfo import ZoneInfo
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = r"C:\dev\rel_trackB\raw"
ARCHIVE_DIR = os.path.join(HERE, "trackB_gps_daily")
STATUS = os.path.join(HERE, "trackB_status.json")
LOG = r"C:\dev\rel_trackB\collector.log"
URL = "https://api-v3.mbta.com/vehicles"
ET = ZoneInfo("America/New_York")
FIELDS = "current_status,current_stop_sequence,latitude,longitude,bearing,speed,direction_id,updated_at"
COLS = ["poll_utc", "service_date", "vehicle_id", "route_id", "trip_id", "stop_id", "current_stop_sequence",
        "current_status", "latitude", "longitude", "bearing", "speed", "direction_id", "updated_at"]

def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")

def service_date(now_utc):
    return (now_utc.astimezone(ET) - dt.timedelta(hours=3)).date().isoformat()

def keep_awake(on):
    try:  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED  (prevents idle sleep while running; display may still sleep)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass

def archive_previous_days(today):
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    for fn in sorted(os.listdir(RAW_DIR)):
        if fn.startswith("gps_") and fn.endswith(".csv") and fn[4:14] < today:
            src = os.path.join(RAW_DIR, fn); dst = os.path.join(ARCHIVE_DIR, fn + ".gz")
            if not os.path.exists(dst):
                with open(src, "rb") as a, gzip.open(dst, "wb") as b:
                    shutil.copyfileobj(a, b)
                log(f"archived {fn} -> {dst}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="1,52"); ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--allow-sleep", action="store_true"); ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    sess = requests.Session()
    hdr = {"accept": "application/vnd.api+json"}
    key = os.environ.get("MBTA_API_KEY", "").strip()
    if key: hdr["x-api-key"] = key
    params = {"filter[route]": a.routes, "fields[vehicle]": FIELDS, "page[limit]": "500"}
    last_mod, last_seen = None, {}           # vehicle_id -> updated_at last written
    stats = {"started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "polls": 0, "not_modified": 0,
             "errors": 0, "rows_today": 0, "service_date": None, "api_key": bool(key)}
    if not a.allow_sleep: keep_awake(True)
    log(f"start routes={a.routes} poll={a.poll}s api_key={'yes' if key else 'no'}")
    last_status = 0.0
    try:
        while True:
            t0 = time.time(); now = dt.datetime.now(dt.timezone.utc); sd = service_date(now)
            if sd != stats["service_date"]:
                archive_previous_days(sd); stats["service_date"] = sd; stats["rows_today"] = 0; last_seen.clear()
            h = dict(hdr)
            if last_mod: h["if-modified-since"] = last_mod
            try:
                try:   # 10 s timeout + one immediate retry (council 2026-10-08: transient SSL drops cost a 340 s gap)
                    r = sess.get(URL, params=params, headers=h, timeout=10)
                except requests.RequestException as e1:
                    log(f"poll retry after: {e1.__class__.__name__}"); stats["retries"] = stats.get("retries", 0) + 1
                    r = sess.get(URL, params=params, headers=h, timeout=10)
                stats["polls"] += 1
                if r.status_code == 304:
                    stats["not_modified"] += 1
                elif r.status_code == 200:
                    last_mod = r.headers.get("last-modified", last_mod)
                    out = []
                    for v in r.json().get("data", []):
                        at = v.get("attributes", {}); rel = v.get("relationships", {})
                        rid = lambda k: ((rel.get(k) or {}).get("data") or {}).get("id", "")
                        vid = v.get("id", ""); upd = at.get("updated_at", "")
                        if not vid or last_seen.get(vid) == upd: continue
                        last_seen[vid] = upd
                        out.append([now.isoformat(timespec="seconds"), sd, vid, rid("route"), rid("trip"), rid("stop"),
                                    at.get("current_stop_sequence"), at.get("current_status"), at.get("latitude"),
                                    at.get("longitude"), at.get("bearing"), at.get("speed"), at.get("direction_id"), upd])
                    if out:
                        fn = os.path.join(RAW_DIR, f"gps_{sd}.csv"); new = not os.path.exists(fn)
                        with open(fn, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            if new: w.writerow(COLS)
                            w.writerows(out)
                        stats["rows_today"] += len(out)
                else:
                    stats["errors"] += 1; log(f"HTTP {r.status_code} remaining={r.headers.get('x-ratelimit-remaining')}")
                    if r.status_code == 429: time.sleep(60)
            except Exception as e:
                stats["errors"] += 1; log(f"poll error: {e}")
            if time.time() - last_status > 300 or a.once:
                stats["last_poll_utc"] = now.isoformat(timespec="seconds"); last_status = time.time()
                with open(STATUS + ".tmp", "w", encoding="utf-8") as f: json.dump(stats, f, indent=1)
                os.replace(STATUS + ".tmp", STATUS)   # atomic: readers never see a half-written file
            if a.once: break
            time.sleep(max(1.0, a.poll - (time.time() - t0)))
    finally:
        keep_awake(False); log("stop")

if __name__ == "__main__":
    sys.exit(main())
