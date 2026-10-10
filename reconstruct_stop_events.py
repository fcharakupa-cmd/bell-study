#!/usr/bin/env python3
"""
reconstruct_stop_events.py -- The Bell Study (Track B): GPS vehicle positions -> stop arrival/departure events.

Implements the frozen rules of BELL_Study_Preregistration (section 3.2):
  1. ARRIVAL at stop s for trip t = time (vehicle `updated_at`) of the first report with current_status = STOPPED_AT at s.
  2. If there is none, PASSAGE = linear interpolation of the time the vehicle crossed s along the GTFS shape, between the
     last report before and the first report after s, provided those two reports are <= 90 s apart; otherwise MISSING.
  3. DEPARTURE from s = interpolated time the vehicle crossed (s + 15 m) along the shape (bracketing reports <= 90 s apart);
     if that can't be interpolated, the time of the first report after arrival showing the vehicle beyond s ("coarse").
     For a passage, departure = arrival = the passage time.
  4. Trips are matched to the date's GTFS version by trip_id; unmatched trips are set aside and counted.

Geometry: each report and each stop is projected onto the trip's GTFS shape (local metric projection). Projection is kept
monotonic along the trip (a report may not move more than 50 m backwards), and reports more than 60 m from the shape are
treated as off-shape (ignored). Stop positions are projected in stop-sequence order with the same monotonic rule.

This script outputs EVENTS and INSTRUMENT METRICS ONLY (coverage, method mix, bracket gaps). It deliberately prints no
delay, on-time, wait or probability figures: during the pilot the pre-registration forbids computing outcome estimands.

USAGE:  python reconstruct_stop_events.py [--date YYYY-MM-DD ... | --yesterday] [--raw-dir DIR] [--out-dir DIR]
        (scheduled nightly as task TrackB_Reconstruct, 05:15 CT, with --yesterday)
Default: every raw day file present. Output: <out-dir>/stop_events_<date>.csv + <project>/trackB_pilot_metrics/<date>.json
"""
import argparse, csv, datetime as dt, glob, gzip, io, json, os, sys, zipfile
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

if sys.stdout is not None: sys.stdout.reconfigure(encoding="utf-8")   # pythonw under Task Scheduler has no console
HERE = os.path.dirname(os.path.abspath(__file__))
ET = ZoneInfo("America/New_York")
GTFS_CANDIDATES = ["MBTA_GTFS.zip", "MBTA_GTFS_archive_20260821.zip"]
ROUTES = {"1", "52"}
MAX_GAP_S, BACK_TOL_M, OFF_SHAPE_M, DEPART_OFFSET_M = 90.0, 50.0, 60.0, 15.0
ANCHOR_STOPS = {"102", "72", "85371", "84921"}   # Arm A (Route 1) + Arm B (Route 52, Oak Hill)

# ---------------------------------------------------------------------------------------------- GTFS
class Feed:
    def __init__(self, path):
        self.path = path; z = zipfile.ZipFile(path)
        rd = lambda n, **k: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, **k)
        self.cal = rd("calendar.txt"); self.cd = rd("calendar_dates.txt")
        self.trips = rd("trips.txt"); self.trips = self.trips[self.trips.route_id.isin(ROUTES)].set_index("trip_id")
        st = rd("stop_times.txt", usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "checkpoint_id"])
        self.st = st[st.trip_id.isin(self.trips.index)].copy()
        self.st["seq"] = self.st.stop_sequence.astype(int)
        stops = rd("stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"]).set_index("stop_id")
        self.stops = stops.astype({"stop_lat": float, "stop_lon": float})
        sh = rd("shapes.txt"); sh = sh[sh.shape_id.isin(set(self.trips.shape_id))].copy()
        sh["seq"] = sh.shape_pt_sequence.astype(int)
        self.shapes = {k: g.sort_values("seq")[["shape_pt_lat", "shape_pt_lon"]].astype(float).values for k, g in sh.groupby("shape_id")}
        self.lo = self.cal.start_date.min(); self.hi = self.cal.end_date.max()
        self._geom = {}
    def covers(self, ymd): return self.lo <= ymd <= self.hi
    def active(self, ymd):
        d = dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
        day = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][d.weekday()]
        on = set(self.cal[(self.cal.start_date <= ymd) & (self.cal.end_date >= ymd) & (self.cal[day] == "1")].service_id)
        x = self.cd[self.cd.date == ymd]
        return (on | set(x[x.exception_type == "1"].service_id)) - set(x[x.exception_type == "2"].service_id)
    def geom(self, shape_id):
        """Shape polyline in local meters + cumulative distance."""
        if shape_id not in self._geom:
            pts = self.shapes[shape_id]; lat0 = pts[:, 0].mean(); kx = 111320.0 * np.cos(np.radians(lat0)); ky = 110540.0
            xy = np.c_[(pts[:, 1]) * kx, (pts[:, 0]) * ky]
            seg = np.diff(xy, axis=0); L = np.hypot(seg[:, 0], seg[:, 1]); cum = np.r_[0, np.cumsum(L)]
            self._geom[shape_id] = (xy, seg, L, cum, kx, ky)
        return self._geom[shape_id]

def project(g, lat, lon):
    """All-segment projections of points -> (along[n_pts, n_seg], lateral[n_pts, n_seg])."""
    xy, seg, L, cum, kx, ky = g
    P = np.c_[np.asarray(lon) * kx, np.asarray(lat) * ky]
    A = xy[:-1]; Ls = np.where(L > 0, L, 1e-9)
    t = np.clip(((P[:, None, 0] - A[None, :, 0]) * seg[None, :, 0] + (P[:, None, 1] - A[None, :, 1]) * seg[None, :, 1]) / Ls**2, 0, 1)
    qx = A[None, :, 0] + t * seg[None, :, 0]; qy = A[None, :, 1] + t * seg[None, :, 1]
    lat_d = np.hypot(P[:, None, 0] - qx, P[:, None, 1] - qy); along = cum[None, :-1] + t * L[None, :]
    return along, lat_d

def monotonic_along(along, lat_d, back_tol=BACK_TOL_M, off=OFF_SHAPE_M):
    """Pick, point by point, the nearest on-shape projection that doesn't go backwards more than back_tol."""
    out = np.full(along.shape[0], np.nan); lat_out = np.full(along.shape[0], np.nan); prev = -np.inf
    for i in range(along.shape[0]):
        ok = (lat_d[i] <= off) & (along[i] >= prev - back_tol)
        if not ok.any(): continue
        cand = np.where(ok)[0]
        j = cand[np.argmin(along[i, cand] - (prev if np.isfinite(prev) else 0) + lat_d[i, cand] * 2)] if np.isfinite(prev) else cand[np.argmin(lat_d[i, cand])]
        out[i] = along[i, j]; lat_out[i] = lat_d[i, j]; prev = max(prev, out[i])
    return out, lat_out

def crossing(times, along, target, after=None):
    """Interpolated time (epoch s) at which `along` first reaches `target`; bracket gap must be <= MAX_GAP_S."""
    for i in range(len(along) - 1):
        a0, a1 = along[i], along[i + 1]
        if np.isnan(a0) or np.isnan(a1) or (after is not None and times[i + 1] < after): continue
        if a0 < target <= a1:
            gap = times[i + 1] - times[i]
            if gap > MAX_GAP_S: return None, gap
            f = (target - a0) / (a1 - a0) if a1 > a0 else 0.0
            return times[i] + f * gap, gap
    return None, None

# ---------------------------------------------------------------------------------------------- per day
def sched_epoch(service_date, hms):
    h, m, s = (int(x) for x in hms.split(":"))
    noon = dt.datetime.combine(dt.date.fromisoformat(service_date), dt.time(12), tzinfo=ET)   # GTFS time = noon - 12h + t
    return (noon - dt.timedelta(hours=12) + dt.timedelta(hours=h, minutes=m, seconds=s)).timestamp()

def iso(ep): return "" if ep is None or (isinstance(ep, float) and np.isnan(ep)) else dt.datetime.fromtimestamp(ep, tz=ET).isoformat(timespec="seconds")

def run_day(path, feeds, out_dir, metrics_dir):
    sd = os.path.basename(path)[4:14]; ymd = sd.replace("-", "")
    raw = pd.read_csv(path, dtype=str)
    raw = raw[raw.route_id.isin(ROUTES) & raw.trip_id.notna() & raw.latitude.notna()].copy()
    raw["t"] = (pd.to_datetime(raw.updated_at, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)   # unit-safe epoch seconds
    raw = raw.drop_duplicates(["vehicle_id", "updated_at"]).sort_values("t")
    feed = next((f for f in feeds if f.covers(ymd) and (set(raw.trip_id) & set(f.trips.index))), None)
    if feed is None: print(f"[{sd}] no GTFS version covers this date with matching trips — skipped"); return None
    active = feed.active(ymd)
    sched_trips = feed.trips[feed.trips.service_id.isin(active)]
    rows, m = [], {"service_date": sd, "gtfs": os.path.basename(feed.path), "raw_reports": int(len(raw)),
                   "trips_observed": 0, "trips_unmatched": 0, "events": {"stopped_at": 0, "interpolated": 0, "missing": 0},
                   "departure": {"interpolated": 0, "coarse": 0, "none": 0}, "bracket_gap_s_p95": None,
                   "anchor": {s: {"events": 0, "missing": 0} for s in ANCHOR_STOPS}, "offshape_reports": 0,
                   "offshape_by_location": {"near_terminal": 0, "mid_route": 0}, "offshape_mid_route_sample": []}
    gaps = []
    for trip, g in raw.groupby("trip_id"):
        if trip not in feed.trips.index: m["trips_unmatched"] += 1; continue
        veh = g.vehicle_id.value_counts().idxmax(); g = g[g.vehicle_id == veh]
        tr = feed.trips.loc[trip]; geo = feed.geom(tr.shape_id)
        times = g.t.values
        al, ld = project(geo, g.latitude.astype(float).values, g.longitude.astype(float).values)
        along, lat_used = monotonic_along(al, ld); m["offshape_reports"] += int(np.isnan(along).sum())
        if np.isfinite(along).sum() < 2: continue
        m["trips_observed"] += 1
        stp = feed.st[feed.st.trip_id == trip].sort_values("seq")
        # Off-shape detector (reporting only; 2026-10-10, external-review follow-up): off-shape reports within 400 m of the trip's
        # first or last stop are layover/turnaround; mid-route clusters can mean a detour or a vehicle logged into the wrong trip.
        off = np.where(np.isnan(along))[0]
        if len(off):
            ends = feed.stops.loc[[stp.stop_id.iloc[0], stp.stop_id.iloc[-1]], ["stop_lat", "stop_lon"]].values.astype(float)
            la = g.latitude.astype(float).values[off]; lo = g.longitude.astype(float).values[off]
            dkm = np.min([np.hypot((la - e[0]) * 111.0, (lo - e[1]) * 111.0 * np.cos(np.radians(e[0]))) for e in ends], axis=0)
            near = dkm <= 0.4
            m["offshape_by_location"]["near_terminal"] += int(near.sum()); m["offshape_by_location"]["mid_route"] += int((~near).sum())
            for k in np.where(~near)[0][:3]:
                if len(m["offshape_mid_route_sample"]) < 25:
                    m["offshape_mid_route_sample"].append({"trip_id": str(trip), "vehicle_id": str(veh),
                        "time": pd.Timestamp(float(times[off[k]]), unit="s", tz="UTC").tz_convert(ET).isoformat(timespec="seconds"),
                        "lat": round(float(la[k]), 5), "lon": round(float(lo[k]), 5), "km_from_terminal": round(float(dkm[k]), 2)})
        s_al, _ = monotonic_along(*project(geo, feed.stops.loc[stp.stop_id, "stop_lat"].values, feed.stops.loc[stp.stop_id, "stop_lon"].values),
                                  back_tol=5.0, off=200.0)
        status = g.current_status.values; stop_ids = g.stop_id.values
        amin, amax = np.nanmin(along), np.nanmax(along)                 # stretch of route this vehicle was observed covering
        for (_, srow), D in zip(stp.iterrows(), s_al):
            s = srow.stop_id; sch = sched_epoch(sd, srow.arrival_time)
            hit = np.where((status == "STOPPED_AT") & (stop_ids == s))[0]
            if not len(hit) and (not np.isfinite(D) or D < amin or D > amax):
                m["not_attempted"] = m.get("not_attempted", 0) + 1            # stop outside the observed stretch: not a miss
                continue
            arr, dep, method, dep_m, gap, t_first = None, None, "missing", "none", None, None
            if len(hit):
                # Rule 1 (pre-registration §3.2, revised 2026-10-10, D-149): arrival = the vehicle's last report BEFORE its first
                # STOPPED_AT at the stop when the two are <= 90 s apart (STOPPED_AT posts ~20 s after the agency's recorded arrival);
                # otherwise the first STOPPED_AT. The first STOPPED_AT time is kept (column first_stopped_at) for the sensitivity.
                i0 = hit[0]; t_first = float(times[i0]); arr = t_first; method = "stopped_at"
                if i0 > 0 and t_first - float(times[i0 - 1]) <= MAX_GAP_S: arr = float(times[i0 - 1])
                if np.isfinite(D):
                    dep, g2 = crossing(times, along, D + DEPART_OFFSET_M, after=t_first)
                    if dep is not None: dep_m = "interpolated"
                if dep is None:
                    later = np.where((times > t_first) & ((stop_ids != s) | (np.nan_to_num(along, nan=-1) > (D if np.isfinite(D) else 1e12) + DEPART_OFFSET_M)))[0]
                    if len(later): dep = float(times[later[0]]); dep_m = "coarse"
            else:
                arr, gap = crossing(times, along, D)
                if arr is not None: dep = arr; method = "interpolated"; dep_m = "interpolated"; gaps.append(gap)
                else: m.setdefault("missing_reason", {}).setdefault("gap_over_90s" if gap else "no_bracket", 0)
                if arr is None: m["missing_reason"]["gap_over_90s" if gap else "no_bracket"] += 1
            m["events"][method] += 1; m["departure"][dep_m] += 1
            if s in ANCHOR_STOPS:
                m["anchor"][s]["events" if method != "missing" else "missing"] += 1
            rows.append([sd, tr.route_id, trip, veh, tr.direction_id, s, srow.stop_sequence, srow.checkpoint_id if isinstance(srow.checkpoint_id, str) else "",
                         iso(sch), iso(arr), iso(dep), method, dep_m, "" if gap is None else round(gap, 1), iso(t_first)])
    m["trips_scheduled_active"] = int(len(sched_trips))
    m["trips_scheduled_seen"] = int(len(set(raw.trip_id) & set(sched_trips.index)))
    if gaps: m["bracket_gap_s_p95"] = round(float(np.percentile(gaps, 95)), 1)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"stop_events_{sd}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["service_date", "route_id", "trip_id", "vehicle_id", "direction_id", "stop_id", "stop_sequence", "checkpoint_id",
                    "scheduled_arrival", "event_arrival", "event_departure", "method", "departure_method", "bracket_gap_s", "first_stopped_at"])
        w.writerows(rows)
    with open(os.path.join(metrics_dir, f"{sd}.json"), "w", encoding="utf-8") as f: json.dump(m, f, indent=1)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", nargs="*"); ap.add_argument("--yesterday", action="store_true", help="previous service day only")
    ap.add_argument("--raw-dir", default=r"C:\dev\rel_trackB\raw")
    ap.add_argument("--out-dir", default=r"C:\dev\rel_trackB\events")
    ap.add_argument("--metrics-dir", default=os.path.join(HERE, "trackB_pilot_metrics"))
    a = ap.parse_args()
    feeds = [Feed(os.path.join(HERE, g)) for g in GTFS_CANDIDATES if os.path.exists(os.path.join(HERE, g))]
    files = sorted(glob.glob(os.path.join(a.raw_dir, "gps_*.csv")))
    if a.yesterday:
        a.date = [(dt.datetime.now(ET) - dt.timedelta(hours=3) - dt.timedelta(days=1)).date().isoformat()]
    if a.date: files = [f for f in files if os.path.basename(f)[4:14] in a.date]
    for f in files:
        m = run_day(f, feeds, a.out_dir, a.metrics_dir)
        if m:
            ev = m["events"]; tot = sum(ev.values())
            print(f"[{m['service_date']}] {m['gtfs']} | reports {m['raw_reports']:,} | trips observed {m['trips_observed']} "
                  f"(unmatched {m['trips_unmatched']}) | stop events {tot:,}: stopped_at {ev['stopped_at']:,}, interpolated "
                  f"{ev['interpolated']:,}, missing {ev['missing']:,} ({100*ev['missing']/max(1,tot):.1f}%) | departures {m['departure']} "
                  f"| bracket gap p95 {m['bracket_gap_s_p95']} s | off-shape reports {m['offshape_reports']} {m['offshape_by_location']} | not attempted {m.get('not_attempted',0)} | missing reasons {m.get('missing_reason',{})} | anchors {m['anchor']}")
    print("Instrument metrics only — no delay/on-time/wait figures are computed during the pilot (pre-registration §2.2).")

if __name__ == "__main__":
    main()
