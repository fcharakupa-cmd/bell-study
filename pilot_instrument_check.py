#!/usr/bin/env python3
"""
pilot_instrument_check.py -- The Bell Study (Track B): instrument acceptance checks (pre-registration section 3.3).

INSTRUMENT METRICS ONLY. Prints no delay distributions, on-time shares, waits or probabilities (pilot rule, section 2.2).
Schedule deviation from the agency record is used solely as a covariate in the capture-vs-lateness test (C3), as the
pre-registration requires; only the test statistics are reported.

Checks per service day (and pooled over the pilot):
  C1  Coverage vs the AGENCY RECORD: of the stop crossings the agency logged (LAMP bus events, TransitMaster actuals) at
      MBTA checkpoint stops on Routes 1/52 -- and separately at the anchor stops 102 and 72 -- the share for which Track B
      reconstructed an event.                                         Acceptance: >= 97%.
  C2  Timing agreement vs the agency record: Track B arrival - TransitMaster actual arrival (and departure), matched on
      (service_date, trip_id, stop_id): bias, SD, % within +/-1 min.     Acceptance: |bias| < 0.5 min and SD < 0.5 min.
      Same-feed cross-check (not acceptance): Track B arrival - LAMP's own GTFS-RT first-STOPPED_AT time.
  C3  Capture independent of lateness: logistic regression captured ~ agency-recorded schedule deviation at the crossing;
      odds ratio per +5 min and p-value.                              Acceptance: p > 0.05 and OR within 0.8-1.25.
  C4  Coverage vs the SCHEDULE: share of scheduled trips active that day seen anywhere in the raw positions (context;
      dropped trips and collector gaps are not separable without the agency record).

Inputs: Track B stop events (C:\\dev\\rel_trackB\\events\\stop_events_<date>.csv), raw positions, GTFS, and -- for C1-C3 --
LAMP_RECENT_Bus_Events.parquet (MBTA LAMP public export, refreshed daily) at --lamp (default C:\\dev\\rel_trackB\\lamp\\).
Without the LAMP file, only C4 runs and C1-C3 are reported as "pending reference data".

USAGE:  python pilot_instrument_check.py [--date YYYY-MM-DD ...] [--lamp PATH]
Writes trackB_pilot_metrics/instrument_check_<date>.json and trackB_pilot_metrics/INSTRUMENT_CHECK.md (pooled summary).
"""
import argparse, datetime as dt, glob, json, os, sys, zipfile, io
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = r"C:\dev\rel_trackB\events"; RAW = r"C:\dev\rel_trackB\raw"
METRICS = os.path.join(HERE, "trackB_pilot_metrics")
ROUTES = {"1", "52"}; ANCHORS = {"102", "72"}

def to_epoch(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)

def load_lamp(path, dates):
    """path = the LAMP RECENT parquet, or the per-day history folder written by trackB_lamp_refresh.py."""
    if not path or not os.path.exists(path): return None
    import pyarrow.parquet as pq
    cols = ["service_date", "trip_id", "stop_id", "route_id", "checkpoint_id", "plan_stop_departure_dt",
            "tm_actual_arrival_dt", "tm_actual_departure_dt", "gtfs_arrival_dt", "vehicle_label"]
    if os.path.isdir(path):
        files = [os.path.join(path, f"lamp_bus_events_{d}.parquet") for d in dates]
        files = [f for f in files if os.path.exists(f)]
        if not files: return None
        t = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    else:
        t = pq.read_table(path, columns=cols, filters=[("route_id", "in", sorted(ROUTES))]).to_pandas()
    t["service_date"] = pd.to_datetime(t.service_date).dt.date.astype(str)
    return t[t.service_date.isin(dates)].copy()

def logistic(x, y):
    """Minimal logistic regression (Newton) of y on x with intercept -> (beta1, se1)."""
    X = np.c_[np.ones_like(x), x]; b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]); g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g); b += step
        if np.abs(step).max() < 1e-8: break
    cov = np.linalg.inv(H + 1e-9 * np.eye(2)); return b[1], np.sqrt(cov[1, 1])

def check_day(sd, lamp, feed_trips_active, feed_trip_span):
    out = {"service_date": sd}
    ev_path = os.path.join(EVENTS, f"stop_events_{sd}.csv")
    ev = pd.read_csv(ev_path, dtype=str) if os.path.exists(ev_path) else pd.DataFrame()
    raw_path = os.path.join(RAW, f"gps_{sd}.csv")
    raw = pd.read_csv(raw_path, dtype=str) if os.path.exists(raw_path) else pd.DataFrame(columns=["trip_id"])
    # C4 schedule coverage (context)
    act = feed_trips_active(sd)
    if len(raw):   # only trips whose whole scheduled run falls inside the hours the collector was running
        t_raw = to_epoch(raw.updated_at); lo, hi = float(t_raw.min()), float(t_raw.max())
        act = {t for t, (s0, s1) in feed_trip_span(sd, act).items() if s0 >= lo and s1 <= hi}
    out["C4_scheduled_trips_in_span"] = int(len(act)); out["C4_seen_in_raw"] = int(len(act & set(raw.trip_id)))
    out["C4_share"] = round(out["C4_seen_in_raw"] / max(1, len(act)), 4) if act else None
    if lamp is None or lamp[lamp.service_date == sd].empty:
        out["C1_C3"] = "pending reference data (LAMP bus events not loaded for this date)"; return out
    L = lamp[(lamp.service_date == sd) & lamp.tm_actual_arrival_dt.notna() & lamp.checkpoint_id.notna()].copy()
    # restrict to the stretch of the day Track B was collecting (collector start/stop), so pre-start trips aren't "misses"
    if len(raw):
        t_raw = to_epoch(raw.updated_at); lo, hi = t_raw.min(), t_raw.max()
        L["ta"] = to_epoch(L.tm_actual_arrival_dt); L = L[(L.ta >= lo + 120) & (L.ta <= hi - 120)]
    good = ev[ev.method.isin(["stopped_at", "interpolated"])].copy() if len(ev) else ev
    key = ["trip_id", "stop_id"]
    M = L.merge(good[key + ["event_arrival", "event_departure", "method"]], on=key, how="left")
    M["captured"] = M.event_arrival.notna()
    out["C1_agency_crossings"] = int(len(M)); out["C1_captured"] = int(M.captured.sum())
    out["C1_coverage"] = round(M.captured.mean(), 4) if len(M) else None
    out["C1_anchor"] = {s: {"crossings": int((M.stop_id == s).sum()),
                            "coverage": round(M[M.stop_id == s].captured.mean(), 4) if (M.stop_id == s).any() else None} for s in ANCHORS}
    C = M[M.captured].copy()
    if len(C):
        d_arr = (to_epoch(C.event_arrival) - to_epoch(C.tm_actual_arrival_dt)) / 60
        C2 = {"n": int(len(C)), "arrival_bias_min": round(float(d_arr.mean()), 3), "arrival_sd_min": round(float(d_arr.std()), 3),
              "arrival_within_1min": round(float((d_arr.abs() <= 1).mean()), 4)}
        dd = (to_epoch(C.event_departure) - to_epoch(C.tm_actual_departure_dt)) / 60; dd = dd.dropna()
        if len(dd): C2.update({"departure_bias_min": round(float(dd.mean()), 3), "departure_sd_min": round(float(dd.std()), 3)})
        g = C[C.gtfs_arrival_dt.notna()]
        if len(g):
            dg = (to_epoch(g.event_arrival) - to_epoch(g.gtfs_arrival_dt)) / 60
            C2["same_feed_vs_LAMP_stopped_at"] = {"n": int(len(g)), "bias_min": round(float(dg.mean()), 3), "sd_min": round(float(dg.std()), 3)}
        out["C2"] = C2
    # C3 capture vs agency-recorded lateness (covariate only)
    dev = (to_epoch(M.tm_actual_arrival_dt) - to_epoch(M.plan_stop_departure_dt)) / 60
    ok = dev.notna() & dev.between(-30, 60)
    if ok.sum() > 30 and M.captured[ok].nunique() == 2:
        b, se = logistic((dev[ok] / 5).values, M.captured[ok].astype(float).values)
        z = b / se; from math import erf, sqrt; p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        out["C3"] = {"n": int(ok.sum()), "odds_ratio_per_5min": round(float(np.exp(b)), 3), "p_value": round(p, 4)}
    elif ok.sum():
        out["C3"] = {"n": int(ok.sum()), "note": "all crossings captured (or too few) — no capture/lateness variation to test"}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", nargs="*"); ap.add_argument("--lamp", default=r"C:\dev\rel_trackB\lamp\history")
    a = ap.parse_args()
    dates = a.date or sorted(os.path.basename(f)[12:22] for f in glob.glob(os.path.join(EVENTS, "stop_events_*.csv")))
    sys.path.insert(0, HERE); import reconstruct_stop_events as R
    feeds = [R.Feed(os.path.join(HERE, g)) for g in R.GTFS_CANDIDATES if os.path.exists(os.path.join(HERE, g))]
    def active_trips(sd):
        ymd = sd.replace("-", ""); f = next((f for f in feeds if f.covers(ymd)), None)
        return set(f.trips[f.trips.service_id.isin(f.active(ymd))].index) if f else set()
    def trip_span(sd, trips):
        ymd = sd.replace("-", ""); f = next((f for f in feeds if f.covers(ymd)), None)
        if not f: return {}
        s = f.st[f.st.trip_id.isin(trips)]
        g = s.groupby("trip_id").arrival_time.agg(["min", "max"])
        return {t: (R.sched_epoch(sd, r["min"]), R.sched_epoch(sd, r["max"])) for t, r in g.iterrows()}
    lamp = load_lamp(a.lamp, dates)
    os.makedirs(METRICS, exist_ok=True); res = []
    for sd in dates:
        r = check_day(sd, lamp, active_trips, trip_span); res.append(r)
        json.dump(r, open(os.path.join(METRICS, f"instrument_check_{sd}.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(r, indent=1))
    # pooled markdown summary
    lines = ["# Bell Study — pilot instrument check (auto-generated; instrument metrics only)", "",
             f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · pre-registration §3.3 acceptance_", "",
             "| Date | C4 sched. trips seen | C1 coverage (checkpoints) | C1 anchors 102 / 72 | C2 bias / SD (min) | C2 within ±1 | C3 OR per 5 min (p) |",
             "|---|---|---|---|---|---|---|"]
    for r in res:
        c2 = r.get("C2", {}); c3 = r.get("C3", {}); an = r.get("C1_anchor", {})
        lines.append(f"| {r['service_date']} | {r['C4_seen_in_raw']}/{r['C4_scheduled_trips_in_span']} | "
                     f"{r.get('C1_coverage', '—') if 'C1_coverage' in r else 'pending'} | "
                     f"{an.get('102', {}).get('coverage', '—')} / {an.get('72', {}).get('coverage', '—')} | "
                     f"{c2.get('arrival_bias_min', '—')} / {c2.get('arrival_sd_min', '—')} | {c2.get('arrival_within_1min', '—')} | "
                     f"{c3.get('odds_ratio_per_5min', '—')} ({c3.get('p_value', '—')}) |")
    lines += ["", "Acceptance (pre-registration §3.3): coverage ≥ 0.97 · |bias| < 0.5 and SD < 0.5 min · capture independent of lateness "
              "(p > 0.05, OR 0.8–1.25). C4 is context (dropped trips and collector gaps are not separable without the agency record)."]
    open(os.path.join(METRICS, "INSTRUMENT_CHECK.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("Instrument metrics only — no delay/on-time/wait figures are computed during the pilot (pre-registration §2.2).")

if __name__ == "__main__":
    main()
