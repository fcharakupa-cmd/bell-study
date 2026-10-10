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

if sys.stdout is not None: sys.stdout.reconfigure(encoding="utf-8")   # pythonw under Task Scheduler has no console
HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = r"C:\dev\rel_trackB\events"; RAW = r"C:\dev\rel_trackB\raw"
METRICS = os.path.join(HERE, "trackB_pilot_metrics")
ROUTES = {"1", "52"}; ANCHORS = {"102", "72", "85371", "84921"}   # Arm A + Arm B (Oak Hill)

def to_epoch(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)

# LAMP timestamps are tz-naive. Verified 2026-10-10 against Track B events (median event - LAMP, 1,188 matched crossings):
#   tm_actual_*_dt and gtfs_arrival_dt are Eastern local time (+0.3 min read as ET; +240 min if read as UTC);
#   plan_stop_departure_dt sits one UTC offset EARLIER than Eastern local (a LAMP export quirk): true ET = value - offset.
def lamp_epoch(s):
    loc = pd.to_datetime(s).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return (loc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)

def lamp_plan_epoch(s):
    loc = pd.to_datetime(s).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    off = loc.map(lambda x: x.utcoffset().total_seconds() if pd.notna(x) else np.nan)   # -14400 in EDT
    return (loc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1) - off

def load_lamp(path, dates):
    """path = the LAMP RECENT parquet, or the per-day history folder written by trackB_lamp_refresh.py."""
    if not path or not os.path.exists(path): return None
    import pyarrow.parquet as pq
    cols = ["service_date", "trip_id", "stop_id", "route_id", "checkpoint_id", "plan_stop_departure_dt",
            "tm_actual_arrival_dt", "tm_actual_departure_dt", "gtfs_arrival_dt", "vehicle_label", "stop_sequence"]
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

def c2_stats(C):
    """Timing agreement of Track B events vs TransitMaster actuals (and LAMP's own GTFS-RT time) for a set of crossings."""
    if not len(C): return None
    d_all = (to_epoch(C.event_arrival) - lamp_epoch(C.tm_actual_arrival_dt)) / 60
    # §3.3 (revised 2026-10-10, D-149): a match more than GROSS_MIN from the agency record is a MATCHING ERROR (two records of
    # different events), counted against its own limit (<= 0.5% of matches) and listed; bias and SD are computed on the rest.
    gross = d_all.abs() > GROSS_MIN; d_arr = d_all[~gross]
    s = {"n": int(len(C)), "matching_errors": int(gross.sum()), "matching_error_rate": round(float(gross.mean()), 4),
         "matching_error_list": [{"trip_id": str(r.trip_id), "stop_id": str(r.stop_id), "diff_min": round(float(x), 2)}
                                 for r, x in zip(C[gross].itertuples(), d_all[gross])],
         "arrival_bias_min": round(float(d_arr.mean()), 3), "arrival_sd_min": round(float(d_arr.std()), 3),
         "arrival_within_1min": round(float((d_arr.abs() <= 1).mean()), 4),
         "arrival_sd_min_all_matches": round(float(d_all.std()), 3)}
    dd = ((to_epoch(C.event_departure) - lamp_epoch(C.tm_actual_departure_dt)) / 60).dropna()
    if len(dd): s.update({"departure_bias_min": round(float(dd.mean()), 3), "departure_sd_min": round(float(dd.std()), 3)})
    g = C[C.gtfs_arrival_dt.notna()]
    if len(g):
        dg = (to_epoch(g.event_arrival) - lamp_epoch(g.gtfs_arrival_dt)) / 60
        s["same_feed_vs_LAMP_stopped_at"] = {"n": int(len(g)), "bias_min": round(float(dg.mean()), 3), "sd_min": round(float(dg.std()), 3)}
    return s

GROSS_MIN = 5.0          # minutes; matching-error threshold (§3.3, revised 2026-10-10)
POOL = []                # (deviation_min, captured) over all pilot days, for the pooled capture-vs-lateness test

def c3_test(dev, cap):
    ok = dev.notna() & dev.between(-30, 60)
    if ok.sum() > 30 and cap[ok].nunique() == 2:
        b, se = logistic((dev[ok] / 5).values, cap[ok].astype(float).values)
        z = b / se; from math import erf, sqrt; p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        return {"n": int(ok.sum()), "odds_ratio_per_5min": round(float(np.exp(b)), 3), "or_95ci": [round(float(lo), 3), round(float(hi), 3)],
                "p_value": round(p, 4), "miss_rate": round(float(1 - cap[ok].mean()), 4)}
    return {"n": int(ok.sum()), "note": "all crossings captured (or too few) — no capture/lateness variation to test",
            "miss_rate": round(float(1 - cap[ok].mean()), 4) if ok.sum() else None}

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
        L["ta"] = lamp_epoch(L.tm_actual_arrival_dt); L = L[(L.ta >= lo + 120) & (L.ta <= hi - 120)]
    good = ev[ev.method.isin(["stopped_at", "interpolated"])].copy() if len(ev) else ev
    key = ["trip_id", "stop_id"]
    M = L.merge(good[key + ["event_arrival", "event_departure", "method"]], on=key, how="left")
    M["captured"] = M.event_arrival.notna()
    # Acceptance scope (pre-registration §3.3, set 2026-10-10): anchor stops + mid-route checkpoints; each trip's first and last
    # stop (layover at the origin; the trip ends on arrival at the terminus) are reported separately, not used for acceptance.
    D = lamp[lamp.service_date == sd].copy(); D["seq"] = pd.to_numeric(D.stop_sequence, errors="coerce")
    ends = D.groupby("trip_id").seq.agg(["min", "max"])
    M["seq"] = pd.to_numeric(M.stop_sequence, errors="coerce")
    M = M.join(ends, on="trip_id"); M["trip_end"] = (M.seq == M["min"]) | (M.seq == M["max"])
    mid, tend = M[~M.trip_end], M[M.trip_end]
    out["C1_agency_crossings"] = int(len(mid)); out["C1_captured"] = int(mid.captured.sum())
    out["C1_coverage"] = round(mid.captured.mean(), 4) if len(mid) else None
    out["C1_trip_ends_reported_separately"] = {"crossings": int(len(tend)),
                                               "coverage": round(tend.captured.mean(), 4) if len(tend) else None}
    out["C1_anchor"] = {s: {"crossings": int((M.stop_id == s).sum()),
                            "coverage": round(M[M.stop_id == s].captured.mean(), 4) if (M.stop_id == s).any() else None} for s in ANCHORS}
    # Arm B anchors (85371/84921) are not TransitMaster checkpoints, so the agency record has no actual there. Basis instead:
    # trips the agency recorded as operated (an actual at any checkpoint) that are planned to serve the stop -> captured there?
    for s in ANCHORS:
        if out["C1_anchor"][s]["coverage"] is None:
            P = lamp[(lamp.service_date == sd) & (lamp.stop_id.astype(str) == s) & lamp.trip_id.isin(set(L.trip_id))][key].drop_duplicates()
            P = P.merge(good[key + ["event_arrival"]], on=key, how="left") if len(good) else P.assign(event_arrival=np.nan)
            out["C1_anchor"][s] = {"crossings": int(len(P)), "basis": "operated trips (no checkpoint at stop)",
                                   "coverage": round(P.event_arrival.notna().mean(), 4) if len(P) else None}
    c2 = c2_stats(mid[mid.captured]);  c2e = c2_stats(tend[tend.captured])
    if c2: out["C2"] = c2
    if c2e: out["C2_trip_ends_reported_separately"] = c2e
    # C3 capture vs agency-recorded lateness (covariate only), on the acceptance scope
    dev = (lamp_epoch(mid.tm_actual_arrival_dt) - lamp_plan_epoch(mid.plan_stop_departure_dt)) / 60
    out["C3"] = c3_test(dev, mid.captured)                       # per day: reported; acceptance is POOLED (§3.3, revised 2026-10-10)
    POOL.extend(zip(dev.tolist(), mid.captured.tolist()))
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
             "| Date | C4 sched. trips seen | C1 coverage (mid-route checkpoints) | C1 anchors 102 / 72 (Arm A) | C1 anchors 85371 / 84921 (Arm B) | C2 bias / SD (min) | C2 within ±1 | C2 matching errors | C3 OR per 5 min (p) — per day, context |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        c2 = r.get("C2", {}); c3 = r.get("C3", {}); an = r.get("C1_anchor", {})
        lines.append(f"| {r['service_date']} | {r['C4_seen_in_raw']}/{r['C4_scheduled_trips_in_span']} | "
                     f"{r.get('C1_coverage', '—') if 'C1_coverage' in r else 'pending'} | "
                     f"{an.get('102', {}).get('coverage', '—')} / {an.get('72', {}).get('coverage', '—')} | "
                     f"{an.get('85371', {}).get('coverage', '—')} / {an.get('84921', {}).get('coverage', '—')} | "
                     f"{c2.get('arrival_bias_min', '—')} / {c2.get('arrival_sd_min', '—')} | {c2.get('arrival_within_1min', '—')} | "
                     f"{c2.get('matching_errors', '—')} ({c2.get('matching_error_rate', '—')}) | "
                     f"{c3.get('odds_ratio_per_5min', '—')} ({c3.get('p_value', '—')}) |")
    # Pooled acceptance over all pilot days checked (§3.3, revised 2026-10-10, D-149)
    if POOL:
        P = pd.DataFrame(POOL, columns=["dev", "cap"]); pc3 = c3_test(P.dev, P.cap)
        mr = pc3.get("miss_rate")
        orv = pc3.get("odds_ratio_per_5min"); pv = pc3.get("p_value")
        pass_a = orv is not None and pv is not None and pv > 0.05 and 0.8 <= orv <= 1.25
        pass_b = mr is not None and mr <= 0.01
        n_match = sum(r.get("C2", {}).get("n", 0) for r in res); n_gross = sum(r.get("C2", {}).get("matching_errors", 0) for r in res)
        lines += ["", f"**Pooled over {len(res)} pilot day(s) (acceptance):** capture vs lateness: odds ratio per 5 min {orv} "
                  f"(95% CI {pc3.get('or_95ci', '—')}), p = {pv}, miss rate {mr} → "
                  + ("PASS (a: OR in band, p > 0.05)" if pass_a else ("PASS (b: miss rate ≤ 1% — worst-case effect on any share ≈ "
                     f"{100*mr:.1f} pts; OR reported)" if pass_b else "NOT MET")) +
                  f" · matching errors {n_gross} of {n_match} ({100*n_gross/max(1,n_match):.2f}%; limit 0.5%) → "
                  + ("PASS" if n_gross <= 0.005 * max(1, n_match) else "NOT MET")]
    lines += ["", "Acceptance (pre-registration §3.3, revised 2026-10-10): coverage ≥ 0.97 · |bias| < 0.5 and SD < 0.5 min on matches within "
              "5 min of the agency record; matching errors (> 5 min) ≤ 0.5% of matches, each listed · capture vs lateness POOLED over the "
              "pilot: (a) p > 0.05 and OR 0.8–1.25, or (b) miss rate ≤ 1% (worst-case bound), OR always reported · scored at the anchors and "
              "mid-route checkpoints; each trip's first/last stop is in the JSON, reported separately. C4 is context."]
    open(os.path.join(METRICS, "INSTRUMENT_CHECK.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("Instrument metrics only — no delay/on-time/wait figures are computed during the pilot (pre-registration §2.2).")

if __name__ == "__main__":
    main()
