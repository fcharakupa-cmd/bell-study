# The Bell Study

**Can a student count on the bus to make first bell — and to get home after dismissal?**
A pre-registered, GPS-based study of two school trips — a frequent route (MBTA Route 1, Cambridge, MA) and an infrequent route (MBTA Route 52, Newton, MA) — 2026–27 school year.
Part of the *Transit Reliability Series* by Faith Charakupa — independent analysis.

> **Status: pilot (instrument shakedown).** No results are computed or published before the baseline. The pre-registration will be
> frozen on OSF before the baseline begins (target 2026-10-21); this repository will be tagged at that moment so the analysis code
> can be checked against the registered rules. Data and results are published after the baseline.

## What's here
| File | Purpose |
|---|---|
| `gps_allday_collector.py` | Records MBTA vehicle positions for Routes 1 and 52 all day (~20 s; one request per poll, sparse fields, `If-Modified-Since`). |
| `reconstruct_stop_events.py` | Rebuilds each bus's stop arrival/departure from positions: first `STOPPED_AT` report, else the interpolated passage along the GTFS shape (reports ≤ 90 s apart); departure = crossing of stop + 15 m. Prints instrument metrics only. |
| `pilot_instrument_check.py` | Instrument acceptance: coverage vs the agency record (≥ 97%), timing vs TransitMaster stop crossings (|bias| < 0.5 min, SD < 0.5 min), capture independent of lateness, schedule coverage. |
| `trackB_lamp_refresh.py` | Daily refresh of MBTA LAMP bus events (the agency reference) with a per-day history, then runs the instrument check. |

## Data sources and attribution
- MBTA V3 API (`api-v3.mbta.com`) vehicle positions; MBTA static GTFS; MBTA LAMP public exports (`performancedata.mbta.com`).
- **Data provided by MassDOT** and used under the **MassDOT Developers License Agreement**. This project is not affiliated with or
  endorsed by MassDOT, MBTA or DART, and makes no representation on their behalf.
- Data files are not committed here (see `.gitignore`).

## Running
Python 3.11+; `pip install -r requirements.txt`. Place `MBTA_GTFS.zip` (current MBTA GTFS) next to the scripts. Paths for raw data,
events and agency history default to `C:\dev\rel_trackB\` and can be changed in each script. Set `MBTA_API_KEY` to use a free MBTA
API key.

## License
Code: MIT (see `LICENSE`). Data remains subject to the MassDOT Developers License.
