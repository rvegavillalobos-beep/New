# Corner QC

Streamlit app for the first-measurement quality of battery modules at **ST020**: how far each corner lands from nominal, why modules fail, whether they are deformed, how the placement drifts, and what an X / Y / yaw compensation would do.

It is a rebuild of [RICKHARDV](https://github.com/rvegavillalobos-beep/RICKHARDV). The numbers match the original (checked automatically, see *Tests*). The layout is new, a few original bugs are fixed, and the compensation tools are deeper.

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or open the repo in GitHub Codespaces: the dev container installs everything and starts the app. Load your *Corner Cell Deviations* export (Excel or CSV), or press **Try it with demo data**.

## Pages

| Page | What it answers |
|---|---|
| **Overview** | FPY against target, per battery type and per week; automatic findings; which corner and axis cause NOK. |
| **Corners** | Where each corner lands (cloud inside the tolerance box), capability (mean, σ, Cpk) and whether each corner/axis has an *offset* (fixable by compensation) or *scatter* (not fixable by an offset). Deviation trends. |
| **Geometry** | Every module drawn against nominal with magnified deviations and tolerance zones; focus on one module to see its corners, squareness and run history. |
| **Squareness** | Deformed share, deformation pattern, a width-vs-length deformation map, and a module table (select a row to open it in Geometry). |
| **Drift** | Weekly path of the module center, weekly median offset and yaw, yaw per module. |
| **Compensation** | Median-based vs FPY-optimized vs your own X / Y / yaw, simulated FPY, a week-by-week **backtest**, a readiness checklist, FPY-vs-yaw and X/Y maps. |
| **Data & export** | Data-quality report (duplicates, look-alike IDs, failed reads, time-zone notes) and the full Excel report. |

Filters (battery type, calendar weeks) and rules (tolerance, FPY target, diagonal limit, incomplete-as-NOK) live in the sidebar and apply to every page.

## Compensation: median-based vs optimized

* **Median-based** (the original method) undoes the median center offset and the median yaw. It centers the *average* module.
* **Optimized** searches X, Y and yaw for the setting that passes the most modules. A module passes only if *all four* corners are in tolerance, so what matters is each module's worst corner. When corners are biased differently, that is not the same as centering the average, which is why small manual tweaks (especially in yaw) can beat the median. A *safety margin* makes it prefer settings where modules pass with room to spare.
* **Backtest**: in-window FPY is measured on the same modules the setting was computed from, so it is optimistic; with few modules, the optimizer partly fits noise. The backtest replays recent weeks as if the method had been in use (each week gets the setting computed from the weeks before it) and is the honest estimate of what to expect on the line. The page states which method held up better.

## Data handling

* **Time**: the export is read as `Europe/Berlin` and shown in `America/Mexico_City` (both configurable). Daylight-saving hours are handled instead of crashing or dropping rows. Calendar weeks carry the ISO year (`2026-CW43`), so January sorts after December.
* **Runs**: a part's run number increases when a corner already measured in the current run appears again; one history per part across all days.
* **First run**: each module's Run 1. With *Count incomplete first runs as NOK* on (default), a first run with a missing corner is a FAIL.
* **Battery type** is decided per part from all its features (Type M: `_dj` / `_di`; Type S: `_da`).
* **Cleaning** (all reported on *Data & export*): exact duplicate rows are dropped; readings with X, Y and Z all exactly 0 are treated as missing corners (switchable); Part IDs that differ only by letter O vs digit 0 are flagged and can be merged.

## Fixed from the original

* The app stopped with an error on any export containing the skipped spring daylight-saving hour in Germany, and lost rows in the repeated autumn hour.
* Time format was guessed from the first row: if the newest row had an "am" time, the PM rows were silently dropped.
* Calendar weeks without the year sorted CW01 before CW52 at the turn of the year.
* An exact duplicate row created a fake, incomplete re-measurement.
* All-zero failed reads were used as perfect measurements.
* Everything was recomputed row by row on every click; parsing and analysis are now vectorized and cached.

## Project layout

```
streamlit_app.py      entry point: sidebar, navigation
app_pages/            one file per page
qc/loading.py         reading the export, time zones, runs, data-quality report
qc/geometry.py        status, centroid / yaw, squareness, rigid compensation
qc/metrics.py         FPY, failure Pareto, capability, findings
qc/compensation.py    median-based, optimizer, backtest, readiness
qc/charts.py          Plotly figures
qc/analysis.py        settings + filters -> every table the pages show
qc/ui.py              Streamlit glue: caching, sidebar, shared state
qc/demo.py            synthetic export for the demo button and tests
tests/                pytest suite (incl. regression against the original logic)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
QC_REAL_FILE=/path/to/export.xlsx pytest   # also check a real export against the original logic
```
