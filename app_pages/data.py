import pandas as pd
import streamlit as st

from qc import metrics, ui

c = ui.ctx()
a = c.a
h = c.data.health
s = c.settings

st.title("Data & export")
ui.scope_line(c)

# ---- Data quality ----------------------------------------------------------------
st.markdown("##### Data quality")
k = st.columns(5)
k[0].metric("Rows in file", f"{h['rows_in_file']:,}", border=True)
k[1].metric("Modules", f"{h['modules']:,}", border=True)
k[2].metric("Measurement runs", f"{h['module_runs']:,}", f"{h['repeat_runs']:,} re-measurements",
            delta_color="off", border=True)
inc_runs = int((a.runs["MissingCorners"] > 0).sum())
k[3].metric("Runs with a missing corner", f"{inc_runs:,}", "in current filters", delta_color="off", border=True)
k[4].metric("Duplicate rows removed", f"{h.get('duplicate_rows_removed', 0):,}", border=True)

notes = []
if h.get("duplicate_rows_removed"):
    notes.append(("warn", f"**{h['duplicate_rows_removed']} exact duplicate row(s)** were dropped. Left in, a "
                          "duplicated corner looks like the start of a new run and creates a fake, incomplete re-measurement."))
for pair in h.get("similar_part_ids", []):
    merged = h.get("lookalike_ids_merged", 0) > 0
    notes.append(("info" if merged else "warn",
                  f"Part IDs **{pair[0]}** and **{pair[1]}** differ only by letter O vs digit 0. "
                  + ("They are merged into one module." if merged else
                     "Probably the same module scanned or typed differently; it is counted twice. "
                     "Use *Advanced → Merge look-alike Part IDs* in the sidebar to merge them.")))
if h.get("zero_readings"):
    feats = ", ".join(f"`{f}` ({n})" for f, n in h.get("zero_reading_features", {}).items())
    notes.append(("warn", f"**{h['zero_readings']} reading(s) with X, Y and Z all exactly 0** ({feats}): almost "
                          "certainly failed reads, not perfect corners. "
                          + ("They are treated as **missing corners**." if h.get("zero_as_missing") else
                             "They are currently used as real measurements (see *Advanced* in the sidebar).")))
if h.get("dst_ambiguous_rows") or h.get("dst_nonexistent_rows"):
    notes.append(("info", f"Daylight-saving change in the export time zone: {h.get('dst_ambiguous_rows', 0)} row(s) in the "
                          f"repeated autumn hour (read as summer time) and {h.get('dst_nonexistent_rows', 0)} in the "
                          "skipped spring hour (moved forward). None were dropped."))
if h.get("rows_without_valid_time"):
    notes.append(("bad", f"**{h['rows_without_valid_time']} row(s) without a readable time** were left out."))
if h.get("rows_unmapped_feature"):
    feats = ", ".join(f"`{f}` ({n})" for f, n in h.get("unmapped_features", {}).items())
    notes.append(("warn", f"**{h['rows_unmapped_feature']} row(s)** have a feature name that is not one of the four "
                          f"corners and were ignored: {feats}"))
if h.get("rows_with_non_numeric_deviation"):
    notes.append(("warn", f"{h['rows_with_non_numeric_deviation']} deviation value(s) were not numbers and count as missing."))
if h.get("corner_measured_twice_in_a_run"):
    notes.append(("info", f"{h['corner_measured_twice_in_a_run']} time(s) a corner appeared twice within one run; the last value was used."))
lim = h.get("limits_in_file", {})
if lim:
    txt = " · ".join(f"{ax} {lo:+g} / {hi:+g} mm" for ax, (lo, hi) in lim.items())
    notes.append(("info", f"Limits found in the file: {txt}. The X/Y tolerance in the sidebar is ±{s.spec_limit:g} mm."))
notes.append(("info", f"Times converted from **{st.session_state.get('sb_src_tz', 'Europe/Berlin')}** to "
                      f"**{st.session_state.get('sb_disp_tz', 'America/Mexico_City')}**; calendar weeks are ISO weeks "
                      "of the converted time and carry the year (e.g. 2026-CW43)."))
with st.container(border=True):
    ui.findings_box([{"level": lvl, "text": t} for lvl, t in notes])

with st.expander("Columns used", icon=":material/view_column:"):
    st.json(h.get("columns_used", {}))

# ---- Module report ---------------------------------------------------------------
st.markdown("##### All measurements in the current filters")
report = a.runs.drop(columns=["_mod_key", "BaseKey"], errors="ignore")
order = ["Date", "CalendarWeek", "PartID", "BatteryType", "RunNum", "FL_X", "FL_Y", "FR_X", "FR_Y",
         "RL_X", "RL_Y", "RR_X", "RR_Y", "CornersOutOfSpec", "MissingCorners", "Status"]
report = report[[col for col in order if col in report.columns]]
st.dataframe(
    ui.status_styler(report, s.spec_limit), hide_index=True, width="stretch", height=420,
    column_config={
        "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
        "CalendarWeek": "Week", "BatteryType": "Type", "RunNum": "Run",
        "CornersOutOfSpec": "Corners out", "MissingCorners": "Missing",
    },
)

# ---- Export ----------------------------------------------------------------------
st.markdown("##### Excel report")
st.caption("Everything above and on the other pages, for the current filters and rules.")


@st.cache_data(show_spinner="Building the Excel file…", max_entries=4)
def build_report(runs, weekly, pareto, cap, sq, vec, comp_rows, offsets, sims, kpi):
    sheets = {
        "Module_Summary_Report": runs,
        "First_Run_Quality_Summary": kpi,
        "Weekly_FPY_Trend": weekly,
        "Failure_Pareto": pareto,
        "Corner_Capability": cap,
        "Squareness_Analysis": sq,
        "Vector_Drift_Analysis": vec,
        "Compensation_Summary": comp_rows,
        "Corner_Offsets": offsets,
        "Simulated_Results": sims,
    }
    return ui.excel_report(sheets)


kpi = metrics.kpi_summary_table(a.first, s.exclude_incomplete)
sq_cols = ["Date", "CalendarWeek", "PartID", "RunNum", "BatteryType", "Status", "Diag1", "Diag2", "DeltaDiag",
           "WidthDelta", "LengthDelta", "AngleDevFL", "SquarenessStatus", "RootCause", "PlacementMax",
           "ShapeMax", "PassIfSquare", "NokCause"]
vec_cols = ["Date", "CalendarWeek", "PartID", "BatteryType", "Centroid_X", "Centroid_Y",
            "Vector_Magnitude", "Rotation_Angle", "Status"]
exp = st.session_state.get("_comp_export", {})
comp_rows, offsets, sims = [], [], []
for t, e in exp.items():
    tb = e["table"].copy()
    tb.insert(0, "BatteryType", t)
    tb.insert(1, "Window", e["window"])
    tb.insert(2, "Modules", e["modules"])
    tb["SafetyMargin_mm"] = e["margin_mm"]
    comp_rows.append(tb)
    o = e["offsets"].copy()
    o.insert(1, "Setting", e["inspected"])
    offsets.append(o)
    sims.append(e["sim"].drop(columns=["_mod_key"], errors="ignore"))
comp_df = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
off_df = pd.concat(offsets, ignore_index=True) if offsets else pd.DataFrame()
sim_df = pd.concat(sims, ignore_index=True) if sims else pd.DataFrame()

xls = build_report(
    report, a.weekly, a.pareto,
    a.capability, a.squareness[[col for col in sq_cols if col in a.squareness.columns]],
    a.vec[vec_cols], comp_df, off_df, sim_df, kpi,
)
st.download_button("Download Excel report", xls, file_name="Corner_QC_Report_MX_Time.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   icon=":material/download:", type="primary")
if not exp:
    st.caption("Open the **Compensation** page once to include its results in the report.")
