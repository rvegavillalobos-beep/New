import numpy as np
import streamlit as st

from qc import analysis, charts, metrics, ui

c = ui.ctx()
a = c.a
s = c.settings

st.title("Overview")
ui.scope_line(c)
ui.require_rows(a.first)

first = a.first
counts = metrics.status_counts(first)
fpy = a.fpy
gap = fpy - s.fpy_target
first_sq = a.squareness[a.squareness["RunNum"] == 1] if not a.squareness.empty else a.squareness
deformed_pct = (first_sq["SquarenessStatus"] == "DEFORMED").mean() * 100 if len(first_sq) else np.nan

# ---- KPI tiles ------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("First-pass yield", ui.fmt_pct(fpy), f"{gap:+.1f} pp vs {s.fpy_target:.0f}% target",
          delta_color="normal", border=True,
          help="Share of modules whose FIRST measurement had every corner within tolerance.")
k2.metric("Modules measured", f"{len(first):,}",
          f"{(a.runs['RunNum'] > 1).sum():,} re-measurements", delta_color="off", border=True,
          help="Unique modules (first run). Re-measurements are later runs of the same modules.")
nok_label = f"{counts['FAIL']:,}"
if not s.exclude_incomplete and counts["INCOMPLETE"]:
    nok_label += f" + {counts['INCOMPLETE']} incomplete"
k3.metric("NOK at first measurement", nok_label,
          f"{counts['FAIL'] / len(first) * 100:.1f}% of modules", delta_color="off", border=True)
k4.metric("Deformed (first run)", ui.fmt_pct(deformed_pct),
          f"diagonal delta > {s.diag_tol:g} mm", delta_color="off", border=True,
          help="Complete first runs whose diagonal delta exceeds the squareness tolerance.")

# ---- Findings -----------------------------------------------------------------
with st.container(border=True):
    st.markdown("##### What the data says")
    items = analysis.findings(a)
    if items:
        ui.findings_box(items)
    else:
        st.caption("Not enough data for findings.")

# ---- Trend + split by type -----------------------------------------------------
left, right = st.columns([2, 1], gap="large")
with left:
    head, opts = st.columns([4, 1], vertical_alignment="bottom")
    head.markdown("##### Weekly first-pass yield")
    with opts.popover("Options", icon=":material/tune:", width="stretch"):
        st.slider("Moving average (weeks)", 2, 10, step=1, key=ui.init("p_ma", 3))
    st.plotly_chart(charts.weekly_fpy(a.weekly, s.fpy_target, c.ma_window, not s.exclude_incomplete),
                    width="stretch", key="ov_weekly")
    if a.weekly["LowSample"].any():
        st.caption("Hollow points: weeks with fewer than 5 modules (FPY swings are mostly noise there).")

with right:
    st.markdown("##### First run by battery type")
    st.plotly_chart(charts.status_by_type(first), width="stretch", key="ov_bytype")
    by_type = (first.groupby("BatteryType")["Status"]
               .agg(Modules="size", FPY=lambda x: (x == "PASS").mean() * 100)
               .reset_index().rename(columns={"BatteryType": "Type"}))
    st.dataframe(by_type, hide_index=True, width="stretch",
                 column_config={"FPY": st.column_config.ProgressColumn(
                     "FPY", format="%.1f%%", min_value=0, max_value=100)})

# ---- Why modules fail ---------------------------------------------------------
left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("##### Why modules fail at first measurement")
    st.plotly_chart(charts.failure_pareto(a.pareto, counts["FAIL"]), width="stretch", key="ov_pareto")
    st.caption("Corner and axis out of tolerance, as a share of NOK modules. "
               "See **Corners** for whether it is an offset or scatter.")
with right:
    st.markdown("##### Week by week")
    wk = a.weekly.copy()
    cols = ["CalendarWeek", "Total", "Passed", "Failed"] + (["Incomplete"] if not s.exclude_incomplete else []) + ["PassRate"]
    wk = wk[cols].iloc[::-1]
    st.dataframe(
        wk, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(wk)),
        column_config={
            "CalendarWeek": "Week", "Total": "Modules",
            "PassRate": st.column_config.ProgressColumn("FPY", format="%.0f%%", min_value=0, max_value=100),
        },
    )
