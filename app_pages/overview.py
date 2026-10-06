import streamlit as st

from qc import analysis, charts, metrics, ui

c = ui.ctx()
a = c.a
s = c.settings


def _toggle_insights():
    st.session_state["p_insights"] = not st.session_state.get("p_insights", False)


head, wand = st.columns([12, 1], vertical_alignment="center")
head.title("Overview")
ui.init("p_insights", False)
wand.button("", icon=":material/auto_fix_high:", type="tertiary", key="ov_wand",
            on_click=_toggle_insights, help="What the data says")
ui.scope_line(c)
ui.require_rows(a.first)

first = a.first
counts = metrics.status_counts(first)
fpy = a.fpy
gap = fpy - s.fpy_target

# ---- KPI tiles ------------------------------------------------------------
k1, k2, k3 = st.columns(3)
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

# ---- Findings (on demand) -----------------------------------------------------
if st.session_state["p_insights"]:
    with st.container(border=True):
        st.markdown("##### What the data says")
        items = analysis.findings(a)
        if items:
            ui.findings_box(items)
        else:
            st.caption("Not enough data for findings.")

# ---- Weekly FPY next to the battery-type split ------------------------------------
head, opts = st.columns([6, 1], vertical_alignment="bottom")
head.markdown("##### Weekly first-pass yield")
with opts.popover("Options", icon=":material/tune:", width="stretch"):
    st.slider("Moving average (weeks)", 2, 10, step=1, key="p_ma")

left, right = st.columns([2, 1], gap="large", vertical_alignment="center")
with left:
    st.plotly_chart(charts.weekly_fpy(a.weekly, s.fpy_target, c.ma_window, not s.exclude_incomplete),
                    width="stretch", key="ov_weekly")
with right:
    st.markdown("**First run by battery type**")
    st.plotly_chart(charts.status_by_type(first), width="stretch", key="ov_bytype")
if a.weekly["LowSample"].any():
    st.caption("\\* and hollow points: weeks with fewer than 5 modules, where FPY swings are mostly noise.")

# ---- Week by week -------------------------------------------------------------------
wk = a.weekly.copy()
cols = (["CalendarWeek", "Total", "Passed", "Failed"]
        + (["Incomplete"] if not s.exclude_incomplete else []) + ["PassRate", "MA_FPY"])
wk = wk[cols].iloc[::-1]
st.dataframe(
    wk, hide_index=True, width="stretch", height=min(400, 38 + 35 * len(wk)),
    column_config={
        "CalendarWeek": "Week", "Total": "Modules", "Passed": "OK", "Failed": "NOK",
        "PassRate": st.column_config.ProgressColumn("FPY", format="%.1f%%", min_value=0, max_value=100),
        "MA_FPY": st.column_config.NumberColumn(f"MA{c.ma_window}", format="%.1f%%"),
    },
)

# ---- Why modules fail (end of page) -----------------------------------------------
st.markdown("##### Why modules fail")
_, mid, _ = st.columns([1, 3, 1])
with mid:
    st.plotly_chart(charts.failure_pareto(a.pareto, counts["FAIL"]), width="stretch", key="ov_pareto")
    st.caption("Corner and axis out of tolerance, as a share of NOK modules (a module can count in "
               "several bars). **Corners** shows whether each is an offset or scatter.")
