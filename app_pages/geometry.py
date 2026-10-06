import pandas as pd
import streamlit as st

from qc import charts, geometry, ui

c = ui.ctx()
a = c.a
limit = c.settings.spec_limit

st.title("Geometry")
ui.scope_line(c)
ui.require_rows(a.runs, "measurements")

ctrl = st.columns([1.2, 2, 1.4], gap="medium", vertical_alignment="bottom")
with ctrl[0]:
    ui.ensure_option("p_geo_scope", ["First run", "All runs"], "First run")
    scope = st.segmented_control("Measurements", ["First run", "All runs"], key="p_geo_scope") or "First run"
runs = a.runs if scope == "All runs" else a.runs[a.runs["RunNum"] == 1]
runs = runs.sort_values("Date").reset_index(drop=True)
ui.require_rows(runs, "measurements")

n = len(runs)
with ctrl[1]:
    if n > 1:
        rng = st.slider("Modules shown (time order)", 1, n, (max(1, n - 9), n), key=f"geo_range_{scope}_{n}",
                        help="Pick the window of modules to draw; 1 = oldest in the filters.")
    else:
        rng = (1, 1)
with ctrl[2]:
    st.slider("Deviation magnification", 1, 60, step=1, key=ui.init("p_exag", 25),
              help="Deviations are a few mm on a ~1.7 m module; magnify them to see the shape.")
exag = st.session_state["p_exag"]

window = runs.iloc[rng[0] - 1: rng[1]]

# ---- Focus module ---------------------------------------------------------------
options = ["(none)"] + runs["_mod_key"].iloc[::-1].tolist()
focus = st.session_state.get("focus")
if focus not in options:
    focus = None
sel = st.selectbox("Focus on a module", options, index=options.index(focus) if focus else 0,
                   help="Type part of a Part ID to search. Selecting a row in Squareness or "
                        "Compensation tables also sets the focus.")
focus = None if sel == "(none)" else sel
st.session_state["focus"] = focus

plot_df = window
if focus and focus not in window["_mod_key"].values:
    plot_df = pd.concat([window, runs[runs["_mod_key"] == focus]])

left, right = st.columns([3, 1.15], gap="large")
with left:
    st.plotly_chart(charts.geometry_plot(plot_df, c.types, limit, exag, focus), width="stretch", key="geo_plot")
    st.caption(f"Deviations and tolerance boxes magnified ×{exag}. Grey: pass · red: fail · "
               "Y axis reversed to match the station view.")
with right:
    if focus:
        row = runs[runs["_mod_key"] == focus].iloc[0]
        st.markdown(f"**{row['PartID']}**")
        st.caption(f"Run {row['RunNum']} · {row['Date']:%Y-%m-%d %H:%M} · {row['BatteryType']} · {row['CalendarWeek']}")
        st.badge(row["Status"], color={"PASS": "green", "FAIL": "red"}.get(row["Status"], "gray"))
        st.plotly_chart(charts.module_corners(row, limit), width="stretch", key="geo_module")
        sq = geometry.squareness(runs[runs["_mod_key"] == focus], c.settings.diag_tol)
        if not sq.empty:
            r = sq.iloc[0]
            st.markdown(f"Diagonal delta **{r['DeltaDiag']:.2f} mm** · {r['SquarenessStatus'].title()}")
            if r["SquarenessStatus"] == "DEFORMED":
                st.caption(r["RootCause"].capitalize())
        others = a.runs[a.runs["PartID"] == row["PartID"]].sort_values("RunNum")
        if len(others) > 1:
            st.caption("All runs of this module: " + " → ".join(
                f"R{int(r)} {s}" for r, s in zip(others["RunNum"], others["Status"])))
    else:
        st.info("Pick a module above to see its corners, squareness and run history.", icon=":material/ads_click:")
    pass_n = (window["Status"] == "PASS").sum()
    st.caption(f"In view: {len(window)} modules, {pass_n} pass, {len(window) - pass_n} not pass.")
