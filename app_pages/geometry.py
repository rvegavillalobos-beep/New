import pandas as pd
import streamlit as st

from qc import charts, ui
from qc.constants import CORNERS

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

row2 = st.columns([3, 1.4], gap="medium", vertical_alignment="bottom")
options = ["(none)"] + runs["_mod_key"].iloc[::-1].tolist()
focus = st.session_state.get("focus")
if focus not in options:
    focus = None
with row2[0]:
    sel = st.selectbox("Focus on a module", options, index=options.index(focus) if focus else 0,
                       help="Type part of a Part ID to search. Selecting a row in Squareness or "
                            "Compensation tables also sets the focus.")
focus = None if sel == "(none)" else sel
st.session_state["focus"] = focus
with row2[1]:
    show_opts = ["Fail", "Pass", "Incomplete"]
    ui.init("p_geo_show", show_opts)
    shown = st.pills("Show", show_opts, selection_mode="multi", key="p_geo_show") or []
status_map = {"Fail": "FAIL", "Pass": "PASS", "Incomplete": "INCOMPLETE"}
keep = [status_map[x] for x in shown]

window = runs.iloc[rng[0] - 1: rng[1]]
window = window[window["Status"].isin(keep)]
plot_df = window
if focus and focus not in window["_mod_key"].values:
    plot_df = pd.concat([window, runs[runs["_mod_key"] == focus]])

left, right = st.columns([3.2, 1], gap="large")
with left:
    st.plotly_chart(charts.geometry_plot(plot_df, c.types, limit, exag, focus), width="stretch", key="geo_plot")
    st.caption(f"Deviations and tolerance boxes magnified ×{exag}; Y axis reversed to match the station view. "
               "**Legend**: click an entry to hide it, double-click to show only that one, "
               "double-click again to bring all back.")
with right:
    if focus:
        row = runs[runs["_mod_key"] == focus].iloc[0]
        st.markdown(f"**{row['PartID']}**")
        st.caption(f"Run {row['RunNum']} · {row['Date']:%Y-%m-%d %H:%M} · {row['BatteryType']} · {row['CalendarWeek']}")
        st.badge(row["Status"].title(), color={"PASS": "green", "FAIL": "red"}.get(row["Status"], "gray"))
        dev = pd.DataFrame({"Corner": CORNERS,
                            "X (mm)": [row[f"{k}_X"] for k in CORNERS],
                            "Y (mm)": [row[f"{k}_Y"] for k in CORNERS]})
        st.dataframe(
            dev.style.map(lambda v: "background-color:#d03b3b26;color:#a32222;font-weight:600"
                          if pd.notna(v) and abs(v) > limit else "", subset=["X (mm)", "Y (mm)"])
               .format("{:+.2f}", subset=["X (mm)", "Y (mm)"], na_rep="missing"),
            hide_index=True, width="stretch")
        sq = a.squareness[a.squareness["_mod_key"] == focus]
        if not sq.empty:
            r = sq.iloc[0]
            geo = pd.DataFrame({
                "Measure": ["Diagonal Δ", "Width Δ (front − rear)", "Length Δ (left − right)",
                            "FL angle Δ", "Shape uses"],
                "Value": [f"{r['DeltaDiag']:.2f} mm", f"{r['WidthDelta']:+.2f} mm", f"{r['LengthDelta']:+.2f} mm",
                          f"{r['AngleDevFL']:+.2f}°", f"{r['ShapeMax']:.2f} of ±{limit:g} mm"],
            })
            st.dataframe(geo, hide_index=True, width="stretch")
            if r["SquarenessStatus"] == "DEFORMED":
                st.markdown(f":red[**Deformed**] · {r['RootCause'].capitalize()}")
            else:
                st.markdown(f":green[**Square**] (diagonal Δ ≤ {c.settings.diag_tol:g} mm)")
            if r["Status"] == "FAIL":
                st.caption("Would pass if it were perfectly square (same placement)."
                           if r["PassIfSquare"] else "Fails on placement alone: it would fail even if perfectly square.")
        else:
            st.caption("Not all four corners were measured: no geometry metrics.")
        others = a.runs[a.runs["PartID"] == row["PartID"]].sort_values("RunNum")
        if len(others) > 1:
            st.caption("All runs: " + " → ".join(
                f"R{int(rn)} {stt.title()}" for rn, stt in zip(others["RunNum"], others["Status"])))
    else:
        st.info("Pick a module above to see its deviations and geometry.", icon=":material/ads_click:")
    pass_n = (window["Status"] == "PASS").sum()
    st.caption(f"In view: {len(window)} modules, {pass_n} pass, {len(window) - pass_n} not pass.")
