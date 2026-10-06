import numpy as np
import pandas as pd
import streamlit as st

from qc import charts, compensation as comp, geometry, ui
from qc.constants import TYPE_COLORS

c = ui.ctx()
a = c.a
spec = c.settings.spec_limit

st.title("Compensation")
ui.scope_line(c)
ui.require_rows(a.vec)

st.caption(
    "Simulates a rigid X / Y / yaw correction of the module placement (rotation about the "
    "module's nominal center) and re-checks every first measurement against the tolerance. "
    "Nothing here changes the measured data."
)

# ---- Controls -------------------------------------------------------------------
types = c.types
ctl = st.columns([1.3, 1.1, 1.6, 0.9], gap="medium", vertical_alignment="bottom")
with ctl[0]:
    if len(types) > 1:
        ui.ensure_option("p_comp_type", types, types[0])
        t = st.segmented_control("Battery type", types, key="p_comp_type") or types[0]
    else:
        t = types[0]
        st.markdown(f"**{t}**")
vec_t = a.vec[a.vec["BatteryType"] == t]
weeks = vec_t[["WeekKey"]].drop_duplicates().sort_values("WeekKey")["WeekKey"].tolist()
with ctl[1]:
    ui.init("p_comp_n", 6)
    if st.session_state["p_comp_n"] > max(1, len(weeks)):
        st.session_state["p_comp_n"] = max(1, len(weeks))
    n_weeks = st.number_input("Last N weeks", 1, max(1, len(weeks)), step=1, key="p_comp_n",
                              help="Analysis window, counted back from the last week in the sidebar range.")
with ctl[2]:
    margin = st.slider("Safety margin (mm)", 0.0, 1.0, step=0.05, key=ui.init("p_margin", 0.2),
                       help="The optimizer prefers settings where modules pass with at least this much "
                            "room on every corner, so the result doesn't rely on modules that barely pass. "
                            "0 = maximize FPY at the limit only.")
with ctl[3]:
    with st.popover("More", icon=":material/settings:", width="stretch"):
        method = st.radio("Median-based uses", ["Median", "Mean"], key=ui.init("p_method", "Median"),
                          horizontal=True)
        min_n = st.number_input("Minimum modules to trust a result", 5, 500, step=5, key=ui.init("p_min_n", 30))

df = vec_t[vec_t["WeekKey"].isin(weeks[-int(n_weeks):])]
ui.require_rows(df)
method_key = "mean" if method == "Mean" else "median"
win_label = df["CalendarWeek"].iloc[0] if df["WeekKey"].nunique() == 1 else \
    f"{df.sort_values('WeekKey')['CalendarWeek'].iloc[0]} – {df.sort_values('WeekKey')['CalendarWeek'].iloc[-1]}"

# ---- Suggestions -----------------------------------------------------------------
mx, my, myaw = comp.median_suggestion(df, method_key)
med_ok = bool(np.isfinite([mx, my, myaw]).all())
res = ui.run_optimize(df, spec, float(margin), float(myaw) if med_ok else 0.0)
bt = ui.run_backtest(vec_t, spec, float(margin), int(n_weeks), method_key)

options = {"No compensation": (0.0, 0.0, 0.0)}
if med_ok:
    options["Median-based"] = (float(mx), float(my), float(myaw))
if res is not None:
    options["Optimized"] = (res.dx, res.dy, res.yaw)

manual_on = st.toggle("Try my own values", key=ui.init("p_manual_on", False),
                      help="Enter X / Y / yaw yourself; every result below updates.")
if manual_on:
    base = options.get("Optimized", options.get("Median-based", (0.0, 0.0, 0.0)))
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1.2], vertical_alignment="bottom")
    kx, ky, kyaw = f"p_man_x_{t}", f"p_man_y_{t}", f"p_man_yaw_{t}"
    ui.init(kx, round(base[0], 2))
    ui.init(ky, round(base[1], 2))
    ui.init(kyaw, round(base[2], 3))
    m1.number_input("X (mm)", step=0.05, format="%.2f", key=kx)
    m2.number_input("Y (mm)", step=0.05, format="%.2f", key=ky)
    m3.number_input("Yaw (°)", step=0.005, format="%.3f", key=kyaw)

    def _copy(src):
        st.session_state[kx], st.session_state[ky], st.session_state[kyaw] = (
            round(src[0], 2), round(src[1], 2), round(src[2], 3))

    with m4:
        b1, b2 = st.columns(2)
        if "Optimized" in options:
            b1.button("= Optimized", on_click=_copy, args=(options["Optimized"],), width="stretch")
        if "Median-based" in options:
            b2.button("= Median", on_click=_copy, args=(options["Median-based"],), width="stretch")
    options["Manual"] = (float(st.session_state[kx]), float(st.session_state[ky]), float(st.session_state[kyaw]))

rows = []
for name, (dx, dy, yw) in options.items():
    ev = comp.evaluate(df, dx, dy, yw, spec, margin)
    backtested = np.nan
    if bt is not None:
        backtested = {"No compensation": bt["none"], "Median-based": bt["median"],
                      "Optimized": bt["optimized"]}.get(name, np.nan)
    rows.append({"Option": name, "X (mm)": dx, "Y (mm)": dy, "Yaw (°)": yw,
                 "FPY": ev["fpy"], "FPY with margin": ev["fpy_margin"], "Backtest": backtested})
table = pd.DataFrame(rows)
base_fpy = table.loc[table["Option"] == "No compensation", "FPY"].iloc[0]

# ---- Headline ----------------------------------------------------------------------
st.markdown(f"##### {t} · {win_label} · {len(df)} modules")
k = st.columns(len(table))
for col, (_, r) in zip(k, table.iterrows()):
    delta = None if r["Option"] == "No compensation" else f"{r['FPY'] - base_fpy:+.1f} pp"
    col.metric(r["Option"], ui.fmt_pct(r["FPY"]), delta, border=True,
               help=f"X {r['X (mm)']:+.2f} mm · Y {r['Y (mm)']:+.2f} mm · yaw {r['Yaw (°)']:+.3f}°")

# Verdict from the backtest (one module = smallest meaningful difference).
if bt is not None:
    step = 100 / bt["n"]
    b_none, b_med, b_opt = bt["none"], bt["median"], bt["optimized"]
    if max(b_med, b_opt) - b_none < step - 1e-9:
        st.warning(f"**Backtest:** on {bt['label']} neither method would have raised FPY "
                   f"({b_none:.1f}% without · median {b_med:.1f}% · optimized {b_opt:.1f}%). "
                   "The in-window gains above come from fitting these same modules.",
                   icon=":material/balance:")
    elif b_opt >= b_med + step - 1e-9:
        st.success(f"**Backtest:** the optimized setting held up best on {bt['label']}: "
                   f"{b_opt:.1f}% vs median {b_med:.1f}% and {b_none:.1f}% without compensation.",
                   icon=":material/verified:")
    elif b_med >= b_opt + step - 1e-9:
        st.info(f"**Backtest:** median-based held up best on {bt['label']}: {b_med:.1f}% vs optimized "
                f"{b_opt:.1f}% and {b_none:.1f}% without. The optimizer's extra in-window gain did not "
                "repeat on new modules: it fitted the sample.", icon=":material/balance:")
    else:
        st.info(f"**Backtest:** median-based and optimized did the same on {bt['label']} "
                f"({b_med:.1f}% vs {b_opt:.1f}%; {b_none:.1f}% without). The optimizer's extra in-window "
                "gain did not carry over yet; with more modules per week it may.", icon=":material/balance:")
else:
    st.caption("No backtest yet: it needs earlier weeks with at least 10 modules. The in-window FPY "
               "above is optimistic, especially for the optimized setting.")

left, right = st.columns([3, 2], gap="large")
with left:
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "X (mm)": st.column_config.NumberColumn(format="%+.2f"),
            "Y (mm)": st.column_config.NumberColumn(format="%+.2f"),
            "Yaw (°)": st.column_config.NumberColumn(format="%+.3f"),
            "FPY": st.column_config.ProgressColumn("FPY", format="%.1f%%", min_value=0, max_value=100),
            "FPY with margin": st.column_config.ProgressColumn(
                f"Pass with {margin:.2f} mm room", format="%.1f%%", min_value=0, max_value=100,
                help="Modules that would pass even with every corner limit tightened by the safety margin."),
            "Backtest": st.column_config.NumberColumn(
                "Backtest", format="%.1f%%",
                help="Replays recent weeks as if the method had been in use: each week gets the setting "
                     "computed from the weeks before it, never from itself. The honest estimate of "
                     "what to expect on the line."),
        },
    )
    if bt is not None:
        st.caption(f"**Backtest**: each of {bt['n_weeks']} weeks ({bt['label']}, {bt['n']} modules) got the "
                   f"setting computed from the {int(n_weeks)} weeks before it. *FPY* is measured on the same "
                   "modules the setting was computed from, so it is optimistic.")
    else:
        st.caption("**Backtest** needs earlier weeks with at least 10 modules before the weeks it tests.")

with right:
    items = comp.readiness(df, min_n=int(min_n), bt=bt)
    ready = all(i["ok"] for i in items)
    with st.container(border=True):
        h1, h2 = st.columns([3, 1.3], vertical_alignment="center")
        h1.markdown("**Ready to apply on the line?**")
        h2.badge("Ready" if ready else "Not yet", color="green" if ready else "orange",
                 icon=":material/check:" if ready else ":material/hourglass_top:")
        for i in items:
            icon = ":green[:material/check_circle:]" if i["ok"] else ":orange[:material/radio_button_unchecked:]"
            st.markdown(f"{icon}&nbsp; **{i['check']}**: {i['detail']}")

# ---- Why the optimized setting differs -----------------------------------------------
pick_opts = [o for o in ("Optimized", "Median-based", "Manual") if o in options]
if not pick_opts:
    st.info("No module in this window has all four corners measured, so no compensation can be "
            "computed. Widen the window or the calendar weeks.")
    st.stop()
ui.ensure_option("p_comp_view", pick_opts, "Manual" if manual_on else pick_opts[0])
st.markdown("##### Inspect a setting")
view = st.segmented_control("Show details for", pick_opts, key="p_comp_view",
                            label_visibility="collapsed") or pick_opts[0]
dx, dy, yw = options[view]

if res is not None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**FPY as yaw changes**")
        fixed = comp.fpy_vs_yaw(df, spec, dx, dy, res.yaw_grid)
        pts = {name: (o[2], table.loc[table["Option"] == name, "FPY"].iloc[0]) for name, o in options.items()}
        st.plotly_chart(charts.yaw_profile(res.yaw_grid, res.best_fpy_by_yaw, fixed, pts, TYPE_COLORS[t]),
                        width="stretch", key="cp_yaw")
        st.caption(f"Solid: best FPY reachable at each yaw when X / Y are re-optimized. Dotted: only yaw "
                   f"changes, X / Y kept at the {view.lower()} values. The peak is often not at the "
                   f"median yaw, because what matters is each module's *worst* corner, not the average.")
    with right:
        st.markdown(f"**FPY over X / Y at yaw {yw:+.3f}°**")
        land = comp.fpy_landscape(df, yw, spec, center=(dx, dy))
        pts_xy = {name: (o[0], o[1]) for name, o in options.items() if abs(o[2] - yw) < 1e-9}
        st.plotly_chart(charts.fpy_landscape(land, pts_xy), width="stretch", key="cp_land")
        st.caption("Darker = higher FPY. A setting in the middle of a dark area is robust; "
                   "one on the edge of a cliff is not. Only settings with this same yaw are marked.")

# ---- Details --------------------------------------------------------------------------
sim = geometry.simulate_compensation(df, dx, dy, yw, spec, incomplete_as_fail=c.settings.exclude_incomplete)
sim = pd.concat([sim, geometry.sim_geometry(sim)], axis=1)
st.session_state.setdefault("_comp_export", {})[t] = {
    "window": win_label, "modules": len(df), "margin_mm": margin, "table": table,
    "offsets": geometry.corner_offsets(t, dx, dy, yw), "inspected": view, "sim": sim,
}

st.markdown(f"##### Details · {view} (X {dx:+.2f} mm · Y {dy:+.2f} mm · yaw {yw:+.3f}°)")
tab1, tab2, tab3, tab5, tab4 = st.tabs(["Per-corner offsets", "Before vs after", "By week", "Backtest", "Modules"])
with tab1:
    off = geometry.corner_offsets(t, dx, dy, yw)
    l, r = st.columns([1, 2], gap="large")
    l.dataframe(off.drop(columns="BatteryType"), hide_index=True, width="stretch",
                column_config={"Offset_X_mm": st.column_config.NumberColumn("X shift (mm)", format="%+.3f"),
                               "Offset_Y_mm": st.column_config.NumberColumn("Y shift (mm)", format="%+.3f")})
    r.caption("How far each nominal corner moves under this rigid correction. They are not "
              "independent: they follow from one X / Y shift plus one rotation about the module center.")
with tab2:
    l, r = st.columns(2, gap="large")
    with l:
        st.markdown("**Module center, measured vs compensated**")
        st.plotly_chart(charts.centroid_before_after(sim), width="stretch", key="cp_ba")
    with r:
        st.markdown("**Average outline**")
        st.plotly_chart(charts.overlay(t, sim), width="stretch", key="cp_overlay")
    trans = sim.groupby(["Status", "Status_Sim"]).size().unstack(fill_value=0)
    trans.index.name, trans.columns.name = "Measured", "With compensation"
    st.markdown("**Status change** (rows: measured, columns: with compensation)")
    st.dataframe(trans, width="content")
with tab3:
    wk = (sim.groupby(["WeekKey", "CalendarWeek"])
          .apply(lambda g: pd.Series({"Real": (g["Status"] == "PASS").mean() * 100,
                                      "Sim": (g["Status_Sim"] == "PASS").mean() * 100, "N": len(g)}),
                 include_groups=False)
          .reset_index().sort_values("WeekKey"))
    st.plotly_chart(charts.weekly_fpy_compare(wk), width="stretch", key="cp_week")
with tab5:
    if bt is None:
        st.caption("Not enough history for a backtest.")
    else:
        st.plotly_chart(charts.backtest_chart(bt["weeks"]), width="stretch", key="cp_bt")
        st.caption(f"Each week uses the setting computed from the {int(n_weeks)} weeks before it "
                   "(same safety margin and method as above).")
        bw = bt["weeks"][["Week", "Modules", "Trained on", "none_fpy", "median_fpy", "optimized_fpy"]]
        st.dataframe(bw, hide_index=True, width="stretch", column_config={
            "Trained on": st.column_config.NumberColumn("Fitted on (modules)"),
            "none_fpy": st.column_config.NumberColumn("No compensation", format="%.0f%%"),
            "median_fpy": st.column_config.NumberColumn("Median-based", format="%.0f%%"),
            "optimized_fpy": st.column_config.NumberColumn("Optimized", format="%.0f%%"),
        })
with tab4:
    cols = ["_mod_key", "CalendarWeek", "PartID", "FL_X", "FL_Y", "FR_X", "FR_Y", "RL_X", "RL_Y", "RR_X", "RR_Y",
            "Status", "FL_X_Sim", "FL_Y_Sim", "FR_X_Sim", "FR_Y_Sim", "RL_X_Sim", "RL_Y_Sim", "RR_X_Sim",
            "RR_Y_Sim", "Status_Sim"]
    piece = sim[cols].reset_index(drop=True)
    st.caption("Select a row to focus that module in Geometry.")
    ev = st.dataframe(ui.status_styler(piece.drop(columns="_mod_key"), spec), hide_index=True, width="stretch",
                      on_select="rerun", selection_mode="single-row", key="cp_table",
                      column_config={"CalendarWeek": "Week"})
    sel = ev.selection.rows if ev is not None else []
    if sel:
        ui.focus_module(piece.iloc[sel[0]]["_mod_key"])
        st.page_link("app_pages/geometry.py", label=f"Open {piece.iloc[sel[0]]['PartID']} in Geometry",
                     icon=":material/open_in_new:")

with st.expander("How the optimized setting is found", icon=":material/help:"):
    st.markdown(
        "- A module passes only if **all four corners** are within tolerance in X and Y. For a given "
        "yaw, the X / Y corrections that make one module pass form a rectangle; the best correction is "
        "the point covered by the most rectangles. Yaw is scanned finely around zero and the median "
        "yaw, and the result is re-checked exactly.\n"
        f"- Priority: (1) modules passing with **{margin:.2f} mm** of room on every corner, (2) modules "
        "passing at the limit, (3) the middle of a good region rather than its edge, (4) the smallest move.\n"
        "- **Median-based** centers the *average* module. When corners are biased differently (one corner "
        "sits further out than the others), that is not the same as passing the most modules, which is "
        "why small tweaks around it, especially in yaw, can do better.\n"
        "- With few modules, any optimizer fits noise: it finds the setting that best suits *these* "
        "modules, and part of that gain does not repeat on the next ones. The **Backtest** column shows "
        "how much survives; trust it more than the in-window FPY. As data accumulates, the two converge."
    )
