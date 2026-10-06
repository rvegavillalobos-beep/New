import numpy as np
import pandas as pd
import streamlit as st

from qc import charts, metrics, ui
from qc.constants import CAUSE_COLORS, CAUSE_ORDER, NOK_CAUSE_COLORS, NOK_CAUSE_ORDER, STATUS_COLORS

c = ui.ctx()
a = c.a
tol = c.settings.diag_tol
limit = c.settings.spec_limit

st.title("Squareness")
ui.scope_line(c)

sq = a.squareness
ui.require_rows(sq, "complete measurements (all four corners)")

ui.ensure_option("p_sq_scope", ["First run", "All runs"], "First run")
scope = st.segmented_control("Measurements", ["First run", "All runs"], key="p_sq_scope") or "First run"
if scope == "First run":
    sq = sq[sq["RunNum"] == 1]
ui.require_rows(sq, "complete first runs")

lk = metrics.shape_link(sq, tol)
deformed = sq[sq["SquarenessStatus"] == "DEFORMED"]
n_fail = lk.get("n_fail", 0)

# ---- KPIs ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Deformed", f"{len(deformed) / len(sq) * 100:.1f}%", f"{len(deformed)} of {len(sq)} modules",
          delta_color="off", border=True, help=f"Diagonal delta above {tol:g} mm.")
if n_fail:
    k2.metric("NOK caused by shape", f"{lk['nok_if_square']} of {n_fail}",
              f"{lk['nok_if_square'] / n_fail * 100:.0f}% of NOK", delta_color="off", border=True,
              help="NOK modules that would have passed with the same placement if they were perfectly square.")
    k3.metric("FPY if perfectly square", f"{lk['fpy_if_square']:.1f}%",
              f"{lk['fpy_if_square'] - lk['fpy']:+.1f} pp vs {lk['fpy']:.1f}% today", border=True,
              help="Complete modules only (all four corners measured).")
else:
    k2.metric("NOK caused by shape", "–", border=True)
    k3.metric("FPY if perfectly square", "–", border=True)
k4.metric("Tolerance used by shape", f"{lk.get('shape_median', np.nan):.2f} mm",
          f"median, of ±{limit:g} mm", delta_color="off", border=True,
          help="Largest corner deviation left once the module is placed as well as possible: "
               "the part of the tolerance its shape uses up no matter how it is placed.")

# ---- Link to NOK -------------------------------------------------------------------
st.markdown("##### Is deformation behind the NOK?")
if n_fail == 0:
    st.success("No NOK modules among the complete measurements in this selection.")
else:
    k = lk["nok_if_square"]
    if k == 0:
        line1 = ("**Shape did not cost any NOK here.** Every NOK module would still fail if it were "
                 "perfectly square: they fail on placement (shift and rotation).")
    else:
        who = (f"**All {n_fail} NOK modules**" if k == n_fail
               else f"**Shape cost {k} of {n_fail} NOK modules** ({k / n_fail * 100:.0f}%)")
        rest = ("" if k == n_fail else f" The other {n_fail - k} fail on placement alone.")
        alone = (" No module fails from its shape alone." if lk["shape_alone"] == 0 else
                 f" {lk['shape_alone']} fail from their shape alone, whatever the placement.")
        line1 = (f"{who}{' would have passed' if k == n_fail else ''}"
                 f"{':' if k != n_fail else ''} with the same placement but a perfectly square shape"
                 f"{' they would have passed' if k != n_fail else ''}, lifting FPY of complete modules from "
                 f"{lk['fpy']:.1f}% to {lk['fpy_if_square']:.1f}%.{rest}{alone}")
    p = lk["p_value"]
    if np.isfinite(lk["fail_rate_deformed"]) and np.isfinite(lk["fail_rate_square"]):
        line2 = (f"Modules flagged *deformed* by the diagonal rule were NOK {lk['fail_rate_deformed']:.0f}% of the "
                 f"time vs {lk['fail_rate_square']:.0f}% for square ones"
                 + (f": a real difference (p = {p:.3f})." if p < 0.05 else
                    f": no clear difference (p = {p:.2f}), so the diagonal flag on its own does not predict NOK."))
    else:
        line2 = ""
    line3 = (f"Shape uses a median {lk['shape_median_fail']:.2f} mm of the ±{limit:g} mm tolerance in NOK modules "
             f"and {lk['shape_median_pass']:.2f} mm in OK ones." if np.isfinite(lk["shape_median_pass"]) else "")
    box = st.warning if k / n_fail >= 0.25 else st.info
    box("\n\n".join(x for x in (line1, line2, line3) if x), icon=":material/troubleshoot:")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Why each NOK module failed**")
    causes = lk.get("causes", pd.DataFrame())
    if causes is not None and not causes.empty:
        st.plotly_chart(charts.stacked_share(causes, NOK_CAUSE_ORDER, NOK_CAUSE_COLORS), width="stretch",
                        key="sq_causes")
    st.caption("**Placement**: fails even if perfectly square · **Shape + placement**: would pass if square "
               "(the shape ate the margin) · **Shape**: fails from its shape even if perfectly placed.")
with right:
    st.markdown("**NOK rate by amount of deformation**")
    if "by_band" in lk:
        st.plotly_chart(charts.fail_rate_by_band(lk["by_band"], 100 - lk["fpy"], tol), width="stretch",
                        key="sq_band")
    st.caption("If deformation drove NOK, bars would rise to the right. Bands with few modules swing a lot.")

with st.expander("How this is worked out", icon=":material/help:"):
    st.markdown(
        "- Each module's four corner deviations are split in two: **placement**, the X/Y shift and rotation "
        "that best explains the four corners together (least squares), and **shape**, what is left over.\n"
        "- *Would pass if square*: keep the module's placement, drop its shape, check the tolerance again.\n"
        "- *Shape uses*: the largest corner deviation left by the shape alone. A module whose shape uses "
        "1 mm has only the remaining ±(tolerance − 1) mm for placement.\n"
        "- The deformed/square comparison uses Fisher's exact test; p < 0.05 means the difference is "
        "unlikely to be chance.\n"
        "- Only modules with all four corners measured are included."
    )

# ---- Over time -----------------------------------------------------------------------
st.markdown("##### Deformation over time")
st.plotly_chart(charts.deformation_trend(metrics.weekly_deformation(sq)), width="stretch", key="sq_trend")

# ---- Pattern -------------------------------------------------------------------------
st.markdown("##### Deformation pattern")
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Square vs deformed**")
    counts = sq.groupby(["BatteryType", "SquarenessStatus"]).size().unstack(fill_value=0)
    st.plotly_chart(charts.stacked_share(counts, ["SQUARE OK", "DEFORMED"], STATUS_COLORS),
                    width="stretch", key="sq_status")
with right:
    st.markdown("**Pattern of the deformed ones**")
    if deformed.empty:
        st.success("No deformed modules in this selection.")
    else:
        cc = deformed.groupby(["BatteryType", "CauseCategory"]).size().unstack(fill_value=0)
        st.plotly_chart(charts.stacked_share(cc, CAUSE_ORDER, CAUSE_COLORS), width="stretch", key="sq_cause")
st.caption(f"Patterns, checked in this order: parallelogram tilt (FL angle off by > 0.15° with little width "
           f"change), trapezoidal width (front and rear widths differ ≥ 0.8 mm), trapezoidal length (left and "
           f"right lengths differ ≥ 0.8 mm), else combined asymmetry. Deformed = diagonal delta above {tol:g} mm.")

# ---- Table ---------------------------------------------------------------------------
st.markdown("##### Modules")
st.caption("Select a row to focus that module, then open it in Geometry.")
view = sq[["Date", "CalendarWeek", "PartID", "RunNum", "BatteryType", "Status", "SquarenessStatus", "NokCause",
           "ShapeMax", "DeltaDiag", "WidthDelta", "LengthDelta", "AngleDevFL", "RootCause", "_mod_key"]] \
    .sort_values("DeltaDiag", ascending=False).reset_index(drop=True)
view = view.rename(columns={"SquarenessStatus": "Squareness"})


def _style(df):
    sty = ui.status_styler(df, limit=float("inf"))
    if not isinstance(sty, pd.DataFrame):
        sty = sty.map(lambda v: "background-color:#d03b3b26;color:#a32222;font-weight:600"
                      if isinstance(v, float) and v > tol else "", subset=["DeltaDiag"])
        sty = sty.map(lambda v: "background-color:#eda10033;font-weight:600"
                      if v == "Shape + placement" else "", subset=["NokCause"])
    return sty


event = st.dataframe(
    _style(view.drop(columns="_mod_key")), width="stretch", hide_index=True,
    on_select="rerun", selection_mode="single-row", key="sq_table",
    column_config={
        "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
        "CalendarWeek": "Week", "RunNum": "Run", "BatteryType": "Type",
        "NokCause": st.column_config.TextColumn("Why NOK", help="Placement / Shape + placement / Shape"),
        "ShapeMax": st.column_config.NumberColumn("Shape uses (mm)", format="%.2f"),
        "DeltaDiag": st.column_config.NumberColumn("Diag Δ", format="%.2f"),
        "WidthDelta": st.column_config.NumberColumn("Width Δ", format="%+.2f"),
        "LengthDelta": st.column_config.NumberColumn("Length Δ", format="%+.2f"),
        "AngleDevFL": st.column_config.NumberColumn("FL angle Δ (°)", format="%+.2f"),
        "RootCause": st.column_config.TextColumn("Pattern", width="large"),
    },
)
rows = event.selection.rows if event is not None else []
if rows:
    ui.focus_module(view.iloc[rows[0]]["_mod_key"])
    st.page_link("app_pages/geometry.py", label=f"Open {view.iloc[rows[0]]['PartID']} in Geometry",
                 icon=":material/open_in_new:")
