import pandas as pd
import streamlit as st

from qc import charts, ui
from qc.constants import CAUSE_COLORS, CAUSE_ORDER, STATUS_COLORS

c = ui.ctx()
a = c.a
tol = c.settings.diag_tol

st.title("Squareness")
ui.scope_line(c)

sq = a.squareness
ui.require_rows(sq, "complete measurements (all four corners)")

ui.ensure_option("p_sq_scope", ["First run", "All runs"], "First run")
scope = st.segmented_control("Measurements", ["First run", "All runs"], key="p_sq_scope") or "First run"
if scope == "First run":
    sq = sq[sq["RunNum"] == 1]
ui.require_rows(sq, "complete first runs")

deformed = sq[sq["SquarenessStatus"] == "DEFORMED"]
k1, k2, k3 = st.columns(3)
k1.metric("Deformed", f"{len(deformed) / len(sq) * 100:.1f}%", f"{len(deformed)} of {len(sq)} modules",
          delta_color="off", border=True)
top = deformed["CauseCategory"].value_counts()
k2.metric("Most common pattern", top.index[0] if len(top) else "–",
          f"{top.iloc[0]} modules" if len(top) else None, delta_color="off", border=True)
k3.metric("Median diagonal delta", f"{sq['DeltaDiag'].median():.2f} mm", f"tolerance {tol:g} mm",
          delta_color="off", border=True)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("##### Square vs deformed")
    counts = sq.groupby(["BatteryType", "SquarenessStatus"]).size().unstack(fill_value=0)
    st.plotly_chart(charts.stacked_share(counts, ["SQUARE OK", "DEFORMED"], STATUS_COLORS),
                    width="stretch", key="sq_status")
with right:
    st.markdown("##### Pattern of the deformed ones")
    if deformed.empty:
        st.success("No deformed modules in this selection.")
    else:
        cc = deformed.groupby(["BatteryType", "CauseCategory"]).size().unstack(fill_value=0)
        st.plotly_chart(charts.stacked_share(cc, CAUSE_ORDER, CAUSE_COLORS), width="stretch", key="sq_cause")

st.markdown("##### Deformation map")
st.plotly_chart(charts.deformation_map(sq, tol), width="stretch", key="sq_map")
with st.expander("How to read this", icon=":material/help:"):
    st.markdown(
        f"- **Diagonal delta**: change of (diagonal FL–RR minus diagonal FR–RL) against nominal. "
        f"Above {tol:g} mm the module is *deformed*.\n"
        "- **Pattern**, checked in this order: *parallelogram tilt* (FL angle off by > 0.15° with "
        "little width change), *trapezoidal width* (front and rear widths differ ≥ 0.8 mm), "
        "*trapezoidal length* (left and right lengths differ ≥ 0.8 mm), else *combined asymmetry*.\n"
        "- **Grey points far from the center** are distorted *symmetrically*: both diagonals change "
        "by the same amount, so the diagonal rule can't see them. Worth a look if there are any."
    )

st.markdown("##### Modules")
st.caption("Select a row to focus that module, then open it in Geometry.")
view = sq[["Date", "CalendarWeek", "PartID", "RunNum", "BatteryType", "Diag1", "Diag2", "DeltaDiag",
           "WidthDelta", "LengthDelta", "AngleDevFL", "SquarenessStatus", "RootCause", "_mod_key"]] \
    .sort_values("DeltaDiag", ascending=False).reset_index(drop=True)
view = view.rename(columns={"SquarenessStatus": "Squareness"})


def _style(df):
    sty = ui.status_styler(df, limit=float("inf"))
    if not isinstance(sty, pd.DataFrame):
        sty = sty.map(lambda v: "background-color:#d03b3b26;color:#a32222;font-weight:600"
                      if isinstance(v, float) and v > tol else "", subset=["DeltaDiag"])
    return sty


event = st.dataframe(
    _style(view.drop(columns="_mod_key")), width="stretch", hide_index=True,
    on_select="rerun", selection_mode="single-row", key="sq_table",
    column_config={
        "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
        "CalendarWeek": "Week", "RunNum": "Run", "BatteryType": "Type",
        "Diag1": st.column_config.NumberColumn("Diag FL–RR", format="%.2f"),
        "Diag2": st.column_config.NumberColumn("Diag FR–RL", format="%.2f"),
        "DeltaDiag": st.column_config.NumberColumn("Diag Δ", format="%.2f"),
        "WidthDelta": st.column_config.NumberColumn("Width Δ", format="%+.2f"),
        "LengthDelta": st.column_config.NumberColumn("Length Δ", format="%+.2f"),
        "AngleDevFL": st.column_config.NumberColumn("FL angle Δ (°)", format="%+.2f"),
        "RootCause": st.column_config.TextColumn("Root cause", width="large"),
    },
)
rows = event.selection.rows if event is not None else []
if rows:
    key = view.iloc[rows[0]]["_mod_key"]
    ui.focus_module(key)
    st.page_link("app_pages/geometry.py", label=f"Open {view.iloc[rows[0]]['PartID']} in Geometry",
                 icon=":material/open_in_new:")
