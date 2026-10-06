import streamlit as st

from qc import charts, metrics, ui
from qc.constants import TYPE_COLORS

c = ui.ctx()
a = c.a
limit = c.settings.spec_limit

st.title("Corners")
ui.scope_line(c)
ui.require_rows(a.first)

types = c.types
if len(types) > 1:
    ui.ensure_option("p_corner_type", types, types[0])
    t = st.segmented_control("Battery type", types, key="p_corner_type")
    t = t or types[0]
else:
    t = types[0]

d = a.first[a.first["BatteryType"] == t]
cap = a.capability[a.capability["BatteryType"] == t].copy()

st.markdown(f"##### Where each corner lands · {t} · first measurement")
left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(charts.corner_clouds(d, limit, TYPE_COLORS[t]), width="stretch", key="cn_clouds")
    st.caption("One dot per module; the dotted box is the tolerance, ✕ is the average. "
               "A cloud **off-center** means a systematic offset (a compensation can fix it). "
               "A **wide** cloud means variation (an offset can't fix it).")
with right:
    st.markdown("**Capability per corner and axis**")
    if cap.empty:
        st.info("No complete corner data.")
    else:
        cap["Where"] = cap["Corner"] + " " + cap["Axis"]
        cap["Diagnosis"] = [metrics.diagnose(m, cp, ck, limit)
                            for m, cp, ck in zip(cap["Mean"], cap["Cp"], cap["Cpk"])]
        view = cap[["Where", "N", "Mean", "Std", "OutPct", "Cpk", "Diagnosis"]].sort_values("OutPct", ascending=False)
        st.dataframe(
            view, hide_index=True, width="stretch", height=38 + 35 * len(view),
            column_config={
                "Where": "Corner",
                "Mean": st.column_config.NumberColumn("Mean (mm)", format="%+.2f"),
                "Std": st.column_config.NumberColumn("σ (mm)", format="%.2f"),
                "OutPct": st.column_config.ProgressColumn("Out of tol.", format="%.0f%%", min_value=0, max_value=100),
                "Cpk": st.column_config.NumberColumn("Cpk", format="%.2f",
                                                     help="≥ 1.33 capable · 1.0–1.33 marginal · < 1.0 not capable"),
                "Diagnosis": st.column_config.TextColumn(
                    "Diagnosis", help="offset: average ≥ ⅓ of the tolerance away from 0 (a compensation can "
                                      "remove it) · scatter: spread too wide for the tolerance (Cp < 1; an "
                                      "offset can't fix it) · ok: Cpk ≥ 1.33 · marginal: in between"),
            },
        )
        worst = view.iloc[0]
        st.caption(f"Worst: **{worst['Where']}**, out of tolerance in {worst['OutPct']:.0f}% of modules "
                   f"(mean {worst['Mean']:+.2f} mm, σ {worst['Std']:.2f} mm).")

st.markdown(f"##### Corner deviation over time · {t}")
st.caption("Each module in time order (first measurement). Hover a point for its Part ID and date.")
r1, r2 = st.columns(2), st.columns(2)
for col, corner in zip([r1[0], r1[1], r2[0], r2[1]], ["FL", "FR", "RL", "RR"]):
    with col:
        st.plotly_chart(charts.corner_trend(d, corner, limit), width="stretch", key=f"cn_trend_{corner}")
