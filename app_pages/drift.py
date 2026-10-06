import streamlit as st

from qc import charts, compensation, ui

c = ui.ctx()
a = c.a

st.title("Drift")
ui.scope_line(c)
vec = a.vec
ui.require_rows(vec)

st.caption("Where the module center lands and how it rotates, week by week (first measurement). "
           "The centroid is the average of the four corner deviations; yaw is the rotation of "
           "the rear-to-front axis against nominal.")

weeks = vec[["WeekKey", "CalendarWeek"]].drop_duplicates().sort_values("WeekKey")["WeekKey"].tolist()
ctl = st.columns([2, 1, 3], vertical_alignment="bottom")
with ctl[0]:
    if len(weeks) > 1:
        n_weeks = st.slider("Weeks on the path", 1, len(weeks), min(8, len(weeks)), key=f"dr_weeks_{len(weeks)}")
    else:
        n_weeks = 1
with ctl[1]:
    split = st.toggle("Split by type", key=ui.init("p_dr_split", True))

recent = vec[vec["WeekKey"].isin(weeks[-n_weeks:])]
left, right = st.columns(2, gap="large")
with left:
    st.markdown("##### Center path")
    st.plotly_chart(charts.drift_path(recent, split), width="stretch", key="dr_path")
    st.caption("Faint dots: modules. Line: weekly mean, first and last week labelled. "
               "Y axis reversed to match the station view.")
with right:
    st.markdown("##### Weekly median offset and yaw")
    wb = {t: compensation.weekly_bias(vec[vec["BatteryType"] == t]) for t in c.types}
    st.plotly_chart(charts.weekly_bias_chart(wb), width="stretch", key="dr_bias")

st.markdown("##### Yaw of every module")
st.plotly_chart(charts.yaw_by_module(vec), width="stretch", key="dr_yaw")

with st.expander("Table: center offset and yaw per module", icon=":material/table:"):
    tbl = vec[["Date", "CalendarWeek", "PartID", "BatteryType", "Centroid_X", "Centroid_Y",
               "Vector_Magnitude", "Rotation_Angle", "Status"]]
    st.dataframe(
        tbl, hide_index=True, width="stretch",
        column_config={
            "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
            "CalendarWeek": "Week", "BatteryType": "Type",
            "Centroid_X": st.column_config.NumberColumn("Center X (mm)", format="%+.2f"),
            "Centroid_Y": st.column_config.NumberColumn("Center Y (mm)", format="%+.2f"),
            "Vector_Magnitude": st.column_config.NumberColumn("Offset R (mm)", format="%.2f"),
            "Rotation_Angle": st.column_config.NumberColumn("Yaw (°)", format="%+.3f"),
        },
    )
