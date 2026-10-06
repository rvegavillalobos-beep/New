import streamlit as st

from qc import ui

st.title("Corner QC")
st.markdown(
    "First-measurement quality of battery modules at **ST020**: how far each corner lands "
    "from nominal, which corners and axes cause NOK, whether modules are deformed, how the "
    "placement drifts week to week, and what X / Y / yaw compensation would do."
)

left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(border=True):
        st.markdown("##### Load a measurement export")
        ui.uploader()
        st.caption(
            "The *Corner Cell Deviations* export: two title rows, then columns for Time, Part ID, "
            "Featurename and X / Y deviation. Excel or CSV."
        )
    st.button("Try it with demo data", icon=":material/science:", on_click=ui.use_demo,
              help="Loads a synthetic file in the same format. The numbers are invented.")

with right:
    st.markdown("##### What you get")
    st.markdown(
        "- **Overview**: FPY against target, weekly trend and the main reasons for NOK\n"
        "- **Corners**: where each corner sits, offset vs scatter, capability\n"
        "- **Geometry**: every module drawn against nominal and the tolerance zones\n"
        "- **Squareness**: deformation and its likely pattern\n"
        "- **Drift**: weekly movement of the module center and yaw\n"
        "- **Compensation**: suggested X / Y / yaw, simulated FPY, readiness check\n"
        "- **Data & export**: data quality report and a full Excel export"
    )
