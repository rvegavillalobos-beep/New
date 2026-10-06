"""Corner QC: ST020 first-measurement quality, geometry and compensation.

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Corner QC", page_icon=":material/crop_free:", layout="wide")

from qc import ui  # noqa: E402  (page config must come first)

ui.keep_widget_state()
ui.inject_css()

context = ui.build_context()
st.session_state["_ctx"] = context

if context is None:
    pages = [st.Page("app_pages/start.py", title="Start", icon=":material/upload_file:", default=True)]
    nav = st.navigation(pages)
else:
    pages = [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/corners.py", title="Corners", icon=":material/crop_free:"),
        st.Page("app_pages/geometry.py", title="Geometry", icon=":material/category:"),
        st.Page("app_pages/squareness.py", title="Squareness", icon=":material/square_foot:"),
        st.Page("app_pages/drift.py", title="Drift", icon=":material/timeline:"),
        st.Page("app_pages/compensation.py", title="Compensation", icon=":material/tune:"),
        st.Page("app_pages/data.py", title="Data & export", icon=":material/table_view:"),
    ]
    nav = st.navigation(pages, position="top")

nav.run()
