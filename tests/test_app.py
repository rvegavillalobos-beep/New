"""Smoke tests of every page with Streamlit's own AppTest (skipped if
Streamlit is not installed)."""
import os

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from qc import demo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["overview", "corners", "geometry", "squareness", "drift", "compensation", "data"]


def _app():
    at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=120)
    return at


def test_start_page_without_data():
    at = _app().run()
    assert not at.exception
    assert any("Corner QC" in t.value for t in at.title)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_with_demo_data(page):
    at = _app()
    at.session_state["_file"] = {"name": "demo", "data": demo.demo_bytes(), "demo": True}
    at.run()
    assert not at.exception
    at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
