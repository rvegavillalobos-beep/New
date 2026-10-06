"""Streamlit glue: data source, sidebar, cached computations, shared context.

Pages call ``ctx()`` to get the same filtered analysis the sidebar defines.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import analysis, compensation, demo, geometry, loading
from .constants import BATTERY_TYPES

PERSIST_PREFIX = "p_"
TIMEZONES = [
    "Europe/Berlin", "America/Mexico_City", "UTC", "America/Chicago",
    "America/New_York", "America/Los_Angeles", "Europe/London", "Asia/Shanghai",
]


# ---------------------------------------------------------------------------
# Widget state that survives page switches
# ---------------------------------------------------------------------------

def keep_widget_state() -> None:
    """Streamlit drops the state of widgets that are not rendered on the
    current page. Re-assigning them at the top of every run keeps page
    settings when the user navigates away and back."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(PERSIST_PREFIX):
            st.session_state[k] = st.session_state[k]


def init(key: str, value) -> str:
    """Give a persisted widget its default once; widgets using it must not
    pass value/index/default themselves."""
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def ensure_option(key: str, options: list, fallback) -> str:
    """Reset a persisted selection that is no longer a valid option."""
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = fallback
    return key


def reset_persisted() -> None:
    """Forget filters and page settings (used when a new file is loaded)."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (k.startswith(PERSIST_PREFIX) or k.startswith("sb_") or k == "focus"):
            del st.session_state[k]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.html("""
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.75rem;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem;}
      .qc-scope {color: #6b6a65; font-size: 0.9rem; margin-top: -0.6rem; margin-bottom: 0.8rem;}
      .qc-finding {margin: 0.15rem 0 0.45rem 0; line-height: 1.45;}
    </style>
    """)


def scope_line(c: "Context") -> None:
    types = " & ".join(c.scope.types) if len(c.scope.types) < 2 else "All types"
    d0, d1 = c.a.first["Date"].min(), c.a.first["Date"].max()
    span = f"{d0:%b %d} – {d1:%b %d, %Y}" if pd.notna(d0) else "no dates"
    st.markdown(
        f"<div class='qc-scope'>{c.file_name} · {len(c.a.first)} modules (first run) · "
        f"{span} · {c.week_label} · {types}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------

def _code_version() -> str:
    """Fingerprint of the analysis code. Passed to every cached function so
    a deployment with changed logic never reuses results computed by the
    previous version (Streamlit only keys its cache on the cached function's
    own code, not on the code it calls)."""
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


CODE_VERSION = _code_version()


@st.cache_data(show_spinner="Reading measurements…", max_entries=6)
def _load(data: bytes, name: str, src_tz: str, disp_tz: str, merge_ids: bool,
          zero_missing: bool, code_version: str) -> loading.LoadedData:
    return loading.build_dataset(data, name, src_tz, disp_tz, merge_ids, zero_missing)


def load(data, name, src_tz, disp_tz, merge_ids, zero_missing) -> loading.LoadedData:
    return _load(data, name, src_tz, disp_tz, merge_ids, zero_missing, CODE_VERSION)


@st.cache_data(show_spinner=False, max_entries=24)
def _run_analysis(summary: pd.DataFrame, settings: analysis.Settings, scope: analysis.Scope,
                  ma_window: int, code_version: str) -> analysis.Analysis:
    return analysis.analyze(summary, settings, scope, ma_window)


def run_analysis(summary, settings, scope, ma_window) -> analysis.Analysis:
    return _run_analysis(summary, settings, scope, ma_window, CODE_VERSION)


@st.cache_data(show_spinner="Searching the best compensation…", max_entries=24)
def _run_optimize(df: pd.DataFrame, spec: float, margin: float, yaw_center: float, code_version: str):
    return compensation.optimize(df, spec, margin, yaw_center=yaw_center)


def run_optimize(df, spec, margin, yaw_center):
    return _run_optimize(df, spec, margin, yaw_center, CODE_VERSION)


@st.cache_data(show_spinner="Replaying past weeks…", max_entries=24)
def _run_backtest(df: pd.DataFrame, spec: float, margin: float, window_weeks: int, method: str,
                  code_version: str):
    return compensation.backtest(df, spec, margin, window_weeks, method)


def run_backtest(df, spec, margin, window_weeks, method):
    return _run_backtest(df, spec, margin, window_weeks, method, CODE_VERSION)


@st.cache_data(show_spinner=False, max_entries=6)
def _demo_file(code_version: str) -> bytes:
    return demo.demo_bytes()


def demo_file() -> bytes:
    return _demo_file(CODE_VERSION)


# ---------------------------------------------------------------------------
# Data source (kept in session state so it survives page switches)
# ---------------------------------------------------------------------------

def _store_upload(widget_key: str) -> None:
    f = st.session_state.get(widget_key)
    if f is not None:
        st.session_state["_file"] = {"name": f.name, "data": f.getvalue(), "demo": False}
        reset_persisted()


def use_demo() -> None:
    st.session_state["_file"] = {"name": "demo data (synthetic)", "data": demo_file(), "demo": True}
    reset_persisted()


def clear_file() -> None:
    st.session_state.pop("_file", None)
    reset_persisted()


def uploader() -> None:
    st.file_uploader("Measurement export (Excel or CSV)", type=["xlsx", "xls", "csv"],
                     key="up_main", on_change=_store_upload, args=("up_main",))


def current_file() -> dict | None:
    return st.session_state.get("_file")


# ---------------------------------------------------------------------------
# Context shared by all pages
# ---------------------------------------------------------------------------

@dataclass
class Context:
    file_name: str
    is_demo: bool
    data: loading.LoadedData
    a: analysis.Analysis
    settings: analysis.Settings
    scope: analysis.Scope
    week_label: str
    ma_window: int

    @property
    def types(self) -> list[str]:
        return [t for t in BATTERY_TYPES if t in self.scope.types and (self.a.first["BatteryType"] == t).any()]


def build_context() -> Context | None:
    """Sidebar (file card, filters, rules) + cached load and analysis.

    Sidebar widgets are rendered on every page, so they keep their state on
    their own ("sb_" keys, normal defaults). Page widgets use "p_" keys.
    """
    f = current_file()
    if f is None:
        return None
    sb = st.sidebar
    ss = st.session_state

    src_tz = ss.get("sb_src_tz", "Europe/Berlin")
    disp_tz = ss.get("sb_disp_tz", "America/Mexico_City")
    merge_ids = ss.get("sb_merge_ids", False)
    zero_missing = ss.get("sb_zero_missing", True)
    try:
        data = load(f["data"], f["name"], src_tz, disp_tz, merge_ids, zero_missing)
    except loading.DataFormatError as e:
        st.error(f"**This file doesn't look like a corner deviation export.** {e}")
        _file_controls(sb, f, None)
        return None
    except Exception as e:  # noqa: BLE001 - show any reading problem to the user
        st.error(f"**The file could not be read.** {type(e).__name__}: {e}")
        _file_controls(sb, f, None)
        return None
    if data.summary.empty:
        st.warning("The file was read, but it contains no corner measurements with a valid time.")
        _file_controls(sb, f, data)
        return None

    _file_controls(sb, f, data)

    # ---- Filters -------------------------------------------------------------
    summary = data.summary
    present = [t for t in BATTERY_TYPES if (summary["BatteryType"] == t).any()]
    type_opts = (["All types"] if len(present) > 1 else []) + present
    with sb:
        st.subheader("Filters", divider="gray")
        if ss.get("sb_types") not in type_opts:
            ss.pop("sb_types", None)
        type_choice = st.segmented_control("Battery type", type_opts, default=type_opts[0],
                                           key="sb_types")
        if type_choice is None:  # clicking the active option deselects it
            type_choice = type_opts[0]
        types = tuple(present) if type_choice == "All types" else (type_choice,)

        weeks = (summary[["WeekKey", "CalendarWeek"]].dropna().drop_duplicates()
                 .sort_values("WeekKey"))
        labels = weeks["CalendarWeek"].tolist()
        keys = weeks["WeekKey"].tolist()
        if len(labels) > 1:
            cur = ss.get("sb_weeks")
            if cur is not None and (not isinstance(cur, (tuple, list)) or any(w not in labels for w in cur)):
                ss.pop("sb_weeks")
            w0, w1 = st.select_slider("Calendar weeks", options=labels,
                                      value=(labels[0], labels[-1]), key="sb_weeks")
            wf, wt = keys[labels.index(w0)], keys[labels.index(w1)]
        else:
            w0 = w1 = labels[0] if labels else "-"
            wf = wt = keys[0] if keys else None
            st.caption(f"Calendar week: {w0}")
        week_label = w0 if w0 == w1 else f"{w0} – {w1}"

    # ---- Rules ---------------------------------------------------------------
    file_limit = loading.symmetric_xy_limit(data.health.get("limits_in_file", {}))
    with sb:
        st.subheader("Rules", divider="gray")
        spec = st.slider("X/Y tolerance (± mm)", 0.5, 6.0, float(file_limit) if file_limit else 3.0,
                         step=0.1, key="sb_spec",
                         help="A module passes when every corner's X and Y deviation is within ± this value.")
        if file_limit and abs(file_limit - spec) > 1e-9:
            st.caption(f":orange[The file's own limit is ±{file_limit:g} mm.]")
        target = st.slider("FPY target (%)", 50.0, 100.0, 90.0, step=1.0, key="sb_target")
        diag = st.slider("Max diagonal delta (mm)", 0.5, 10.0, 1.5, step=0.1, key="sb_diag",
                         help="Above this, a module is flagged DEFORMED in the squareness analysis.")
        excl = st.toggle("Count incomplete first runs as NOK", value=True, key="sb_excl",
                         help="On: a first measurement with a missing corner is a FAIL (strict). "
                              "Off: it is kept apart as INCOMPLETE (it still lowers FPY).")

        with st.expander("Advanced"):
            st.selectbox("Time zone of the export", TIMEZONES, index=TIMEZONES.index("Europe/Berlin"),
                         key="sb_src_tz", help="Timestamps in the file are read in this zone…")
            st.selectbox("Show times in", TIMEZONES, index=TIMEZONES.index("America/Mexico_City"),
                         key="sb_disp_tz",
                         help="…and converted to this one. Calendar weeks follow the converted time.")
            n_sim = len(data.health.get("similar_part_ids", []))
            st.toggle(f"Merge look-alike Part IDs (O vs 0){f' · {n_sim} found' if n_sim else ''}",
                      value=False, key="sb_merge_ids",
                      help="Treat IDs that differ only by letter O vs digit 0 as the same module.")
            n_zero = data.health.get("zero_readings", 0)
            st.toggle(f"All-zero readings are missing{f' · {n_zero} found' if n_zero else ''}",
                      value=True, key="sb_zero_missing",
                      help="A corner with X, Y and Z all exactly 0.000 is almost certainly a failed read. "
                           "On: counted as a missing corner. Off: used as a real (perfect) measurement.")

    init("p_ma", 3)
    settings = analysis.Settings(spec_limit=float(spec), diag_tol=float(diag),
                                 fpy_target=float(target), exclude_incomplete=bool(excl))
    scope = analysis.Scope(types=types, week_from=wf, week_to=wt)
    a = run_analysis(summary, settings, scope, int(ss["p_ma"]))
    return Context(file_name=f["name"], is_demo=bool(f.get("demo")), data=data, a=a,
                   settings=settings, scope=scope, week_label=week_label,
                   ma_window=int(ss["p_ma"]))


def _file_controls(sb, f: dict, data) -> None:
    with sb:
        st.markdown(f"**{f['name']}**")
        if data is not None and not data.summary.empty:
            h = data.health
            st.caption(f"{h['modules']} modules · {h['module_runs']} runs · "
                       f"{h['date_min']:%b %d} – {h['date_max']:%b %d, %Y}")
        if f.get("demo"):
            st.caption(":orange[Synthetic demo data: the numbers are invented.]")
        with st.expander("Replace or clear file"):
            st.file_uploader("Replace file", type=["xlsx", "xls", "csv"], key="up_sidebar",
                             on_change=_store_upload, args=("up_sidebar",),
                             label_visibility="collapsed")
            st.button("Clear file", on_click=clear_file, icon=":material/close:", width="stretch")


def ctx() -> Context:
    c = st.session_state.get("_ctx")
    if c is None:
        st.info("Load a measurement file first.")
        st.stop()
    return c


def require_rows(df: pd.DataFrame, what: str = "modules") -> None:
    if df is None or len(df) == 0:
        st.info(f"No {what} in the current filters. Widen the calendar weeks or battery type in the sidebar.")
        st.stop()


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

LEVEL_ICON = {
    "good": ":green[:material/check_circle:]",
    "warn": ":orange[:material/warning:]",
    "bad": ":red[:material/error:]",
    "info": ":blue[:material/info:]",
}


def findings_box(items: list[dict]) -> None:
    for it in items:
        st.markdown(f"{LEVEL_ICON.get(it['level'], '')}&nbsp; {it['text']}")


def focus_module(key: str | None) -> None:
    if key:
        st.session_state["focus"] = key


def fmt_pct(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.1f}%"


def status_styler(df: pd.DataFrame, limit: float, max_rows: int = 4000):
    """Red cells for out-of-tolerance deviations, colored status cells.
    Plain dataframe above ``max_rows`` (styling large tables is slow)."""
    if len(df) > max_rows:
        return df
    dev_cols = [c for c in geometry.DEV_COLS if c in df.columns]
    colors = {"PASS": "background-color:#0ca30c22;color:#0b6b0b;font-weight:600",
              "FAIL": "background-color:#d03b3b22;color:#a32222;font-weight:600",
              "INCOMPLETE": "background-color:#89878133;color:#52514e;font-weight:600",
              "DEFORMED": "background-color:#d03b3b22;color:#a32222;font-weight:600",
              "SQUARE OK": "color:#0b6b0b"}

    def dev_style(v):
        try:
            return "background-color:#d03b3b26;color:#a32222;font-weight:600" if abs(float(v)) > limit else ""
        except (TypeError, ValueError):
            return ""

    sty = df.style
    if dev_cols:
        sty = sty.map(dev_style, subset=dev_cols)
    for col in ("Status", "Status_Sim", "Squareness"):
        if col in df.columns:
            sty = sty.map(lambda v: colors.get(str(v), ""), subset=[col])
    return sty.format(precision=2, na_rep="–")


def excel_report(sheets: dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue
            d = df.copy()
            for c in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    d[c] = d[c].dt.tz_localize(None) if getattr(d[c].dt, "tz", None) else d[c]
            d.to_excel(w, sheet_name=name[:31], index=False)
            ws = w.sheets[name[:31]]
            for i, col in enumerate(d.columns, start=1):
                width = min(45, max(10, len(str(col)) + 2))
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
            ws.freeze_panes = "A2"
    return out.getvalue()

