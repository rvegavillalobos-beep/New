"""Reading the raw measurement export and turning it into one row per module run.

Pipeline:
    raw file -> column detection -> timestamps (source tz -> display tz)
    -> ISO year/week -> corner mapping -> run inference -> module summary

Everything here is pure pandas/numpy (no Streamlit), so it can be tested and
cached independently of the UI.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import CORNER_INDEX, CORNERS

DEFAULT_SOURCE_TZ = "Europe/Berlin"
DEFAULT_DISPLAY_TZ = "America/Mexico_City"


class DataFormatError(ValueError):
    """Raised when the uploaded file does not look like a measurement export."""


# ---------------------------------------------------------------------------
# Domain rules (unchanged from the original app)
# ---------------------------------------------------------------------------

def determine_battery_type(part_id, feature_names) -> str:
    """Type M or Type S, from the PartID and the feature name(s)."""
    p_id = str(part_id).upper().strip()
    if isinstance(feature_names, (list, set, tuple, pd.Series, np.ndarray)):
        f_combined = " ".join(str(f).upper().strip() for f in feature_names)
    else:
        f_combined = str(feature_names).upper().strip()

    if "_DJ" in p_id or "_DJ" in f_combined or "_DI" in p_id or "_DI" in f_combined:
        return "Type M"
    if (
        "_M" in p_id or " M" in p_id or "TYPE M" in p_id
        or "TYPEM" in p_id or p_id.endswith("M")
    ):
        return "Type M"
    if "_DA" in p_id or "_DA" in f_combined:
        return "Type S"
    if "_S" in p_id or " S" in p_id or "TYPE S" in p_id or "TYPES" in p_id:
        return "Type S"
    return "Type S"


def extract_corner_index(feature_name, part_id) -> int:
    """1=FL, 2=FR, 3=RL, 4=RR, 0=not a corner feature."""
    f = str(feature_name).lower().strip()
    if not f:
        return 0
    is_type_m = determine_battery_type(part_id, feature_name) == "Type M"

    if "l0324_aa" in f or "l324_aa" in f:
        return 1
    if "r0301_aa" in f or "r301_aa" in f:
        return 2

    if is_type_m:
        if any(k in f for k in ["l0324_dj", "l324_dj", "l0324_di", "l324_di"]):
            return 3
        if any(k in f for k in ["r0301_dj", "r301_dj", "r0301_di", "r301_di"]):
            return 4
    else:
        if "l0324_da" in f or "l324_da" in f:
            return 3
        if "r0302_da" in f or "r302_da" in f:
            return 4

    if "fl" in f or "c1" in f:
        return 1
    elif "fr" in f or "c2" in f:
        return 2
    elif "rl" in f or "c3" in f:
        return 3
    elif "rr" in f or "c4" in f or "r302" in f or "r301" in f or "r00301" in f:
        return 4
    return 0


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

@dataclass
class Columns:
    time: str
    part: str
    feature: str
    x_dev: str
    y_dev: str


def _find_columns(columns) -> Columns | None:
    cols = [str(c).strip() for c in columns]
    low = [c.lower() for c in cols]

    def first(pred):
        for c, lc in zip(cols, low):
            if pred(lc):
                return c
        return None

    found = Columns(
        time=first(lambda c: "time" in c),
        part=first(lambda c: "part" in c),
        feature=first(lambda c: "feature" in c),
        x_dev=first(lambda c: "x" in c and "deviation" in c),
        y_dev=first(lambda c: "y" in c and "deviation" in c),
    )
    vals = list(vars(found).values())
    if all(vals) and len(set(vals)) == len(vals):
        return found
    return None


def _describe_missing(columns) -> str:
    cols = [str(c).strip() for c in columns]
    low = [c.lower() for c in cols]
    needs = {
        "a timestamp column (name contains 'time')": any("time" in c for c in low),
        "a part column (name contains 'part')": any("part" in c for c in low),
        "a feature column (name contains 'feature')": any("feature" in c for c in low),
        "an X deviation column (name contains 'x' and 'deviation')": any(
            "x" in c and "deviation" in c for c in low
        ),
        "a Y deviation column (name contains 'y' and 'deviation')": any(
            "y" in c and "deviation" in c for c in low
        ),
    }
    missing = [k for k, ok in needs.items() if not ok]
    shown = ", ".join(cols[:12]) + (" …" if len(cols) > 12 else "")
    return "Could not find " + "; ".join(missing) + f". Columns seen: {shown}"


def read_measurements(data: bytes, filename: str) -> tuple[pd.DataFrame, Columns, dict]:
    """Read the export and locate its columns.

    The original export has two title rows above the header (skiprows=2); that
    is tried first. If the expected columns are not there, other header
    positions (and semicolon/decimal-comma CSV) are tried before giving up.
    """
    is_csv = filename.lower().endswith(".csv")
    attempts = [2, 0, 1, 3, 4, 5, 6, 7, 8]
    last_columns = []

    def read(skip, **kw):
        buf = io.BytesIO(data)
        if is_csv:
            return pd.read_csv(buf, skiprows=skip, **kw)
        return pd.read_excel(buf, skiprows=skip)

    csv_variants = [{}, {"sep": ";", "decimal": ","}] if is_csv else [{}]
    for variant in csv_variants:
        for skip in attempts:
            try:
                df = read(skip, **variant)
            except Exception:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            last_columns = list(df.columns)
            cols = _find_columns(df.columns)
            if cols is not None:
                meta = {"header_rows_skipped": skip, "csv_variant": variant or "default"}
                return df, cols, meta
    if not last_columns:
        raise DataFormatError("The file could not be read as Excel or CSV.")
    raise DataFormatError(_describe_missing(last_columns))


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------

KNOWN_TIME_FORMATS = [
    "%b %d, %Y %I:%M%p",      # Aug 19, 2026 7:16pm  (measurement export)
    "%b %d, %Y %I:%M:%S%p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
]


def parse_timestamps(raw: pd.Series) -> tuple[pd.Series, int]:
    """Parse timestamps. Known export formats are tried first (fast and
    unambiguous); otherwise every value is parsed on its own. Values that
    still fail get a second attempt. Returns (parsed, n_recovered)."""
    parsed = None
    if not pd.api.types.is_datetime64_any_dtype(raw):
        sample = raw.dropna().astype(str).str.strip().head(200)
        for fmt in KNOWN_TIME_FORMATS:
            if len(sample) and pd.to_datetime(sample, format=fmt, errors="coerce").notna().all():
                parsed = pd.to_datetime(raw.astype(str).str.strip(), format=fmt, errors="coerce")
                break
    if parsed is None:
        # Not a known format: parse each value on its own. Letting pandas
        # infer one format from the first row is unsafe; e.g. a first row
        # with an "am" time makes it reject every "pm" time.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    failed = parsed.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
    recovered = 0
    if failed.any():
        retry = pd.to_datetime(raw[failed].astype(str), errors="coerce", format="mixed")
        recovered = int(retry.notna().sum())
        parsed = parsed.copy()
        parsed[failed] = retry
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed, recovered


def convert_timezone(naive: pd.Series, source_tz: str | None, display_tz: str | None):
    """Convert naive source-local timestamps to naive display-local timestamps.

    Daylight-saving edge cases never drop a measurement:
      * ambiguous times (the repeated hour in autumn) are read as summer time;
      * non-existent times (the skipped hour in spring) are shifted forward.
    Returns (converted, n_ambiguous, n_nonexistent).
    """
    if not source_tz or not display_tz or source_tz == display_tz:
        return naive, 0, 0
    valid = naive.notna()
    strict = naive.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
    problems = strict.isna() & valid
    n_nonexistent = 0
    if problems.any():
        shifted = naive[problems].dt.tz_localize(
            source_tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        n_nonexistent = int(shifted.notna().sum())
    n_ambiguous = int(problems.sum()) - n_nonexistent

    localized = naive.dt.tz_localize(
        source_tz,
        ambiguous=np.ones(len(naive), dtype=bool),
        nonexistent="shift_forward",
    )
    converted = localized.dt.tz_convert(display_tz).dt.tz_localize(None)
    return converted, n_ambiguous, n_nonexistent


def add_week_columns(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """ISO calendar week that survives the year boundary.

    WeekLabel ("2026-CW43") sorts chronologically as text; WeekKey (202643) is
    the numeric equivalent.
    """
    iso = df[date_col].dt.isocalendar()
    year = iso["year"].astype("Int64")
    week = iso["week"].astype("Int64")
    df["WeekKey"] = (year * 100 + week).astype("Int64")
    df["CalendarWeek"] = [
        f"{y}-CW{w:02d}" if pd.notna(y) and pd.notna(w) else None
        for y, w in zip(year, week)
    ]
    return df


# ---------------------------------------------------------------------------
# Run inference and module summary
# ---------------------------------------------------------------------------

def infer_runs(part_keys, features) -> list[int]:
    """Sequential run counter per part.

    Walking the measurements in time order, a part's run number increases when
    a feature that was already measured in the current run shows up again.
    A part keeps one continuous run history across all days it was measured.
    """
    run_tracker: dict[str, int] = {}
    seen: dict[str, set] = {}
    runs = []
    for key, feat in zip(part_keys, features):
        if key not in run_tracker:
            run_tracker[key] = 1
            seen[key] = {feat}
        elif feat in seen[key]:
            run_tracker[key] += 1
            seen[key] = {feat}
        else:
            seen[key].add(feat)
        runs.append(run_tracker[key])
    return runs


SUMMARY_COLUMNS = [
    "Date", "CalendarWeek", "WeekKey", "PartID", "BatteryType", "RunNum",
    "FL_X", "FL_Y", "FR_X", "FR_Y", "RL_X", "RL_Y", "RR_X", "RR_Y",
]


@dataclass
class LoadedData:
    raw: pd.DataFrame
    summary: pd.DataFrame  # one row per (PartID, RunNum); no status yet
    health: dict = field(default_factory=dict)
    columns: Columns | None = None


def build_dataset(
    data: bytes,
    filename: str,
    source_tz: str | None = DEFAULT_SOURCE_TZ,
    display_tz: str | None = DEFAULT_DISPLAY_TZ,
    merge_lookalike_ids: bool = False,
    zero_as_missing: bool = True,
) -> LoadedData:
    """Read an export and build the per-run module summary.

    ``merge_lookalike_ids``: treat Part IDs that differ only by letter O vs
    digit 0 as the same module (the most common spelling wins).
    ``zero_as_missing``: a reading whose X, Y (and Z, when exported) are all
    exactly 0 is almost certainly a failed read, not a perfect corner; treat
    it as a missing corner.
    """
    df, cols, meta = read_measurements(data, filename)
    health: dict = {"file_name": filename, "rows_in_file": int(len(df)), **meta}
    health["columns_used"] = vars(cols).copy()

    df = df.dropna(how="all").copy()
    health["blank_rows"] = health["rows_in_file"] - int(len(df))
    n_before = len(df)
    df = df.drop_duplicates().copy()
    health["duplicate_rows_removed"] = n_before - int(len(df))
    health["limits_in_file"] = _limits_in_file(df)

    parsed, recovered = parse_timestamps(df[cols.time])
    health["timestamps_recovered"] = recovered
    converted, n_amb, n_nonexist = convert_timezone(parsed, source_tz, display_tz)
    health["dst_ambiguous_rows"] = n_amb
    health["dst_nonexistent_rows"] = n_nonexist
    df["ParsedDate"] = converted

    bad_time = df["ParsedDate"].isna()
    health["rows_without_valid_time"] = int(bad_time.sum())
    df = df[~bad_time].copy()

    df["PartKey"] = df[cols.part].astype(str)
    lookalikes = _similar_part_ids(df["PartKey"].unique())
    health["similar_part_ids"] = lookalikes
    health["lookalike_ids_merged"] = 0
    if merge_lookalike_ids and lookalikes:
        counts = df["PartKey"].value_counts()
        mapping = {}
        for group in lookalikes:
            keep = max(group, key=lambda g: (counts.get(g, 0), g.count("O")))
            mapping.update({g: keep for g in group if g != keep})
        df["PartKey"] = df["PartKey"].replace(mapping)
        df[cols.part] = df["PartKey"]
        health["lookalike_ids_merged"] = len(mapping)
    df["FeatureName"] = df[cols.feature].astype(str)
    df["X_Val"] = pd.to_numeric(df[cols.x_dev], errors="coerce")
    df["Y_Val"] = pd.to_numeric(df[cols.y_dev], errors="coerce")
    health["rows_with_non_numeric_deviation"] = int(
        (df["X_Val"].isna() & df[cols.x_dev].notna()).sum()
        + (df["Y_Val"].isna() & df[cols.y_dev].notna()).sum()
    )
    z_col = next((c for c in df.columns if c.lower().startswith("z") and "deviation" in c.lower()), None)
    zero = (df["X_Val"] == 0) & (df["Y_Val"] == 0)
    if z_col is not None:
        zero &= pd.to_numeric(df[z_col], errors="coerce") == 0
    health["zero_readings"] = int(zero.sum())
    health["zero_reading_features"] = df.loc[zero, "FeatureName"].value_counts().to_dict()
    health["zero_as_missing"] = bool(zero_as_missing)
    if zero_as_missing and zero.any():
        df.loc[zero, ["X_Val", "Y_Val"]] = np.nan

    # Corner mapping evaluated once per unique (part, feature) pair.
    pairs = df[["PartKey", "FeatureName", cols.part, cols.feature]].drop_duplicates(
        ["PartKey", "FeatureName"]
    )
    pairs["CornerIndex"] = [
        extract_corner_index(f, p) for p, f in zip(pairs[cols.part], pairs[cols.feature])
    ]
    df = df.merge(pairs[["PartKey", "FeatureName", "CornerIndex"]],
                  on=["PartKey", "FeatureName"], how="left")

    unmapped = df[df["CornerIndex"] == 0]
    health["rows_unmapped_feature"] = int(len(unmapped))
    health["unmapped_features"] = (
        unmapped["FeatureName"].value_counts().head(15).to_dict()
    )

    df = df.sort_values("ParsedDate", kind="stable").reset_index(drop=True)
    df["RunNum"] = infer_runs(df["PartKey"].tolist(), df["FeatureName"].tolist())
    df = add_week_columns(df, "ParsedDate")

    summary = _summarize_modules(df, cols)
    health["modules"] = int(summary["PartID"].nunique()) if not summary.empty else 0
    health["module_runs"] = int(len(summary))
    health["repeat_runs"] = int((summary["RunNum"] > 1).sum()) if not summary.empty else 0

    dup = (
        df[df["CornerIndex"] > 0]
        .groupby(["PartKey", "RunNum", "CornerIndex"]).size()
    )
    health["corner_measured_twice_in_a_run"] = int((dup > 1).sum())

    if not summary.empty:
        health["date_min"] = summary["Date"].min()
        health["date_max"] = summary["Date"].max()
    return LoadedData(raw=df, summary=summary, health=health, columns=cols)


def _limits_in_file(df: pd.DataFrame) -> dict:
    """Tolerance limits exported next to the deviations, if present
    (e.g. 'X lower limit' / 'X upper limit')."""
    out = {}
    for axis in ("X", "Y", "Z"):
        lo = [c for c in df.columns if c.lower().startswith(axis.lower()) and "lower" in c.lower()]
        hi = [c for c in df.columns if c.lower().startswith(axis.lower()) and "upper" in c.lower()]
        if lo and hi:
            lo_v = pd.to_numeric(df[lo[0]], errors="coerce").dropna()
            hi_v = pd.to_numeric(df[hi[0]], errors="coerce").dropna()
            if len(lo_v) and len(hi_v):
                out[axis] = (float(lo_v.mode().iloc[0]), float(hi_v.mode().iloc[0]))
    return out


def symmetric_xy_limit(limits: dict) -> float | None:
    """The ±limit when X and Y share one symmetric tolerance, else None."""
    x, y = limits.get("X"), limits.get("Y")
    if x and y and x == y and abs(x[0] + x[1]) < 1e-9 and x[1] > 0:
        return float(x[1])
    return None


def _similar_part_ids(part_ids) -> list[tuple[str, str]]:
    """Pairs of Part IDs that differ only by letter O vs digit 0 (likely a
    typing/scanning slip that would split one module's history)."""
    groups: dict[str, list[str]] = {}
    for pid in part_ids:
        groups.setdefault(re.sub("[Oo]", "0", str(pid)), []).append(str(pid))
    return [tuple(sorted(v)) for v in groups.values() if len(v) > 1]


def _summarize_modules(df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    keys = ["PartKey", "RunNum"]
    first = df.groupby(keys, sort=False).first()  # first row of each run
    base = pd.DataFrame({
        "Date": first["ParsedDate"],
        "CalendarWeek": first["CalendarWeek"],
        "WeekKey": first["WeekKey"],
        "PartID": first[cols.part],
    })
    # A module's type cannot change between runs, so it is decided from all
    # features ever measured on that part (a run with only the front corners
    # would otherwise fall back to Type S).
    part_feats = df.groupby("PartKey", sort=False)["FeatureName"].agg(list)
    part_ids = df.groupby("PartKey", sort=False)[cols.part].first()
    part_type = {
        k: determine_battery_type(part_ids[k], part_feats[k]) for k in part_feats.index
    }
    base["BatteryType"] = [part_type[k] for k in base.index.get_level_values("PartKey")]

    # Last measurement of each corner in the run wins (NaN included).
    corners = df[df["CornerIndex"].between(1, 4)].drop_duplicates(
        keys + ["CornerIndex"], keep="last"
    )
    for idx, name in CORNER_INDEX.items():
        sub = corners[corners["CornerIndex"] == idx].set_index(keys)
        base[f"{name}_X"] = sub["X_Val"].reindex(base.index)
        base[f"{name}_Y"] = sub["Y_Val"].reindex(base.index)

    base = base.reset_index()
    base = base.rename(columns={"PartKey": "BaseKey"})
    base = base.sort_values(["Date", "RunNum"], kind="stable").reset_index(drop=True)
    for c in CORNERS:
        base[f"{c}_X"] = base[f"{c}_X"].astype(float)
        base[f"{c}_Y"] = base[f"{c}_Y"].astype(float)
    return base[SUMMARY_COLUMNS + ["BaseKey"]]
