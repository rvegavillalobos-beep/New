"""Geometry on module corners: status, centroid/yaw, squareness, rigid moves.

All functions are vectorized over modules (rows) and return new dataframes;
inputs are never modified.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .constants import ANGULAR_DEV_TOL, CORNERS, DIM_DELTA_TOL, NOMINALS

DEV_COLS = [f"{c}_{a}" for c in CORNERS for a in ("X", "Y")]


def normalize_type(bat_type) -> str:
    """Same fallback as the original app: anything that is not Type S is Type M."""
    return "Type S" if str(bat_type).upper() == "TYPE S" else "Type M"


def nominal_center(bat_type) -> tuple[float, float]:
    nom = NOMINALS[normalize_type(bat_type)]
    cx = sum(nom[c][0] for c in CORNERS) / 4.0
    cy = sum(nom[c][1] for c in CORNERS) / 4.0
    return cx, cy


def _nominal_arrays(types: pd.Series) -> dict[str, np.ndarray]:
    """Per-row nominal coordinates, keyed like the deviation columns."""
    norm = types.map(normalize_type)
    out = {}
    for c in CORNERS:
        out[f"{c}_X"] = norm.map(lambda t, c=c: NOMINALS[t][c][0]).to_numpy(float)
        out[f"{c}_Y"] = norm.map(lambda t, c=c: NOMINALS[t][c][1]).to_numpy(float)
    return out


def _dev(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype=float)


def is_complete(df: pd.DataFrame) -> pd.Series:
    return df[DEV_COLS].notna().all(axis=1)


# ---------------------------------------------------------------------------
# PASS / FAIL / INCOMPLETE
# ---------------------------------------------------------------------------

def corner_out_of_spec(df: pd.DataFrame, spec_limit: float, suffix: str = "") -> pd.DataFrame:
    """Boolean frame: is corner c / axis a outside ±spec_limit (NaN -> False)."""
    out = {}
    for c in CORNERS:
        for a in ("X", "Y"):
            v = _dev(df, f"{c}_{a}{suffix}")
            out[f"{c}_{a}"] = np.abs(v) > spec_limit
    return pd.DataFrame(out, index=df.index)


def evaluate_status(df: pd.DataFrame, spec_limit: float, suffix: str = "") -> pd.DataFrame:
    """Adds CornersOutOfSpec, MissingCorners, IsComplete and Status.

    A corner is missing when its X or Y is missing. A complete module FAILs
    when any corner has |X| or |Y| above the limit.
    """
    out = df.copy()
    oos = np.zeros(len(df), dtype=int)
    missing = np.zeros(len(df), dtype=int)
    for c in CORNERS:
        x, y = _dev(df, f"{c}_X{suffix}"), _dev(df, f"{c}_Y{suffix}")
        miss = np.isnan(x) | np.isnan(y)
        missing += miss
        oos += (~miss) & ((np.abs(x) > spec_limit) | (np.abs(y) > spec_limit))
    out[f"CornersOutOfSpec{suffix}"] = oos
    out[f"MissingCorners{suffix}"] = missing
    out[f"IsComplete{suffix}"] = missing == 0
    out[f"Status{suffix}"] = np.where(missing > 0, "INCOMPLETE", np.where(oos > 0, "FAIL", "PASS"))
    return out


# ---------------------------------------------------------------------------
# Centroid and yaw
# ---------------------------------------------------------------------------

def add_centroid_and_rotation(df: pd.DataFrame) -> pd.DataFrame:
    """Centroid of the corner deviations, its magnitude, and yaw.

    Yaw is the angle change of the line from the rear-corners midpoint to the
    front-corners midpoint, relative to nominal, wrapped to (-180, 180].
    """
    out = df.copy()
    xs = df[[f"{c}_X" for c in CORNERS]].to_numpy(float)
    ys = df[[f"{c}_Y" for c in CORNERS]].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        out["Centroid_X"] = np.nanmean(xs, axis=1) if len(df) else np.array([], float)
        out["Centroid_Y"] = np.nanmean(ys, axis=1) if len(df) else np.array([], float)
    out["Vector_Magnitude"] = np.hypot(out["Centroid_X"], out["Centroid_Y"])
    out["Rotation_Angle"] = yaw_angles(df)
    return out


def yaw_angles(df: pd.DataFrame, suffix: str = "") -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    nom = _nominal_arrays(df["BatteryType"])
    fnx = (nom["FL_X"] + nom["FR_X"]) / 2
    fny = (nom["FL_Y"] + nom["FR_Y"]) / 2
    rnx = (nom["RL_X"] + nom["RR_X"]) / 2
    rny = (nom["RL_Y"] + nom["RR_Y"]) / 2
    angle_nom = np.degrees(np.arctan2(fny - rny, fnx - rnx))

    act = {k: nom[k] + _dev(df, f"{k}{suffix}") for k in nom}
    fax = (act["FL_X"] + act["FR_X"]) / 2
    fay = (act["FL_Y"] + act["FR_Y"]) / 2
    rax = (act["RL_X"] + act["RR_X"]) / 2
    ray = (act["RL_Y"] + act["RR_Y"]) / 2
    angle_act = np.degrees(np.arctan2(fay - ray, fax - rax))
    return (angle_act - angle_nom + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Squareness / deformation
# ---------------------------------------------------------------------------

def _dist(ax, ay, bx, by):
    return np.hypot(bx - ax, by - ay)


def _corner_angle(a, b, c):
    """Angle at a between a->b and a->c, degrees (arrays of (x, y))."""
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - a[0], c[1] - a[1]
    dot = v1x * v2x + v1y * v2y
    mag = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / mag, -1.0, 1.0)
    return np.where(mag == 0, 0.0, np.degrees(np.arccos(cos)))


def squareness(df: pd.DataFrame, max_diag_tol: float) -> pd.DataFrame:
    """Deformation metrics for complete modules.

    Diagonal delta is the change of (diag FL-RR minus diag FR-RL) versus
    nominal. Above the tolerance the module is DEFORMED and the dominant
    pattern is named, checked in this order: parallelogram tilt, trapezoidal
    width, trapezoidal length, combined asymmetry.
    """
    d = df[is_complete(df)].copy()
    if d.empty:
        cols = ["DeltaDiag", "Diag1", "Diag2", "WidthDelta", "LengthDelta",
                "AngleDevFL", "SquarenessStatus", "RootCause", "CauseCategory"]
        return d.assign(**{c: pd.Series(dtype=float) for c in cols})

    n = _nominal_arrays(d["BatteryType"])
    a = {k: n[k] + _dev(d, k) for k in n}

    def metrics(p):
        d1 = _dist(p["FL_X"], p["FL_Y"], p["RR_X"], p["RR_Y"])
        d2 = _dist(p["FR_X"], p["FR_Y"], p["RL_X"], p["RL_Y"])
        w_top = _dist(p["FL_X"], p["FL_Y"], p["FR_X"], p["FR_Y"])
        w_bot = _dist(p["RL_X"], p["RL_Y"], p["RR_X"], p["RR_Y"])
        l_left = _dist(p["FL_X"], p["FL_Y"], p["RL_X"], p["RL_Y"])
        l_right = _dist(p["FR_X"], p["FR_Y"], p["RR_X"], p["RR_Y"])
        ang = _corner_angle((p["FL_X"], p["FL_Y"]), (p["FR_X"], p["FR_Y"]), (p["RL_X"], p["RL_Y"]))
        return d1, d2, w_top, w_bot, l_left, l_right, ang

    d1n, d2n, wtn, wbn, lln, lrn, angn = metrics(n)
    d1a, d2a, wta, wba, lla, lra, anga = metrics(a)

    delta = np.abs((d1a - d2a) - (d1n - d2n))
    width_delta = (wta - wtn) - (wba - wbn)
    length_delta = (lla - lln) - (lra - lrn)
    ang_dev = anga - angn

    deformed = delta > max_diag_tol
    is_par = deformed & (np.abs(ang_dev) > ANGULAR_DEV_TOL) & (np.abs(width_delta) < DIM_DELTA_TOL)
    is_w = deformed & ~is_par & (np.abs(width_delta) >= DIM_DELTA_TOL)
    is_l = deformed & ~is_par & ~is_w & (np.abs(length_delta) >= DIM_DELTA_TOL)
    is_c = deformed & ~is_par & ~is_w & ~is_l

    category = np.select(
        [is_par, is_w, is_l, is_c],
        ["Parallelogram Tilt", "Trapezoidal Width", "Trapezoidal Length", "Combined Asymmetry"],
        default="",
    )
    detail = np.select(
        [is_par, is_w, is_l, is_c],
        [
            [f"PARALLELOGRAM DISTORTION (Tilt: {v:+.2f}°)" for v in ang_dev],
            [f"TRAPEZOIDAL WIDTH VARIATION (Delta: {v:+.2f} mm)" for v in width_delta],
            [f"TRAPEZOIDAL LENGTH VARIATION (Delta: {v:+.2f} mm)" for v in length_delta],
            [f"COMBINED ASYMMETRY (Diagonal Delta: {v:.2f} mm)" for v in delta],
        ],
        default="Geometry within acceptable tolerance",
    )

    d["Diag1"] = d1a
    d["Diag2"] = d2a
    d["DeltaDiag"] = delta
    d["WidthDelta"] = width_delta
    d["LengthDelta"] = length_delta
    d["AngleDevFL"] = ang_dev
    d["SquarenessStatus"] = np.where(deformed, "DEFORMED", "SQUARE OK")
    d["RootCause"] = detail
    d["CauseCategory"] = np.where(category == "", None, category)
    return d


# ---------------------------------------------------------------------------
# Rigid roto-translation (compensation)
# ---------------------------------------------------------------------------

def rigid_transform(x, y, pivot_x, pivot_y, dx, dy, theta_deg):
    """Rotate (x, y) by theta about the pivot, then translate by (dx, dy)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    xr, yr = x - pivot_x, y - pivot_y
    return xr * c - yr * s + pivot_x + dx, xr * s + yr * c + pivot_y + dy


def simulate_compensation(
    df: pd.DataFrame,
    dx: float,
    dy: float,
    yaw: float,
    spec_limit: float,
    incomplete_as_fail: bool = False,
) -> pd.DataFrame:
    """Apply a rigid compensation to every module (pivot: nominal module
    center) and re-evaluate the status. Adds <corner>_<axis>_Sim columns and
    CornersOutOfSpec_Sim, MissingCorners_Sim, IsComplete_Sim, Status_Sim."""
    out = df.copy()
    if df.empty:
        for k in DEV_COLS:
            out[f"{k}_Sim"] = pd.Series(dtype=float)
        return evaluate_status(out, spec_limit, suffix="_Sim")
    nom = _nominal_arrays(df["BatteryType"])
    centers = np.array([nominal_center(t) for t in df["BatteryType"]])
    px, py = centers[:, 0], centers[:, 1]
    for c in CORNERS:
        nx, ny = nom[f"{c}_X"], nom[f"{c}_Y"]
        ax, ay = nx + _dev(df, f"{c}_X"), ny + _dev(df, f"{c}_Y")
        cx, cy = rigid_transform(ax, ay, px, py, dx, dy, yaw)
        out[f"{c}_X_Sim"] = cx - nx
        out[f"{c}_Y_Sim"] = cy - ny
    out = evaluate_status(out, spec_limit, suffix="_Sim")
    if incomplete_as_fail:
        out.loc[out["Status_Sim"] == "INCOMPLETE", "Status_Sim"] = "FAIL"
    return out


def sim_geometry(df_sim: pd.DataFrame) -> pd.DataFrame:
    """Centroid/yaw of the compensated geometry (from the *_Sim columns)."""
    geo = df_sim[["BatteryType"]].copy()
    for k in DEV_COLS:
        geo[k] = df_sim[f"{k}_Sim"]
    geo = add_centroid_and_rotation(geo)
    return pd.DataFrame({
        "Centroid_X_Sim": geo["Centroid_X"].to_numpy(),
        "Centroid_Y_Sim": geo["Centroid_Y"].to_numpy(),
        "Rotation_Angle_Sim": geo["Rotation_Angle"].to_numpy(),
    }, index=df_sim.index)


def corner_offsets(bat_type, dx, dy, yaw) -> pd.DataFrame:
    """Per-corner displacement produced by the rigid compensation."""
    nom = NOMINALS[normalize_type(bat_type)]
    px, py = nominal_center(bat_type)
    rows = []
    for c in CORNERS:
        nx, ny = nom[c]
        cx, cy = rigid_transform(nx, ny, px, py, dx, dy, yaw)
        rows.append({
            "BatteryType": normalize_type(bat_type),
            "Corner": c,
            "Offset_X_mm": round(float(cx - nx), 3),
            "Offset_Y_mm": round(float(cy - ny), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def order_corners_convex(points: dict) -> list:
    """Corner names ordered by angle around the centroid, so the drawn
    polygon never self-intersects even with a large exaggeration factor."""
    cx = np.mean([p[0] for p in points.values()])
    cy = np.mean([p[1] for p in points.values()])
    return sorted(points, key=lambda k: np.arctan2(points[k][1] - cy, points[k][0] - cx))


def polygon(points: dict) -> tuple[list, list]:
    order = order_corners_convex(points)
    xs = [points[c][0] for c in order] + [points[order[0]][0]]
    ys = [points[c][1] for c in order] + [points[order[0]][1]]
    return xs, ys


def actual_points(row, exaggeration: float = 1.0, suffix: str = "") -> dict:
    nom = NOMINALS[normalize_type(row["BatteryType"])]
    return {
        c: (nom[c][0] + exaggeration * row[f"{c}_X{suffix}"],
            nom[c][1] + exaggeration * row[f"{c}_Y{suffix}"])
        for c in CORNERS
    }


def nominal_points(bat_type) -> dict:
    return dict(NOMINALS[normalize_type(bat_type)])
