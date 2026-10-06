"""Rigid compensation (X/Y translation + yaw) and its optimization.

Two ways to suggest a compensation for a battery type:

* **Median-based** (the original method): undo the median centroid offset
  and the median yaw.  It centers the *average* module.

* **FPY-optimized**: search X, Y and yaw for the setting under which the
  largest number of modules would pass.  A module passes only when *every*
  corner is within tolerance, so what matters is the worst corner of each
  module, not the average one.  When corners are biased differently (e.g. the
  rear-left corner sits lower than the others), centering the average is not
  the same as maximizing the pass rate; that is why small manual tweaks on
  top of the median (especially in yaw) often do better.

How the search works: for a fixed yaw, the X/Y settings under which one
module passes form a rectangle (each corner gives an interval in X and in Y,
and all must hold).  The best X/Y for that yaw is the point covered by the
most rectangles, found on a fine grid with 2D prefix sums.  Yaw is scanned on
a grid, then refined.  Grid edges are snapped inwards, so every module
counted really passes; the chosen setting is then re-checked exactly.

Robustness: modules that would pass only by a hair make a fragile setting.
The search primarily maximizes modules that pass with a *safety margin* on
every corner, then modules that pass at the limit, then prefers the middle of
a good region over its edge, and finally smaller moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import CORNERS, NOMINALS
from .geometry import nominal_center, normalize_type, simulate_compensation


# ---------------------------------------------------------------------------
# Median-based suggestion (original method)
# ---------------------------------------------------------------------------

def robust_bias(df_vec: pd.DataFrame, method: str = "median") -> dict:
    """Central value and spread of Centroid_X/Y and Rotation_Angle."""
    out = {}
    for col in ("Centroid_X", "Centroid_Y", "Rotation_Angle"):
        s = df_vec[col].dropna()
        if s.empty:
            out[col] = {"value": np.nan, "std": np.nan, "iqr": np.nan, "mad": np.nan, "n": 0}
            continue
        central = s.mean() if method == "mean" else s.median()
        q75, q25 = np.percentile(s, [75, 25])
        out[col] = {
            "value": float(central),
            "std": float(s.std()) if len(s) > 1 else np.nan,
            "iqr": float(q75 - q25),
            "mad": float(np.median(np.abs(s - s.median()))),
            "n": int(len(s)),
        }
    return out


def median_suggestion(df_vec: pd.DataFrame, method: str = "median") -> tuple[float, float, float]:
    b = robust_bias(df_vec, method)
    return (-b["Centroid_X"]["value"], -b["Centroid_Y"]["value"], -b["Rotation_Angle"]["value"])


# ---------------------------------------------------------------------------
# Exact evaluation
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, dx: float, dy: float, yaw: float, spec_limit: float,
             margin: float = 0.0) -> dict:
    """Pass counts after applying the compensation (exact)."""
    n = len(df)
    if n == 0 or not np.isfinite([dx, dy, yaw]).all():
        return {"n": n, "pass": 0, "pass_margin": 0, "fpy": np.nan, "fpy_margin": np.nan}
    sim = simulate_compensation(df, dx, dy, yaw, spec_limit)
    n_pass = int((sim["Status_Sim"] == "PASS").sum())
    if margin > 0:
        sim_m = simulate_compensation(df, dx, dy, yaw, spec_limit - margin)
        n_pass_m = int((sim_m["Status_Sim"] == "PASS").sum())
    else:
        n_pass_m = n_pass
    return {
        "n": n, "pass": n_pass, "pass_margin": n_pass_m,
        "fpy": n_pass / n * 100, "fpy_margin": n_pass_m / n * 100,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    dx: float
    dy: float
    yaw: float
    n: int
    n_complete: int
    pass_count: int
    pass_margin_count: int
    yaw_grid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    best_fpy_by_yaw: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    landscape: dict | None = field(repr=False, default=None)

    @property
    def fpy(self) -> float:
        return self.pass_count / self.n * 100 if self.n else np.nan

    @property
    def fpy_margin(self) -> float:
        return self.pass_margin_count / self.n * 100 if self.n else np.nan


def _relative_positions(df: pd.DataFrame):
    """Actual corner positions relative to each module's nominal center,
    plus nominal offsets from the center: arrays (n, 4)."""
    types = df["BatteryType"].map(normalize_type).to_numpy()
    n = len(df)
    rel_x = np.empty((n, 4))
    rel_y = np.empty((n, 4))
    nom_rel_x = np.empty((n, 4))
    nom_rel_y = np.empty((n, 4))
    for t in np.unique(types):
        m = types == t
        px, py = nominal_center(t)
        for j, c in enumerate(CORNERS):
            nx, ny = NOMINALS[t][c]
            nom_rel_x[m, j] = nx - px
            nom_rel_y[m, j] = ny - py
            rel_x[m, j] = nx - px + df.loc[m, f"{c}_X"].to_numpy(float)
            rel_y[m, j] = ny - py + df.loc[m, f"{c}_Y"].to_numpy(float)
    return rel_x, rel_y, nom_rel_x, nom_rel_y


def _rect_bounds(rel_x, rel_y, nom_rel_x, nom_rel_y, yaws_deg, limit):
    """For each yaw and module: the X/Y offsets under which the module passes.

    Returns lo_x, hi_x, lo_y, hi_y with shape (n_yaw, n).
    """
    t = np.radians(np.asarray(yaws_deg))[:, None, None]
    c, s = np.cos(t), np.sin(t)
    dev_x = rel_x[None] * c - rel_y[None] * s - nom_rel_x[None]
    dev_y = rel_x[None] * s + rel_y[None] * c - nom_rel_y[None]
    lo_x = -limit - dev_x.min(axis=2)
    hi_x = limit - dev_x.max(axis=2)
    lo_y = -limit - dev_y.min(axis=2)
    hi_y = limit - dev_y.max(axis=2)
    return lo_x, hi_x, lo_y, hi_y


def _depth_grid(lo_x, hi_x, lo_y, hi_y, gx0, gy0, step, nx, ny):
    """Number of rectangles covering each grid point (conservative snapping)."""
    ix0 = np.ceil((lo_x - gx0) / step - 1e-9).astype(int)
    ix1 = np.floor((hi_x - gx0) / step + 1e-9).astype(int)
    iy0 = np.ceil((lo_y - gy0) / step - 1e-9).astype(int)
    iy1 = np.floor((hi_y - gy0) / step + 1e-9).astype(int)
    ix0 = np.clip(ix0, 0, nx)
    iy0 = np.clip(iy0, 0, ny)
    ix1 = np.clip(ix1, -1, nx - 1)
    iy1 = np.clip(iy1, -1, ny - 1)
    ok = (ix0 <= ix1) & (iy0 <= iy1)
    ix0, ix1, iy0, iy1 = ix0[ok], ix1[ok], iy0[ok], iy1[ok]
    diff = np.zeros((nx + 1, ny + 1), dtype=np.int32)
    np.add.at(diff, (ix0, iy0), 1)
    np.add.at(diff, (ix1 + 1, iy0), -1)
    np.add.at(diff, (ix0, iy1 + 1), -1)
    np.add.at(diff, (ix1 + 1, iy1 + 1), 1)
    return diff.cumsum(0).cumsum(1)[:nx, :ny]


def _box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)x(2r+1) window (edges padded with zeros)."""
    if r <= 0:
        return a.astype(float)
    p = np.pad(a.astype(float), r)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def _score(depth_nom, depth_margin, n, gx, gy, smooth_r, yaw, move_weight=1e-3):
    """Lexicographic score as one number: margin count, then nominal count,
    then neighborhood (plateau) quality, then smaller move."""
    big = float(n + 1)
    smooth = _box_mean(depth_nom, smooth_r)  # <= n
    move = np.abs(gx)[:, None] + np.abs(gy)[None, :] + 10.0 * abs(yaw)
    return depth_margin * big * big + depth_nom * big + smooth - move_weight * move


def optimize(
    df: pd.DataFrame,
    spec_limit: float,
    margin: float = 0.25,
    yaw_center: float = 0.0,
    yaw_half_range: float | None = None,
    coarse_step: float = 0.05,
    fine_step: float = 0.01,
    coarse_yaw_step: float = 0.005,
    fine_yaw_step: float = 0.001,
    keep_landscape: bool = True,
    fast: bool = False,
) -> SearchResult | None:
    """Search the rigid compensation that maximizes robust FPY.

    ``fast`` skips the fine refinement (used by the backtest, where the
    search runs once per week).

    ``df`` holds first-run modules of ONE battery type. Incomplete modules
    cannot pass and are only counted in the denominator.
    """
    n = len(df)
    comp = df[df[[f"{c}_{a}" for c in CORNERS for a in "XY"]].notna().all(axis=1)]
    n_c = len(comp)
    if n_c == 0:
        return None
    rel_x, rel_y, nrx, nry = _relative_positions(comp)
    margin = max(0.0, min(margin, spec_limit * 0.9))
    lim_m = spec_limit - margin

    if yaw_half_range is None:
        yaw_half_range = max(0.4, 1.5 * abs(yaw_center) + 0.2)
    lo, hi = min(0.0, yaw_center) - yaw_half_range, max(0.0, yaw_center) + yaw_half_range
    yaws = np.arange(lo, hi + coarse_yaw_step / 2, coarse_yaw_step)

    # Coarse pass over the whole yaw range.
    bx = _rect_bounds(rel_x, rel_y, nrx, nry, yaws, spec_limit)
    bm = _rect_bounds(rel_x, rel_y, nrx, nry, yaws, lim_m)
    feasible = (bx[0] <= bx[1]) & (bx[2] <= bx[3])
    if feasible.any():
        gx_min = float(np.floor(bx[0][feasible].min() / coarse_step) * coarse_step)
        gx_max = float(np.ceil(bx[1][feasible].max() / coarse_step) * coarse_step)
        gy_min = float(np.floor(bx[2][feasible].min() / coarse_step) * coarse_step)
        gy_max = float(np.ceil(bx[3][feasible].max() / coarse_step) * coarse_step)
    else:
        gx_min, gx_max, gy_min, gy_max = -1.0, 1.0, -1.0, 1.0
    gx_min, gy_min = max(gx_min, -20.0), max(gy_min, -20.0)
    gx_max, gy_max = min(gx_max, 20.0), min(gy_max, 20.0)
    nx = int(round((gx_max - gx_min) / coarse_step)) + 1
    ny = int(round((gy_max - gy_min) / coarse_step)) + 1
    gx = gx_min + np.arange(nx) * coarse_step
    gy = gy_min + np.arange(ny) * coarse_step

    best = None
    best_by_yaw = np.zeros(len(yaws))
    smooth_r = max(1, int(round(0.1 / coarse_step)))
    for k, yw in enumerate(yaws):
        d_nom = _depth_grid(bx[0][k], bx[1][k], bx[2][k], bx[3][k], gx_min, gy_min, coarse_step, nx, ny)
        best_by_yaw[k] = d_nom.max()
        if margin > 0:
            d_m = _depth_grid(bm[0][k], bm[1][k], bm[2][k], bm[3][k], gx_min, gy_min, coarse_step, nx, ny)
        else:
            d_m = d_nom
        sc = _score(d_nom, d_m, n_c, gx, gy, smooth_r, yw)
        i, j = np.unravel_index(np.argmax(sc), sc.shape)
        if best is None or sc[i, j] > best[0]:
            best = (sc[i, j], yw, gx[i], gy[j])

    if fast:
        _, yaw_opt, dx_opt, dy_opt = best
        dx_opt, dy_opt, yaw_opt = round(float(dx_opt), 3), round(float(dy_opt), 3), round(float(yaw_opt), 4)
        ev = evaluate(df, dx_opt, dy_opt, yaw_opt, spec_limit, margin)
        return SearchResult(dx=dx_opt, dy=dy_opt, yaw=yaw_opt, n=n, n_complete=n_c,
                            pass_count=ev["pass"], pass_margin_count=ev["pass_margin"])

    # Fine pass around the coarse optimum.
    _, yw0, dx0, dy0 = best
    yaws_f = np.arange(yw0 - 2 * coarse_yaw_step, yw0 + 2 * coarse_yaw_step + fine_yaw_step / 2, fine_yaw_step)
    win = 4 * coarse_step + 0.2
    fx_min, fy_min = dx0 - win, dy0 - win
    fnx = int(round(2 * win / fine_step)) + 1
    fny = fnx
    fgx = fx_min + np.arange(fnx) * fine_step
    fgy = fy_min + np.arange(fny) * fine_step
    fbx = _rect_bounds(rel_x, rel_y, nrx, nry, yaws_f, spec_limit)
    fbm = _rect_bounds(rel_x, rel_y, nrx, nry, yaws_f, lim_m)
    smooth_rf = max(1, int(round(0.1 / fine_step)))
    fbest = None
    for k, yw in enumerate(yaws_f):
        d_nom = _depth_grid(fbx[0][k], fbx[1][k], fbx[2][k], fbx[3][k], fx_min, fy_min, fine_step, fnx, fny)
        d_m = (_depth_grid(fbm[0][k], fbm[1][k], fbm[2][k], fbm[3][k], fx_min, fy_min, fine_step, fnx, fny)
               if margin > 0 else d_nom)
        sc = _score(d_nom, d_m, n_c, fgx, fgy, smooth_rf, yw)
        i, j = np.unravel_index(np.argmax(sc), sc.shape)
        if fbest is None or sc[i, j] > fbest[0]:
            fbest = (sc[i, j], yw, fgx[i], fgy[j])

    _, yaw_opt, dx_opt, dy_opt = fbest
    dx_opt, dy_opt, yaw_opt = round(float(dx_opt), 3), round(float(dy_opt), 3), round(float(yaw_opt), 4)

    # Exact check of the chosen setting on all modules (incl. incomplete).
    ev = evaluate(df, dx_opt, dy_opt, yaw_opt, spec_limit, margin)

    landscape = None
    if keep_landscape:
        landscape = fpy_landscape(df, yaw_opt, spec_limit, center=(dx_opt, dy_opt))

    return SearchResult(
        dx=dx_opt, dy=dy_opt, yaw=yaw_opt, n=n, n_complete=n_c,
        pass_count=ev["pass"], pass_margin_count=ev["pass_margin"],
        yaw_grid=yaws, best_fpy_by_yaw=best_by_yaw / n * 100 if n else best_by_yaw,
        landscape=landscape,
    )


def fpy_landscape(df: pd.DataFrame, yaw: float, spec_limit: float,
                  center=(0.0, 0.0), half_width: float | None = None,
                  step: float = 0.05) -> dict | None:
    """FPY (%) for every X/Y setting at a fixed yaw, for the heatmap."""
    n = len(df)
    comp = df[df[[f"{c}_{a}" for c in CORNERS for a in "XY"]].notna().all(axis=1)]
    if comp.empty or n == 0:
        return None
    rel_x, rel_y, nrx, nry = _relative_positions(comp)
    b = _rect_bounds(rel_x, rel_y, nrx, nry, [yaw], spec_limit)
    if half_width is None:
        half_width = max(1.5, abs(center[0]) + 1.0, abs(center[1]) + 1.0)
    cx, cy = 0.0, 0.0  # keep "no compensation" (0, 0) in view
    x0, y0 = cx - half_width, cy - half_width
    nx = ny = int(round(2 * half_width / step)) + 1
    d = _depth_grid(b[0][0], b[1][0], b[2][0], b[3][0], x0, y0, step, nx, ny)
    return {
        "x": x0 + np.arange(nx) * step,
        "y": y0 + np.arange(ny) * step,
        "fpy": d.T / n * 100,  # rows = y, cols = x (heatmap convention)
        "yaw": yaw,
    }


def fpy_vs_yaw(df: pd.DataFrame, spec_limit: float, dx: float, dy: float,
               yaws: np.ndarray) -> np.ndarray:
    """FPY (%) when only yaw changes and X/Y stay fixed."""
    n = len(df)
    comp = df[df[[f"{c}_{a}" for c in CORNERS for a in "XY"]].notna().all(axis=1)]
    if comp.empty or n == 0:
        return np.full(len(yaws), np.nan)
    rel_x, rel_y, nrx, nry = _relative_positions(comp)
    lo_x, hi_x, lo_y, hi_y = _rect_bounds(rel_x, rel_y, nrx, nry, yaws, spec_limit)
    inside = (lo_x <= dx) & (dx <= hi_x) & (lo_y <= dy) & (dy <= hi_y)
    return inside.sum(axis=1) / n * 100


# ---------------------------------------------------------------------------
# Backtest: would it have worked week by week?
# ---------------------------------------------------------------------------

def backtest(df_vec: pd.DataFrame, spec_limit: float, margin: float, window_weeks: int,
             method: str = "median", max_test_weeks: int = 8, min_train: int = 10) -> dict | None:
    """Replay the last weeks as if the compensation had been in use.

    For each of the most recent ``max_test_weeks`` weeks: compute the median-
    based and the optimized setting from the ``window_weeks`` weeks *before*
    it, apply them to that week's modules, and count passes. Nothing a week
    is tested on was used to compute its setting, so this is the honest
    estimate of what each method would deliver on the line. In-window FPY
    is always optimistic; for the optimizer, with few modules, much more so.
    """
    weeks = sorted(df_vec["WeekKey"].dropna().unique())
    labels = df_vec.drop_duplicates("WeekKey").set_index("WeekKey")["CalendarWeek"]
    rows = []
    for wk in weeks[-max_test_weeks:]:
        prior = [w for w in weeks if w < wk][-window_weeks:]
        train = df_vec[df_vec["WeekKey"].isin(prior)]
        test = df_vec[df_vec["WeekKey"] == wk]
        if len(train) < min_train or test.empty:
            continue
        mx, my, myaw = median_suggestion(train, method)
        row = {"WeekKey": wk, "Week": labels[wk], "Modules": len(test), "Trained on": len(train),
               "none": evaluate(test, 0.0, 0.0, 0.0, spec_limit)["pass"]}
        if np.isfinite([mx, my, myaw]).all():
            row["median"] = evaluate(test, mx, my, myaw, spec_limit)["pass"]
            r = optimize(train, spec_limit, margin, yaw_center=myaw, keep_landscape=False, fast=True)
        else:
            row["median"] = row["none"]
            r = optimize(train, spec_limit, margin, keep_landscape=False, fast=True)
        row["optimized"] = evaluate(test, r.dx, r.dy, r.yaw, spec_limit)["pass"] if r else row["none"]
        rows.append(row)
    if not rows:
        return None
    t = pd.DataFrame(rows)
    n = int(t["Modules"].sum())
    out = {"weeks": t, "n": n, "n_weeks": len(t),
           "label": t["Week"].iloc[0] + (f" – {t['Week'].iloc[-1]}" if len(t) > 1 else "")}
    for k in ("none", "median", "optimized"):
        out[k] = t[k].sum() / n * 100
        t[f"{k}_fpy"] = t[k] / t["Modules"] * 100
    return out


# ---------------------------------------------------------------------------
# Readiness / stability
# ---------------------------------------------------------------------------

def weekly_bias(df_vec: pd.DataFrame) -> pd.DataFrame:
    if df_vec.empty:
        return pd.DataFrame(columns=["WeekKey", "CalendarWeek", "N", "X", "Y", "Yaw"])
    g = df_vec.groupby(["WeekKey", "CalendarWeek"])
    return pd.DataFrame({
        "N": g.size(),
        "X": g["Centroid_X"].median(),
        "Y": g["Centroid_Y"].median(),
        "Yaw": g["Rotation_Angle"].median(),
    }).reset_index().sort_values("WeekKey").reset_index(drop=True)


def readiness(df_vec: pd.DataFrame, min_n: int = 30, min_weeks: int = 4,
              min_week_n: int = 5, xy_band: float = 0.5, yaw_band: float = 0.05,
              bt: dict | None = None) -> list[dict]:
    """Checklist answering: is the process stable enough to trust a
    compensation? Each item: {"check", "ok", "detail"}."""
    items = []
    n = len(df_vec)
    items.append({
        "check": "Enough modules",
        "ok": n >= min_n,
        "detail": f"{n} first-run modules (need ≥ {min_n})",
    })
    wb = weekly_bias(df_vec)
    wb_ok = wb[wb["N"] >= min_week_n]
    items.append({
        "check": "Enough weeks",
        "ok": len(wb_ok) >= min_weeks,
        "detail": f"{len(wb_ok)} weeks with ≥ {min_week_n} modules (need ≥ {min_weeks})",
    })
    recent = wb_ok.tail(min_weeks)
    if len(recent) >= 2:
        rx = recent["X"].max() - recent["X"].min()
        ry = recent["Y"].max() - recent["Y"].min()
        ryaw = recent["Yaw"].max() - recent["Yaw"].min()
        items.append({
            "check": "Stable X/Y offset",
            "ok": bool(rx <= xy_band and ry <= xy_band),
            "detail": f"weekly medians moved {rx:.2f} mm in X and {ry:.2f} mm in Y over the last "
                      f"{len(recent)} weeks (limit {xy_band:.2f} mm)",
        })
        items.append({
            "check": "Stable yaw",
            "ok": bool(ryaw <= yaw_band),
            "detail": f"weekly median yaw moved {ryaw:.3f}° (limit {yaw_band:.3f}°)",
        })
    else:
        items.append({"check": "Stable X/Y offset", "ok": False, "detail": "not enough weeks to judge"})
        items.append({"check": "Stable yaw", "ok": False, "detail": "not enough weeks to judge"})
    if bt is not None:
        best = max(bt["median"], bt["optimized"])
        items.append({
            "check": "Helps in the backtest",
            "ok": bool(best - bt["none"] >= 100 / max(bt["n"], 1) - 1e-9),
            "detail": f"applied week by week over {bt['label']} ({bt['n']} modules): "
                      f"{bt['none']:.1f}% → median {bt['median']:.1f}% · optimized {bt['optimized']:.1f}%",
        })
    else:
        items.append({"check": "Helps in the backtest", "ok": False,
                      "detail": "needs earlier weeks with enough modules to test on"})
    return items
