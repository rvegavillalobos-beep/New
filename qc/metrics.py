"""Quality KPIs: first-run selection, FPY, failure Pareto, capability, findings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BATTERY_TYPES, CORNERS
from .geometry import corner_out_of_spec


def first_runs(summary: pd.DataFrame, exclude_incomplete: bool) -> pd.DataFrame:
    """The first measurement (Run 1) of every module.

    With ``exclude_incomplete`` a first run with a missing corner is counted
    as FAIL (NOK); otherwise it stays INCOMPLETE.
    """
    if summary.empty:
        return summary.copy()
    fr = (
        summary.sort_values(["RunNum", "Date"], kind="stable")
        .drop_duplicates("PartID", keep="first")
        .sort_values("Date", kind="stable")
        .copy()
    )
    if exclude_incomplete:
        fr.loc[fr["MissingCorners"] > 0, "Status"] = "FAIL"
    return fr.reset_index(drop=True)


def fpy(df: pd.DataFrame) -> float:
    return float((df["Status"] == "PASS").mean() * 100) if len(df) else float("nan")


def status_counts(df: pd.DataFrame) -> dict:
    vc = df["Status"].value_counts() if len(df) else pd.Series(dtype=int)
    return {s: int(vc.get(s, 0)) for s in ("PASS", "FAIL", "INCOMPLETE")}


def weekly_fpy(first: pd.DataFrame, ma_window: int = 3, low_sample: int = 5) -> pd.DataFrame:
    cols = ["CalendarWeek", "WeekKey", "Total", "Passed", "Failed", "Incomplete",
            "PassRate", "LowSample", "MA_FPY"]
    if first.empty:
        return pd.DataFrame(columns=cols)
    g = first.groupby(["WeekKey", "CalendarWeek"])["Status"]
    w = pd.DataFrame({
        "Total": g.size(),
        "Passed": g.apply(lambda s: int((s == "PASS").sum())),
        "Failed": g.apply(lambda s: int((s == "FAIL").sum())),
        "Incomplete": g.apply(lambda s: int((s == "INCOMPLETE").sum())),
    }).reset_index().sort_values("WeekKey")
    w["PassRate"] = w["Passed"] / w["Total"] * 100
    w["LowSample"] = w["Total"] < low_sample
    w["MA_FPY"] = w["PassRate"].rolling(window=ma_window, min_periods=1).mean()
    return w[cols].reset_index(drop=True)


def failure_pareto(first: pd.DataFrame, spec_limit: float) -> pd.DataFrame:
    """Which corner/axis puts failed modules out of spec.

    Share = % of NOK modules in which that corner/axis is out of spec (a
    module can count in several bars). Missing corners are counted as their
    own cause when incomplete first runs are treated as NOK.
    """
    nok = first[first["Status"] == "FAIL"]
    rows = []
    if nok.empty:
        return pd.DataFrame(columns=["Cause", "Corner", "Axis", "Modules", "Share"])
    oos = corner_out_of_spec(nok, spec_limit)
    for c in CORNERS:
        for a in ("X", "Y"):
            n = int(oos[f"{c}_{a}"].sum())
            if n:
                rows.append({"Cause": f"{c} {a}", "Corner": c, "Axis": a, "Modules": n})
    n_missing = int((nok["MissingCorners"] > 0).sum())
    if n_missing:
        rows.append({"Cause": "Missing corner", "Corner": "-", "Axis": "-", "Modules": n_missing})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Cause", "Corner", "Axis", "Modules", "Share"])
    out["Share"] = out["Modules"] / len(nok) * 100
    return out.sort_values("Modules", ascending=False).reset_index(drop=True)


def capability(df: pd.DataFrame, spec_limit: float) -> pd.DataFrame:
    """Per battery type, corner and axis: mean, sigma, Cp, Cpk and % out of spec.

    Cpk < Cp means the corner is off-center (systematic, compensable);
    a low Cp means spread (random, not fixable by an offset).
    """
    rows = []
    for t in BATTERY_TYPES:
        sub = df[df["BatteryType"] == t]
        if sub.empty:
            continue
        for c in CORNERS:
            for a in ("X", "Y"):
                v = sub[f"{c}_{a}"].dropna().to_numpy(float)
                if len(v) == 0:
                    continue
                mu = float(v.mean())
                sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                cp = (2 * spec_limit) / (6 * sd) if sd and sd > 0 else float("nan")
                cpk = min(spec_limit - mu, mu + spec_limit) / (3 * sd) if sd and sd > 0 else float("nan")
                rows.append({
                    "BatteryType": t, "Corner": c, "Axis": a, "N": len(v),
                    "Mean": mu, "Std": sd, "Min": float(v.min()), "Max": float(v.max()),
                    "OutPct": float((np.abs(v) > spec_limit).mean() * 100),
                    "Cp": cp, "Cpk": cpk,
                })
    return pd.DataFrame(rows)


def diagnose(mean: float, cp: float, cpk: float, spec_limit: float) -> str:
    """Offset (systematic, fixable by compensation) and/or scatter (spread,
    not fixable by an offset)."""
    offset = abs(mean) >= spec_limit / 3
    scatter = np.isfinite(cp) and cp < 1.0
    if offset and scatter:
        return "offset + scatter"
    if offset:
        return "offset"
    if scatter:
        return "scatter"
    if np.isfinite(cpk) and cpk >= 1.33:
        return "ok"
    return "marginal"


def findings(
    first: pd.DataFrame,
    weekly: pd.DataFrame,
    pareto: pd.DataFrame,
    cap: pd.DataFrame,
    sq: pd.DataFrame,
    fpy_target: float,
    spec_limit: float,
) -> list[dict]:
    """Plain-language findings, most important first.

    Each item: {"level": "good"|"warn"|"bad"|"info", "text": str}
    """
    out: list[dict] = []
    if first.empty:
        return out

    # 1) FPY against target, per type when both exist.
    types = [t for t in BATTERY_TYPES if (first["BatteryType"] == t).any()]
    for t in types:
        sub = first[first["BatteryType"] == t]
        f = fpy(sub)
        gap = f - fpy_target
        level = "good" if gap >= 0 else ("warn" if gap > -10 else "bad")
        rel = "above" if gap >= 0 else "below"
        out.append({
            "level": level,
            "text": f"**{t}** first-pass yield is **{f:.1f}%** over {len(sub)} modules, "
                    f"{abs(gap):.1f} pp {rel} the {fpy_target:.0f}% target.",
        })

    # 2) Latest week with a usable sample vs the weeks before it.
    w = weekly[weekly["Total"] >= 5] if not weekly.empty else weekly
    if len(w) >= 3:
        last = w.iloc[-1]
        prev_w = w.iloc[-4:-1] if len(w) >= 4 else w.iloc[:-1]
        prev = prev_w["PassRate"].mean()
        diff = last["PassRate"] - prev
        if abs(diff) >= 5:
            is_latest = last["WeekKey"] == weekly["WeekKey"].max()
            which = "The latest week" if is_latest else "The latest week with 5+ modules"
            out.append({
                "level": "good" if diff > 0 else "bad",
                "text": f"{which}, {last['CalendarWeek']}, ran at **{last['PassRate']:.0f}%** FPY "
                        f"({int(last['Total'])} modules), {abs(diff):.0f} pp {'above' if diff > 0 else 'below'} "
                        f"the {len(prev_w)} weeks before it ({prev:.0f}%).",
            })

    # 3) Dominant failure cause.
    if not pareto.empty:
        top = pareto.iloc[0]
        n_nok = int((first["Status"] == "FAIL").sum())
        if top["Cause"] == "Missing corner":
            text = (f"**Missing corners** are the most frequent NOK reason "
                    f"({int(top['Modules'])} of {n_nok} NOK modules).")
        else:
            text = (f"**{top['Corner']} {top['Axis']}** is out of spec in "
                    f"**{top['Share']:.0f}%** of NOK modules ({int(top['Modules'])} of {n_nok}).")
        out.append({"level": "warn", "text": text})

    # 4) Systematic offsets: mean far from zero relative to the tolerance and
    #    clearly larger than the spread effect (Cpk well below Cp).
    if not cap.empty:
        c = cap.dropna(subset=["Cp", "Cpk"]).copy()
        c = c[c["N"] >= 10]
        c["Offset"] = c["Mean"].abs()
        sys_ = c[c["Offset"] >= spec_limit / 3]
        for _, r in sys_.sort_values("Offset", ascending=False).head(2).iterrows():
            beyond = " — the average itself is outside tolerance" if abs(r["Mean"]) > spec_limit else ""
            out.append({
                "level": "bad" if beyond else "warn",
                "text": f"**{r['BatteryType']} {r['Corner']} {r['Axis']}** sits **{r['Mean']:+.2f} mm** off "
                        f"nominal on average{beyond} (σ {r['Std']:.2f} mm). A systematic offset: centering it "
                        f"would lift Cpk from {r['Cpk']:.2f} to {r['Cp']:.2f}.",
            })
        spread = c[(c["Cp"] < 1.0) & (c["Offset"] < spec_limit / 3)]
        for _, r in spread.sort_values("Cp").head(1).iterrows():
            out.append({
                "level": "warn",
                "text": f"**{r['BatteryType']} {r['Corner']} {r['Axis']}** is centered but scattered "
                        f"(σ {r['Std']:.2f} mm, Cp {r['Cp']:.2f}): an offset will not fix it, look for a "
                        f"source of variation.",
            })

    # 5) Deformation.
    if sq is not None and not sq.empty:
        deformed = sq[sq["SquarenessStatus"] == "DEFORMED"]
        pct = len(deformed) / len(sq) * 100
        if len(deformed):
            cause = deformed["CauseCategory"].value_counts()
            out.append({
                "level": "warn" if pct >= 5 else "info",
                "text": f"**{pct:.1f}%** of complete modules are deformed (diagonal delta above tolerance); "
                        f"most common pattern: **{cause.index[0]}** ({cause.iloc[0]} modules).",
            })
        else:
            out.append({"level": "good", "text": "No deformed modules in this scope."})
    return out[:8]


def kpi_summary_table(first: pd.DataFrame, exclude_incomplete: bool) -> pd.DataFrame:
    counts = status_counts(first)
    total = len(first)
    rows = [
        ("Total Unique Modules Evaluated", str(total)),
        ("Passed First Inspection (OK)", str(counts["PASS"])),
        ("Failed First Inspection (NOK)", str(counts["FAIL"])),
    ]
    if not exclude_incomplete:
        rows.append(("Incomplete Measurements", str(counts["INCOMPLETE"])))
    rows.append(("First Pass Yield (FPY)", f"{fpy(first):.1f}%" if total else "n/a"))
    return pd.DataFrame(rows, columns=["Metric", "Value"])

