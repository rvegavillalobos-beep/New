"""Plotly figures. Pure functions: data in, go.Figure out (no Streamlit).

Conventions: thin marks, recessive grid, no dual axes, identity colors follow
the entity (battery type, axis, compensation option), status colors only for
status, legends whenever there is more than one series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import (
    MA_COLOR, NOK_CAUSE_COLORS, NOK_CAUSE_ORDER,
    AXIS_COLORS, BATTERY_TYPES, CAUSE_COLORS, CAUSE_ORDER, CORNER_LABELS, CORNERS,
    HIGHLIGHT_COLOR, INK, LIMIT_COLOR, NOMINAL_COLOR, OPTION_COLORS, STATUS_COLORS,
    TARGET_COLOR, TYPE_COLORS,
)
from .geometry import actual_points, nominal_points, polygon

FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12)
SEQ_BLUE = [
    [0.0, "#f4f8fd"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
    [0.6, "#3987e5"], [0.75, "#256abf"], [0.9, "#184f95"], [1.0, "#0d366b"],
]


def _base(fig: go.Figure, height: int = 360, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=40 if title else 16, b=8),
        font=FONT,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right" if title else "left",
                    x=1 if title else 0, title_text="", traceorder="normal"),
        hoverlabel=dict(font=FONT),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14)) if title else None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=INK["axis"], ticks="outside", tickcolor=INK["axis"])
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.22)", zeroline=False, linecolor=INK["axis"])
    return fig


def _limit_lines(fig, limit, row=None, col=None, axis="y", label=True):
    for v in (limit, -limit):
        kw = dict(line_color=LIMIT_COLOR, line_width=1, line_dash="dot")
        if label:
            kw.update(annotation_text=f"{v:+.1f} mm", annotation_position="top right" if v > 0 else "bottom right",
                      annotation_font=dict(size=10, color=INK["muted"]))
        if row is not None:
            kw.update(row=row, col=col)
        if axis == "y":
            fig.add_hline(y=v, **kw)
        else:
            fig.add_vline(x=v, **kw)


def _week_axis(fig: go.Figure, labels, **kw) -> None:
    """Category week axis with short ticks: 'CW23', plus the year on the
    first tick and wherever the year changes."""
    labels = [str(x) for x in labels]
    text, prev = [], None
    for lab in labels:
        year, _, wk = lab.partition("-")
        text.append(f"{wk}<br>{year}" if year != prev else wk)
        prev = year
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels,
                     tickmode="array", tickvals=labels, ticktext=text, tickangle=0, **kw)


def empty(message: str = "No data for this selection", height: int = 220) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=INK["muted"], size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height, legend=False)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def weekly_fpy(weekly: pd.DataFrame, target: float, ma_window: int, show_incomplete: bool) -> go.Figure:
    """FPY per week (top) and tested volume per week (bottom). Two stacked
    panels with their own y axes instead of one dual-axis chart."""
    if weekly.empty:
        return empty()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    x = weekly["CalendarWeek"]
    low = weekly["LowSample"].to_numpy()
    labels = [f"{r:.0f}%*" if lo else f"{r:.0f}%" for r, lo in zip(weekly["PassRate"], low)]
    fig.add_trace(go.Scatter(
        x=x, y=weekly["PassRate"], mode="lines+markers+text", name="Weekly FPY",
        line=dict(color="#2a78d6", width=2),
        marker=dict(size=8, color=np.where(low, "rgba(0,0,0,0)", "#2a78d6"),
                    line=dict(width=2, color="#2a78d6")),
        text=labels, textposition="top center", textfont=dict(size=11, color=INK["primary"]),
        cliponaxis=False,
        customdata=np.stack([weekly["Total"], np.where(low, "· low sample (N<5)", "")], axis=1),
        hovertemplate="<b>%{x}</b><br>FPY %{y:.1f}%<br>%{customdata[0]} modules %{customdata[1]}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=weekly["MA_FPY"], mode="lines", name=f"MA{ma_window} ({ma_window}-week moving average)",
        line=dict(color=MA_COLOR, width=2.5, dash="dash"),
        hovertemplate=f"<b>%{{x}}</b><br>MA{ma_window}: %{{y:.1f}}%<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=target, line_color=TARGET_COLOR, line_width=1.5, line_dash="dash",
                  annotation_text=f"Target {target:.0f}%", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK["secondary"]), row=1, col=1)
    statuses = [("Passed", "PASS"), ("Failed", "FAIL")]
    if show_incomplete and weekly["Incomplete"].sum() > 0:
        statuses.append(("Incomplete", "INCOMPLETE"))
    for colname, status in statuses:
        fig.add_trace(go.Bar(
            x=x, y=weekly[colname], name=f"{colname} (volume)",
            marker=dict(color=STATUS_COLORS[status], line=dict(width=0)), opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>{colname}: %{{y}}<extra></extra>",
        ), row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(title_text="FPY (%)", range=[0, 112], row=1, col=1)
    fig.update_yaxes(title_text="Modules", row=2, col=1)
    _base(fig, 450)
    _week_axis(fig, x)
    return fig


def status_by_type(first: pd.DataFrame, statuses=("PASS", "FAIL", "INCOMPLETE")) -> go.Figure:
    """100% horizontal bars of first-run status per battery type, plus an
    'All types' bar when there are several. Row labels carry the totals so
    nobody has to add up the segments."""
    if first.empty:
        return empty()
    types = [t for t in BATTERY_TYPES if (first["BatteryType"] == t).any()]
    counts = (first.groupby(["BatteryType", "Status"]).size().unstack(fill_value=0)
              .reindex(types).fillna(0))
    if len(types) > 1:
        counts.loc["All types"] = counts.sum()
    rows = list(counts.index)
    tot = counts.sum(axis=1)
    ylab = [f"<b>{r}</b><br>{int(n)} modules" for r, n in zip(rows, tot)]
    fig = go.Figure()
    for s_ in statuses:
        if s_ not in counts.columns or counts[s_].sum() == 0:
            continue
        c = counts[s_]
        pct = c / tot * 100
        fig.add_trace(go.Bar(
            y=ylab, x=pct, orientation="h", name=s_.title(),
            marker=dict(color=STATUS_COLORS[s_], line=dict(width=2, color="rgba(255,255,255,0.9)")),
            text=[f"{p:.0f}% ({int(n)})" if p >= 8 else "" for p, n in zip(pct, c)],
            textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=12),
            customdata=np.stack([c, tot], axis=1),
            hovertemplate="%{y}<br>" + s_.title()
                          + ": %{x:.1f}% (%{customdata[0]:.0f} of %{customdata[1]:.0f})<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.4)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="rgba(137,135,129,0.22)")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _base(fig, 100 + 62 * len(rows))


def failure_pareto(pareto: pd.DataFrame, n_nok: int) -> go.Figure:
    if pareto.empty:
        return empty("No NOK modules in this selection")
    p = pareto.head(9).iloc[::-1]
    colors = [AXIS_COLORS.get(a, INK["muted"]) for a in p["Axis"]]
    fig = go.Figure(go.Bar(
        y=p["Cause"], x=p["Share"], orientation="h", showlegend=False,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.0f}%  ({int(m)})" for s, m in zip(p["Share"], p["Modules"])],
        textposition="outside", cliponaxis=False, textfont=dict(color=INK["secondary"]),
        hovertemplate="<b>%{y}</b><br>out of spec in %{x:.1f}% of NOK modules<extra></extra>",
    ))
    # Legend proxies for the axis colors.
    for a in ("X", "Y"):
        if a in set(p["Axis"]):
            fig.add_trace(go.Bar(y=[None], x=[None], name=f"{a} axis", marker_color=AXIS_COLORS[a],
                                 orientation="h", showlegend=True, hoverinfo="skip"))
    fig.update_xaxes(range=[0, min(100, p["Share"].max() * 1.25 + 5)], ticksuffix="%",
                     showgrid=True, gridcolor="rgba(137,135,129,0.22)",
                     title_text=f"% of NOK modules (N = {n_nok})")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(bargap=0.35)
    return _base(fig, 80 + 34 * len(p))


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------

def corner_trend(df_type: pd.DataFrame, corner: str, limit: float) -> go.Figure:
    """X and Y deviation of one corner, module by module in time order."""
    d = df_type[["Date", "PartID", "CalendarWeek", f"{corner}_X", f"{corner}_Y"]].dropna()
    if d.empty:
        return empty(f"No {CORNER_LABELS[corner]} data", 260)
    d = d.sort_values("Date").reset_index(drop=True)
    seq = np.arange(1, len(d) + 1)
    custom = np.stack([d["PartID"], d["Date"].dt.strftime("%Y-%m-%d %H:%M"), d["CalendarWeek"]], axis=1)
    fig = go.Figure()
    for a in ("X", "Y"):
        fig.add_trace(go.Scatter(
            x=seq, y=d[f"{corner}_{a}"], mode="lines+markers", name=a,
            line=dict(color=AXIS_COLORS[a], width=1.5), marker=dict(size=5),
            customdata=custom,
            hovertemplate=f"<b>#%{{x}}</b> {a}: %{{y:+.2f}} mm<br>%{{customdata[0]}}<br>%{{customdata[1]}} · %{{customdata[2]}}<extra></extra>",
        ))
    _limit_lines(fig, limit, label=False)
    ymax = max(limit + 1.0, float(np.nanmax(np.abs(d[[f"{corner}_X", f"{corner}_Y"]].to_numpy()))) + 0.5)
    fig.update_yaxes(range=[-ymax, ymax], title_text="mm", zeroline=True, zerolinecolor=INK["axis"], zerolinewidth=1)
    fig.update_xaxes(title_text="Module # (time order)")
    return _base(fig, 250, title=f"{CORNER_LABELS[corner]} ({corner})")


def corner_clouds(df_type: pd.DataFrame, limit: float, color: str) -> go.Figure:
    """X vs Y deviation of every module at each corner, inside the
    tolerance box. A cloud off-center = systematic offset; a wide cloud =
    variation. Panels are laid out like the module seen from above
    (front at the top)."""
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.14,
                        subplot_titles=[CORNER_LABELS[c] for c in ("FL", "FR", "RL", "RR")])
    pos = {"FL": (1, 1), "FR": (1, 2), "RL": (2, 1), "RR": (2, 2)}
    m = limit + 1.5
    for c in CORNERS:
        r, k = pos[c]
        d = df_type[[f"{c}_X", f"{c}_Y", "PartID", "Status"]].dropna(subset=[f"{c}_X", f"{c}_Y"])
        m = max(m, float(np.nanmax(np.abs(d[[f"{c}_X", f"{c}_Y"]].to_numpy()), initial=0)) + 0.5)
        fig.add_shape(type="rect", x0=-limit, x1=limit, y0=-limit, y1=limit,
                      line=dict(color=LIMIT_COLOR, width=1, dash="dot"), row=r, col=k)
        out = (d[f"{c}_X"].abs() > limit) | (d[f"{c}_Y"].abs() > limit)
        fig.add_trace(go.Scatter(
            x=d[f"{c}_X"], y=d[f"{c}_Y"], mode="markers", showlegend=False,
            marker=dict(size=7, color=np.where(out, LIMIT_COLOR, color), opacity=0.55,
                        line=dict(width=1, color="rgba(255,255,255,0.8)")),
            customdata=d[["PartID"]],
            hovertemplate="%{customdata[0]}<br>X %{x:+.2f} · Y %{y:+.2f} mm<extra></extra>",
        ), row=r, col=k)
        if len(d):
            mx, my = d[f"{c}_X"].mean(), d[f"{c}_Y"].mean()
            fig.add_trace(go.Scatter(
                x=[mx], y=[my], mode="markers", showlegend=False,
                marker=dict(symbol="x-thin", size=16, line=dict(width=3, color=INK["primary"])),
                hovertemplate=f"<b>Mean</b><br>X {mx:+.2f} · Y {my:+.2f} mm<extra></extra>",
            ), row=r, col=k)
    for r in (1, 2):
        for k in (1, 2):
            fig.update_xaxes(range=[-m, m], zeroline=True, zerolinecolor=INK["axis"], title_text="X dev (mm)" if r == 2 else None,
                             showgrid=True, gridcolor="rgba(137,135,129,0.15)", constrain="domain", row=r, col=k)
            fig.update_yaxes(range=[-m, m], zeroline=True, zerolinecolor=INK["axis"], title_text="Y dev (mm)" if k == 1 else None,
                             scaleanchor=f"x{'' if (r, k) == (1, 1) else (r - 1) * 2 + k}", scaleratio=1, row=r, col=k)
    return _base(fig, 640, legend=False)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def short_id(part_id) -> str:
    """Compact module label for legends: batch + serial when the ID has the
    usual ...B156N00001... pattern, else the last 12 characters."""
    import re
    m = re.search(r"(B\d{3}N\d{5})", str(part_id))
    return m.group(1) if m else str(part_id)[-12:]


# Screen position of each corner (front = right, left side = top because the
# Y axis is reversed); labels are anchored so they extend into the module.
CORNER_ANCHOR = {"FL": ("right", "top"), "FR": ("right", "bottom"),
                 "RL": ("left", "top"), "RR": ("left", "bottom")}


def geometry_plot(df_plot: pd.DataFrame, types_in_scope, limit: float, exaggeration: float,
                  focus_key: str | None = None) -> go.Figure:
    """Modules drawn against nominal. Every module is its own legend entry,
    grouped by status: click an entry to hide it, double-click to show only
    that one. The focused module carries its corner deviations as labels."""
    fig = go.Figure()
    for t in types_in_scope:
        xs, ys = polygon(nominal_points(t))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"Nominal {t}", legendgroup="ref",
            legendgrouptitle_text="Reference",
            line=dict(color=NOMINAL_COLOR, width=2, dash="dash"), hoverinfo="skip",
        ))
    eff = limit * exaggeration
    bx, by = [], []
    for t in types_in_scope:
        for cx, cy in nominal_points(t).values():
            bx += [cx - eff, cx + eff, cx + eff, cx - eff, cx - eff, None]
            by += [cy - eff, cy - eff, cy + eff, cy + eff, cy - eff, None]
    if bx:
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines", name=f"Tolerance ±{limit:g} mm", legendgroup="ref",
            line=dict(color="rgba(208,59,59,0.6)", width=1, dash="dot"), hoverinfo="skip",
        ))
    names = {"FAIL": "Fail", "PASS": "Pass", "INCOMPLETE": "Incomplete"}
    focus_row = None
    for status in ("FAIL", "INCOMPLETE", "PASS"):
        first_in_group = True
        for _, row in df_plot[df_plot["Status"] == status].iterrows():
            if row.get("_mod_key") == focus_key:
                focus_row = row
                continue
            pts = actual_points(row, exaggeration)
            if any(np.isnan(v) for p in pts.values() for v in p):
                continue
            xs, ys = polygon(pts)
            is_bad = status != "PASS"
            kw = {"legendgrouptitle_text": names[status]} if first_in_group else {}
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", legendgroup=status,
                name=f"{short_id(row['PartID'])} · R{row['RunNum']}",
                line=dict(color=STATUS_COLORS[status] if is_bad else "rgba(82,81,78,0.6)",
                          width=1.6 if is_bad else 1.2),
                opacity=0.85 if is_bad else 0.7,
                hovertemplate=f"<b>{row['PartID']}</b><br>Run {row['RunNum']} · {status}"
                              f"<br>{row['Date']:%Y-%m-%d %H:%M}<extra></extra>",
                **kw,
            ))
            first_in_group = False
    if focus_row is not None:
        pts = actual_points(focus_row, exaggeration)
        if not any(np.isnan(v) for p in pts.values() for v in p):
            xs, ys = polygon(pts)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", legendgroup="focus", legendgrouptitle_text="Focused",
                name=f"{short_id(focus_row['PartID'])} · R{focus_row['RunNum']}",
                line=dict(color=HIGHLIGHT_COLOR, width=3.5),
                hovertemplate=f"<b>{focus_row['PartID']}</b><br>Run {focus_row['RunNum']} · "
                              f"{focus_row['Status']}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[pts[c][0] for c in CORNERS], y=[pts[c][1] for c in CORNERS],
                mode="markers", legendgroup="focus", showlegend=False,
                marker=dict(size=10, color=HIGHLIGHT_COLOR, line=dict(width=2, color="white")),
                hoverinfo="skip",
            ))
            # Labels sit inside the outline, next to their corner, on a light
            # background so they stay readable over the module lines.
            for c in CORNERS:
                vx, vy = focus_row[f"{c}_X"], focus_row[f"{c}_Y"]
                out = abs(vx) > limit or abs(vy) > limit
                xa, ya = CORNER_ANCHOR[c]
                fig.add_annotation(
                    x=pts[c][0], y=pts[c][1], text=f"<b>{c}</b>  X {vx:+.2f} · Y {vy:+.2f}",
                    showarrow=False, xanchor=xa, yanchor=ya, xshift=-10 if xa == "right" else 10,
                    yshift=-10 if ya == "top" else 10,
                    font=dict(size=12, color=LIMIT_COLOR if out else INK["primary"]),
                    bgcolor="rgba(255,255,255,0.88)", bordercolor=LIMIT_COLOR if out else HIGHLIGHT_COLOR,
                    borderwidth=1, borderpad=3,
                )
    fig.update_xaxes(title_text="Global X (mm)", showgrid=True, gridcolor="rgba(137,135,129,0.15)")
    fig.update_yaxes(title_text="Global Y (mm)", scaleanchor="x", scaleratio=1, autorange="reversed")
    _base(fig, 700)
    fig.update_layout(legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
                                  font=dict(size=11), groupclick="toggleitem", tracegroupgap=10,
                                  traceorder="grouped", itemclick="toggle", itemdoubleclick="toggleothers"))
    return fig


# ---------------------------------------------------------------------------
# Squareness
# ---------------------------------------------------------------------------

def stacked_share(counts: pd.DataFrame, order, colors, height=None) -> go.Figure:
    """100% horizontal bars, one per battery type, for any categorical split."""
    if counts.empty:
        return empty()
    types = [t for t in BATTERY_TYPES if t in counts.index]
    counts = counts.reindex(types).fillna(0)
    tot = counts.sum(axis=1)
    fig = go.Figure()
    for cat in order:
        if cat not in counts.columns or counts[cat].sum() == 0:
            continue
        pct = counts[cat] / tot * 100
        fig.add_trace(go.Bar(
            y=types, x=pct, orientation="h", name=cat,
            marker=dict(color=colors[cat], line=dict(width=2, color="rgba(255,255,255,0.9)")),
            text=[f"{p:.0f}% ({int(n)})" if p >= 10 else "" for p, n in zip(pct, counts[cat])],
            textposition="inside", insidetextanchor="middle", textfont=dict(color="white"),
            customdata=np.stack([counts[cat], tot], axis=1),
            hovertemplate="<b>%{y}</b><br>" + cat + ": %{x:.1f}% (%{customdata[0]} of %{customdata[1]})<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="rgba(137,135,129,0.22)")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _base(fig, height or (90 + 60 * len(types)))


def deformation_trend(weekly: pd.DataFrame) -> go.Figure:
    """Deformed share per week (top, per battery type, plus a linear trend
    over all modules) and deformed / square counts per week (bottom)."""
    if weekly.empty:
        return empty()
    order = (weekly[["WeekKey", "CalendarWeek"]].drop_duplicates()
             .sort_values("WeekKey")["CalendarWeek"].tolist())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.07)
    for t in BATTERY_TYPES:
        w = weekly[weekly["BatteryType"] == t]
        if w.empty:
            continue
        fig.add_trace(go.Scatter(
            x=w["CalendarWeek"], y=w["Share"], mode="lines+markers+text", name=f"{t} deformed %",
            line=dict(color=TYPE_COLORS[t], width=2), marker=dict(size=8),
            text=[f"{v:.0f}%" for v in w["Share"]], textposition="top center",
            textfont=dict(size=10, color=INK["secondary"]), cliponaxis=False,
            customdata=np.stack([w["Deformed"], w["N"], w["MedianDiag"]], axis=1),
            hovertemplate=f"<b>{t}</b> %{{x}}<br>%{{y:.0f}}% deformed (%{{customdata[0]}} of %{{customdata[1]}})"
                          "<br>median diagonal Δ %{customdata[2]:.2f} mm<extra></extra>",
        ), row=1, col=1)
    tot = (weekly.groupby(["WeekKey", "CalendarWeek"])[["Deformed", "N"]].sum()
           .reset_index().sort_values("WeekKey"))
    tot["Share"] = tot["Deformed"] / tot["N"] * 100
    if len(tot) >= 3:
        xi = np.arange(len(tot))
        slope, icpt = np.polyfit(xi, tot["Share"], 1, w=np.sqrt(tot["N"]))
        fig.add_trace(go.Scatter(
            x=tot["CalendarWeek"], y=np.clip(icpt + slope * xi, 0, 100), mode="lines",
            name=f"Trend, all types ({slope:+.1f} pp per week)",
            line=dict(color=INK["secondary"], width=2, dash="dot"), hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=tot["CalendarWeek"], y=tot["Deformed"], name="Deformed (count)",
        marker=dict(color=STATUS_COLORS["DEFORMED"], line=dict(width=0)), opacity=0.85,
        hovertemplate="%{x}<br>%{y} deformed<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(
        x=tot["CalendarWeek"], y=tot["N"] - tot["Deformed"], name="Square (count)",
        marker=dict(color="rgba(137,135,129,0.55)", line=dict(width=0)),
        hovertemplate="%{x}<br>%{y} square<extra></extra>"), row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(title_text="Deformed (%)", range=[0, 115], row=1, col=1)
    fig.update_yaxes(title_text="Modules", row=2, col=1)
    _base(fig, 460)
    _week_axis(fig, order)
    return fig


def fail_rate_by_band(by_band: pd.DataFrame, overall: float, diag_tol: float) -> go.Figure:
    """NOK rate for each band of diagonal delta: does more deformation mean
    more NOK?"""
    if by_band.empty:
        return empty()
    grey = "rgba(137,135,129,0.75)"
    colors = [STATUS_COLORS["DEFORMED"] if a else grey for a in by_band["AboveTol"]]
    fig = go.Figure(go.Bar(
        x=by_band["Label"], y=by_band["FailRate"], marker=dict(color=colors, line=dict(width=0)),
        text=[f"{r:.0f}%  (N={n})" for r, n in zip(by_band["FailRate"], by_band["N"])],
        textposition="outside", cliponaxis=False, showlegend=False,
        customdata=by_band[["N"]],
        hovertemplate="Diagonal Δ %{x} mm<br>NOK rate %{y:.0f}% of %{customdata[0]} modules<extra></extra>",
    ))
    fig.add_hline(y=overall, line_color=INK["secondary"], line_width=1.5, line_dash="dot",
                  annotation_text=f"All modules {overall:.0f}%", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK["secondary"]))
    for name, col in (("Square (below tolerance)", grey), (f"Deformed (≥ {diag_tol:g} mm)", STATUS_COLORS["DEFORMED"])):
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, marker_color=col, hoverinfo="skip"))
    fig.update_xaxes(title_text="Diagonal delta (mm)", type="category")
    fig.update_yaxes(title_text="NOK rate (%)", range=[0, 118])
    fig.update_layout(bargap=0.35)
    return _base(fig, 380)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def drift_path(df_vec: pd.DataFrame, split: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Nominal (0, 0)",
                             marker=dict(symbol="cross-thin", size=18, line=dict(width=2, color=TARGET_COLOR)),
                             hoverinfo="skip"))
    groups = [(t, df_vec[df_vec["BatteryType"] == t], TYPE_COLORS[t]) for t in BATTERY_TYPES
              if (df_vec["BatteryType"] == t).any()] if split else [("All", df_vec, "#2a78d6")]
    for name, d, color in groups:
        d = d.dropna(subset=["Centroid_X", "Centroid_Y"])
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["Centroid_X"], y=d["Centroid_Y"], mode="markers", name=f"Modules ({name})",
            marker=dict(size=6, color=color, opacity=0.25), legendgroup=name,
            customdata=np.stack([d["PartID"], d["CalendarWeek"]], axis=1),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>X %{x:+.2f} · Y %{y:+.2f} mm<extra></extra>",
        ))
        w = (d.groupby(["WeekKey", "CalendarWeek"])
             .agg(X=("Centroid_X", "mean"), Y=("Centroid_Y", "mean"), N=("PartID", "count"))
             .reset_index().sort_values("WeekKey"))
        fig.add_trace(go.Scatter(
            x=w["X"], y=w["Y"], mode="lines+markers+text", name=f"Weekly mean ({name})",
            line=dict(color=color, width=2), legendgroup=name,
            marker=dict(size=10, color=color, line=dict(width=2, color="white")),
            text=[cw.split("-")[-1] if i in (0, len(w) - 1) else "" for i, cw in enumerate(w["CalendarWeek"])],
            textposition="top right", textfont=dict(size=11, color=INK["secondary"]),
            customdata=np.stack([w["CalendarWeek"], w["N"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b> (N=%{customdata[1]})<br>mean X %{x:+.2f} · Y %{y:+.2f} mm<extra></extra>",
        ))
    fig.update_xaxes(title_text="Centroid X deviation (mm)", zeroline=True, zerolinecolor=INK["axis"],
                     showgrid=True, gridcolor="rgba(137,135,129,0.15)", constrain="domain")
    fig.update_yaxes(title_text="Centroid Y deviation (mm)", zeroline=True, zerolinecolor=INK["axis"],
                     scaleanchor="x", scaleratio=1, autorange="reversed")
    return _base(fig, 500)


def weekly_bias_chart(wb_by_type: dict) -> go.Figure:
    """Weekly median X, Y (mm) and yaw (°) per battery type, three panels."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=["Median centroid X (mm)", "Median centroid Y (mm)", "Median yaw (°)"])
    weeks = sorted({(k, w) for wb in wb_by_type.values() for k, w in zip(wb["WeekKey"], wb["CalendarWeek"])})
    order = [w for _, w in weeks]
    for t, wb in wb_by_type.items():
        if wb.empty:
            continue
        for r, col in enumerate(["X", "Y", "Yaw"], start=1):
            fig.add_trace(go.Scatter(
                x=wb["CalendarWeek"], y=wb[col], mode="lines+markers", name=t, legendgroup=t,
                showlegend=r == 1, line=dict(color=TYPE_COLORS[t], width=2), marker=dict(size=8),
                customdata=wb[["N"]],
                hovertemplate=f"<b>{t}</b> %{{x}} (N=%{{customdata[0]}})<br>{col}: %{{y:+.3f}}<extra></extra>",
            ), row=r, col=1)
            fig.add_hline(y=0, line_color=INK["axis"], line_width=1, row=r, col=1)
    _week_axis(fig, order)
    fig.update_annotations(font=dict(size=12, color=INK["secondary"]), x=0, xanchor="left")
    return _base(fig, 560)


def yaw_by_module(df_vec: pd.DataFrame) -> go.Figure:
    d = df_vec.dropna(subset=["Rotation_Angle"]).sort_values("Date")
    if d.empty:
        return empty()
    fig = go.Figure()
    for t in BATTERY_TYPES:
        s = d[d["BatteryType"] == t]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["Date"], y=s["Rotation_Angle"], mode="markers", name=t,
            marker=dict(size=7, color=TYPE_COLORS[t], opacity=0.7, line=dict(width=1, color="white")),
            customdata=np.stack([s["PartID"], s["Status"]], axis=1),
            hovertemplate="%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M}<br>Yaw %{y:+.3f}° · %{customdata[1]}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=INK["axis"], line_width=1)
    fig.update_yaxes(title_text="Yaw (°)")
    return _base(fig, 300)


# ---------------------------------------------------------------------------
# Compensation
# ---------------------------------------------------------------------------

def yaw_profile(yaws, best_fpy, fixed_fpy, points: dict, color: str) -> go.Figure:
    """Best reachable FPY for each yaw (X/Y re-optimized), plus FPY when only
    yaw changes around the chosen X/Y. Vertical markers show the options."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yaws, y=best_fpy, mode="lines", name="Best FPY at each yaw (X/Y re-optimized)",
        line=dict(color=color, width=2), hovertemplate="yaw %{x:+.3f}°<br>best FPY %{y:.1f}%<extra></extra>",
    ))
    if fixed_fpy is not None:
        fig.add_trace(go.Scatter(
            x=yaws, y=fixed_fpy, mode="lines", name="FPY changing only yaw (optimized X/Y kept)",
            line=dict(color=INK["secondary"], width=1.5, dash="dot"),
            hovertemplate="yaw %{x:+.3f}°<br>FPY %{y:.1f}%<extra></extra>",
        ))
    for name, (yaw, fpy) in points.items():
        if yaw is None or not np.isfinite(yaw):
            continue
        fig.add_trace(go.Scatter(
            x=[yaw], y=[fpy], mode="markers", name=name,
            marker=dict(size=12, color=OPTION_COLORS.get(name, INK["primary"]),
                        symbol="diamond" if name == "Optimized" else "circle",
                        line=dict(width=2, color="white")),
            hovertemplate=f"<b>{name}</b><br>yaw %{{x:+.3f}}°<br>FPY %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_xaxes(title_text="Yaw compensation (°)", showgrid=True, gridcolor="rgba(137,135,129,0.15)",
                     zeroline=True, zerolinecolor=INK["axis"])
    fig.update_yaxes(title_text="FPY (%)", rangemode="tozero")
    return _base(fig, 360)


def fpy_landscape(land: dict, points: dict) -> go.Figure:
    """Heatmap: FPY for every X/Y setting at a fixed yaw."""
    if not land:
        return empty()
    fig = go.Figure(go.Heatmap(
        x=land["x"], y=land["y"], z=land["fpy"], colorscale=SEQ_BLUE, zmin=0,
        colorbar=dict(title=dict(text="FPY %", side="right"), thickness=12, len=0.85),
        hovertemplate="X %{x:+.2f} mm · Y %{y:+.2f} mm<br>FPY %{z:.1f}%<extra></extra>",
    ))
    for name, (x, y) in points.items():
        if x is None or not np.isfinite([x, y]).all():
            continue
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name=name,
            marker=dict(size=13, color=OPTION_COLORS.get(name, INK["primary"]),
                        symbol="diamond" if name == "Optimized" else "circle",
                        line=dict(width=2, color="white")),
            hovertemplate=f"<b>{name}</b><br>X %{{x:+.2f}} · Y %{{y:+.2f}} mm<extra></extra>",
        ))
    fig.update_xaxes(title_text="X compensation (mm)", showgrid=False, constrain="domain")
    fig.update_yaxes(title_text="Y compensation (mm)", showgrid=False, scaleanchor="x", scaleratio=1)
    return _base(fig, 430)


def backtest_chart(weeks: pd.DataFrame) -> go.Figure:
    """FPY per week when each week gets the setting computed from the weeks
    before it."""
    fig = go.Figure()
    for key, name in (("none_fpy", "No compensation"), ("median_fpy", "Median-based"),
                      ("optimized_fpy", "Optimized")):
        fig.add_trace(go.Scatter(
            x=weeks["Week"], y=weeks[key], mode="lines+markers", name=name,
            line=dict(color=OPTION_COLORS[name], width=2,
                      dash="dot" if name == "No compensation" else "solid"),
            marker=dict(size=8, symbol="diamond" if name == "Optimized" else "circle"),
            customdata=weeks[["Modules"]],
            hovertemplate=f"<b>{name}</b> %{{x}}<br>FPY %{{y:.0f}}% (N=%{{customdata[0]}})<extra></extra>",
        ))
    _week_axis(fig, weeks["Week"])
    fig.update_yaxes(title_text="FPY (%)", range=[0, 105])
    return _base(fig, 340)


def centroid_before_after(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Nominal",
                             marker=dict(symbol="cross-thin", size=18, line=dict(width=2, color=TARGET_COLOR)),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["Centroid_X"], y=df["Centroid_Y"], mode="markers", name="Measured",
                             marker=dict(size=7, color=OPTION_COLORS["No compensation"], opacity=0.6),
                             hovertemplate="X %{x:+.2f} · Y %{y:+.2f} mm<extra>measured</extra>"))
    fig.add_trace(go.Scatter(x=df["Centroid_X_Sim"], y=df["Centroid_Y_Sim"], mode="markers", name="Compensated",
                             marker=dict(size=7, color=OPTION_COLORS["Optimized"], opacity=0.6),
                             hovertemplate="X %{x:+.2f} · Y %{y:+.2f} mm<extra>compensated</extra>"))
    fig.update_xaxes(title_text="Centroid X (mm)", zeroline=True, zerolinecolor=INK["axis"],
                     showgrid=True, gridcolor="rgba(137,135,129,0.15)", constrain="domain")
    fig.update_yaxes(title_text="Centroid Y (mm)", zeroline=True, zerolinecolor=INK["axis"],
                     scaleanchor="x", scaleratio=1, autorange="reversed")
    return _base(fig, 380)


def weekly_fpy_compare(wk: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk["CalendarWeek"], y=wk["Real"], mode="lines+markers", name="Measured",
                             line=dict(color=OPTION_COLORS["No compensation"], width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=wk["CalendarWeek"], y=wk["Sim"], mode="lines+markers", name="With compensation",
                             line=dict(color=OPTION_COLORS["Optimized"], width=2), marker=dict(size=8),
                             customdata=wk[["N"]],
                             hovertemplate="%{x}<br>FPY %{y:.1f}% (N=%{customdata[0]})<extra></extra>"))
    _week_axis(fig, wk["CalendarWeek"])
    fig.update_yaxes(title_text="FPY (%)", range=[0, 105])
    return _base(fig, 380)


def overlay(bat_type: str, df: pd.DataFrame, exaggeration: float = 1.0, limit: float | None = None,
            height: int = 540) -> go.Figure:
    """Nominal vs average measured vs average compensated outline. The
    deviations are magnified so the difference is visible."""
    fig = go.Figure()
    nom = nominal_points(bat_type)
    xs, ys = polygon(nom)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Nominal",
                             line=dict(color=NOMINAL_COLOR, width=2, dash="dash"), hoverinfo="skip"))
    if limit:
        eff = limit * exaggeration
        bx, by = [], []
        for cx, cy in nom.values():
            bx += [cx - eff, cx + eff, cx + eff, cx - eff, cx - eff, None]
            by += [cy - eff, cy - eff, cy + eff, cy + eff, cy - eff, None]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", name=f"Tolerance ±{limit:g} mm",
                                 line=dict(color="rgba(208,59,59,0.6)", width=1, dash="dot"),
                                 hoverinfo="skip"))
    if not df.empty:
        for name, suffix, color in (("Measured (avg)", "", OPTION_COLORS["No compensation"]),
                                    ("Compensated (avg)", "_Sim", OPTION_COLORS["Optimized"])):
            dev = {c: (df[f"{c}_X{suffix}"].mean(), df[f"{c}_Y{suffix}"].mean()) for c in CORNERS}
            pts = {c: (nom[c][0] + exaggeration * dev[c][0], nom[c][1] + exaggeration * dev[c][1])
                   for c in CORNERS}
            xs, ys = polygon(pts)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, legendgroup=name,
                                     line=dict(color=color, width=2.5), hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[pts[c][0] for c in CORNERS], y=[pts[c][1] for c in CORNERS], mode="markers",
                legendgroup=name, showlegend=False,
                marker=dict(size=9, color=color, line=dict(width=1.5, color="white")),
                customdata=[[c, dev[c][0], dev[c][1]] for c in CORNERS],
                hovertemplate=f"<b>{name}</b> %{{customdata[0]}}<br>X %{{customdata[1]:+.2f}} · "
                              f"Y %{{customdata[2]:+.2f}} mm<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed", title_text="Global Y (mm)")
    fig.update_xaxes(title_text="Global X (mm)", showgrid=True, gridcolor="rgba(137,135,129,0.15)")
    return _base(fig, height)
