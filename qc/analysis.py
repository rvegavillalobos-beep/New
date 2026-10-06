"""One place that turns the module summary + settings + filters into every
table the pages show, so all pages agree on the same numbers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import geometry, metrics


@dataclass(frozen=True)
class Settings:
    spec_limit: float = 3.0
    diag_tol: float = 1.5
    fpy_target: float = 90.0
    exclude_incomplete: bool = True


@dataclass(frozen=True)
class Scope:
    types: tuple = ("Type S", "Type M")
    week_from: int | None = None  # WeekKey, inclusive
    week_to: int | None = None


@dataclass
class Analysis:
    runs: pd.DataFrame        # every run in scope, with status
    first: pd.DataFrame       # first run of every module in scope
    vec: pd.DataFrame         # first runs + centroid / yaw
    squareness: pd.DataFrame  # complete runs in scope + deformation + placement/shape split
    weekly: pd.DataFrame      # weekly first-run FPY
    pareto: pd.DataFrame
    capability: pd.DataFrame
    settings: Settings
    scope: Scope

    @property
    def fpy(self) -> float:
        return metrics.fpy(self.first)


def module_key(df: pd.DataFrame) -> pd.Series:
    return (df["PartID"].astype(str) + " | Run " + df["RunNum"].astype(str)
            + " | " + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"))


def _in_scope(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    m = df["BatteryType"].isin(scope.types)
    if scope.week_from is not None:
        m &= df["WeekKey"] >= scope.week_from
    if scope.week_to is not None:
        m &= df["WeekKey"] <= scope.week_to
    return df[m]


def analyze(summary: pd.DataFrame, settings: Settings, scope: Scope, ma_window: int = 3) -> Analysis:
    runs_all = geometry.evaluate_status(summary, settings.spec_limit)
    runs_all["_mod_key"] = module_key(runs_all) if len(runs_all) else pd.Series(dtype=str)
    # First runs are chosen on the full history (a module's Run 1 is its
    # Run 1 regardless of the filters), then filtered.
    first_all = metrics.first_runs(runs_all, settings.exclude_incomplete)
    runs = _in_scope(runs_all, scope).reset_index(drop=True)
    first = _in_scope(first_all, scope).reset_index(drop=True)
    vec = geometry.add_centroid_and_rotation(first)
    sq = geometry.decompose(geometry.squareness(runs, settings.diag_tol), settings.spec_limit)
    weekly = metrics.weekly_fpy(first, ma_window)
    pareto = metrics.failure_pareto(first, settings.spec_limit)
    cap = metrics.capability(first, settings.spec_limit)
    return Analysis(runs=runs, first=first, vec=vec, squareness=sq, weekly=weekly,
                    pareto=pareto, capability=cap, settings=settings, scope=scope)


def findings(a: Analysis) -> list[dict]:
    """Findings for the Overview (FPY, trend, NOK causes, offsets / scatter)."""
    return metrics.findings(a.first, a.weekly, a.pareto, a.capability, None,
                            a.settings.fpy_target, a.settings.spec_limit)
