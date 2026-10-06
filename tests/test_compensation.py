import numpy as np
import pytest

from qc import analysis, compensation as comp, loading
from synthetic import make_records, to_export_bytes


@pytest.fixture(scope="module")
def vec():
    data = to_export_bytes(make_records(n_modules=260, start="2026-06-01 06:00", days=70, seed=4), "csv")
    ld = loading.build_dataset(data, "f.csv")
    a = analysis.analyze(ld.summary, analysis.Settings(), analysis.Scope())
    return a.vec


@pytest.fixture(scope="module")
def type_s(vec):
    return vec[vec["BatteryType"] == "Type S"]


def test_optimized_never_worse_than_median_in_window(type_s):
    mx, my, myaw = comp.median_suggestion(type_s)
    med = comp.evaluate(type_s, mx, my, myaw, 3.0)
    res = comp.optimize(type_s, 3.0, margin=0.0, yaw_center=myaw)
    assert res.pass_count >= med["pass"]
    assert res.pass_count >= comp.evaluate(type_s, 0, 0, 0, 3.0)["pass"]


def test_reported_count_is_exact(type_s):
    res = comp.optimize(type_s, 3.0, margin=0.2)
    ev = comp.evaluate(type_s, res.dx, res.dy, res.yaw, 3.0, 0.2)
    assert res.pass_count == ev["pass"]
    assert res.pass_margin_count == ev["pass_margin"]


def test_random_search_does_not_beat_optimizer(type_s):
    res = comp.optimize(type_s, 3.0, margin=0.0)
    rng = np.random.default_rng(0)
    best = max(comp.evaluate(type_s, rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-0.3, 0.3), 3.0)["pass"]
               for _ in range(300))
    assert res.pass_count >= best


def test_margin_prefers_robust_settings(type_s):
    tight = comp.optimize(type_s, 3.0, margin=0.0)
    robust = comp.optimize(type_s, 3.0, margin=0.4)
    m_tight = comp.evaluate(type_s, tight.dx, tight.dy, tight.yaw, 3.0, 0.4)["pass_margin"]
    m_robust = comp.evaluate(type_s, robust.dx, robust.dy, robust.yaw, 3.0, 0.4)["pass_margin"]
    assert m_robust >= m_tight


def test_landscape_and_yaw_profile_agree_with_exact_count(type_s):
    res = comp.optimize(type_s, 3.0, margin=0.0)
    prof = comp.fpy_vs_yaw(type_s, 3.0, res.dx, res.dy, np.array([res.yaw]))
    assert prof[0] == pytest.approx(res.fpy)
    assert res.landscape["fpy"].max() <= res.fpy + 1e-9


def test_backtest_uses_only_past_weeks(vec):
    bt = comp.backtest(vec[vec["BatteryType"] == "Type S"], 3.0, 0.2, window_weeks=4)
    assert bt is not None and bt["n"] > 0
    w = bt["weeks"]
    assert (w["Trained on"] >= 10).all()
    for k in ("none", "median", "optimized"):
        assert 0 <= bt[k] <= 100


def test_readiness_checklist(type_s):
    bt = comp.backtest(type_s, 3.0, 0.2, window_weeks=4)
    items = comp.readiness(type_s, bt=bt)
    assert [i["check"] for i in items] == [
        "Enough modules", "Enough weeks", "Stable X/Y offset", "Stable yaw", "Helps in the backtest"]


def test_no_complete_modules_returns_none(type_s):
    d = type_s.copy()
    d["FL_X"] = np.nan
    assert comp.optimize(d, 3.0) is None
