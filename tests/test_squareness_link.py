import numpy as np
import pandas as pd
import pytest

from qc import analysis, geometry, loading, metrics
from qc.constants import CORNERS, NOMINALS
from synthetic import make_records, to_export_bytes


def _module(dev: dict, status="FAIL", bat="Type S"):
    row = {"BatteryType": bat, "Status": status, "PartID": "P", "RunNum": 1}
    for c in CORNERS:
        row[f"{c}_X"], row[f"{c}_Y"] = dev[c]
    return row


def _rigid(dx, dy, yaw_deg, bat="Type S"):
    """Corner deviations of a perfectly square module moved rigidly."""
    nom = NOMINALS[bat]
    cx = np.mean([nom[c][0] for c in CORNERS])
    cy = np.mean([nom[c][1] for c in CORNERS])
    out = {}
    for c in CORNERS:
        x, y = geometry.rigid_transform(nom[c][0], nom[c][1], cx, cy, dx, dy, yaw_deg)
        out[c] = (x - nom[c][0], y - nom[c][1])
    return out


def test_pure_placement_has_no_shape():
    df = pd.DataFrame([_module(_rigid(1.2, -0.7, 0.08))])
    d = geometry.decompose(df, 3.0)
    assert d["ShapeMax"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert d["PlacementMax"].iloc[0] > 1.2


def test_shape_only_module_has_no_placement():
    # Opposite corners pushed in opposite directions: no net shift or rotation.
    dev = {"FL": (0.5, 0.0), "FR": (-0.5, 0.0), "RL": (-0.5, 0.0), "RR": (0.5, 0.0)}
    d = geometry.decompose(pd.DataFrame([_module(dev)]), 3.0)
    assert d["ShapeMax"].iloc[0] > 0.3
    assert d["PlacementMax"].iloc[0] < 0.3


def test_nok_cause_classes():
    L = 3.0
    shape = {"FL": (0.9, 0.0), "FR": (-0.9, 0.0), "RL": (-0.9, 0.0), "RR": (0.9, 0.0)}
    near = _rigid(2.6, 0.0, 0.0)  # passes when square, fails with the shape on top
    eaten = {c: (near[c][0] + shape[c][0], near[c][1] + shape[c][1]) for c in CORNERS}
    far = _rigid(3.6, 0.0, 0.0)  # fails even when square
    df = pd.DataFrame([_module(eaten), _module(far), _module(_rigid(0.2, 0.1, 0.0), "PASS")])
    df = geometry.evaluate_status(df, L)
    d = geometry.decompose(df, L)
    assert list(d["Status"]) == ["FAIL", "FAIL", "PASS"]
    assert list(d["NokCause"]) == ["Shape + placement", "Placement", ""]


def test_fisher_exact_known_values():
    assert metrics.fisher_exact(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-6)
    assert metrics.fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)


def test_shape_link_on_data():
    data = to_export_bytes(make_records(n_modules=200, start="2026-06-01 06:00", days=50, seed=9), "csv")
    ld = loading.build_dataset(data, "f.csv")
    a = analysis.analyze(ld.summary, analysis.Settings(), analysis.Scope())
    sq = a.squareness[a.squareness["RunNum"] == 1]
    lk = metrics.shape_link(sq, 1.5)
    assert lk["n"] == len(sq[sq["Status"].isin(["PASS", "FAIL"])])
    assert 0 <= lk["nok_if_square"] <= lk["n_fail"]
    assert lk["fpy_if_square"] >= lk["fpy"] - 100 * lk["pass_lost_if_square"] / lk["n"] - 1e-9
    assert int(lk["by_band"]["N"].sum()) == lk["n"]
    w = metrics.weekly_deformation(sq)
    assert int(w["N"].sum()) == len(sq)
    assert ((w["Share"] >= 0) & (w["Share"] <= 100)).all()
