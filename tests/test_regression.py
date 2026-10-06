"""The new code must reproduce the original app's numbers.

Runs on synthetic data (no DST edge cases, no duplicate rows, no all-zero
reads: the original app crashes on the first and miscounts the others).
Set QC_REAL_FILE=/path/to/export.xlsx to also check a real export.
"""
import io
import os

import numpy as np
import pandas as pd
import pytest

import original_reference as ref
from qc import geometry, loading, metrics
from synthetic import make_records, to_export_bytes

DEV = ["FL_X", "FL_Y", "FR_X", "FR_Y", "RL_X", "RL_Y", "RR_X", "RR_Y"]


class _Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def _pm_first(recs: pd.DataFrame) -> pd.DataFrame:
    """Put a PM row on top. The ORIGINAL app infers the time format from the
    first row; an AM first row makes it drop the PM rows (see the test
    below), which would make this comparison meaningless."""
    i = recs.index[recs["Time"].str.endswith("pm")][0]
    return pd.concat([recs.loc[[i]], recs.drop(index=i)], ignore_index=True)


def _files():
    yield "synthetic", to_export_bytes(_pm_first(
        make_records(n_modules=150, start="2026-06-01 06:00", days=60, seed=5)), "xlsx"), "s.xlsx"
    real = os.environ.get("QC_REAL_FILE")
    if real and os.path.exists(real):
        yield "real", open(real, "rb").read(), os.path.basename(real)


@pytest.fixture(params=list(_files()), ids=lambda f: f[0])
def both(request):
    _, data, name = request.param
    o_raw, o_sum, o_an, o_first = ref.original_pipeline(_Upload(data, name))
    ld = loading.build_dataset(data, name, zero_as_missing=False)
    new = geometry.evaluate_status(ld.summary, 3.0)
    # The original creates a fake run from an exact duplicate row; drop those.
    keys = new[["PartID", "RunNum"]].astype(str).agg("|".join, axis=1)
    o_keys = o_sum[["PartID", "RunNum"]].astype(str).agg("|".join, axis=1)
    o_sum = o_sum[o_keys.isin(keys)]
    return o_sum, o_first, new


def test_module_summary_matches(both):
    o_sum, _, new = both
    m = o_sum.merge(new, on=["PartID", "RunNum"], suffixes=("_o", "_n"))
    assert len(m) == len(o_sum)
    for c in DEV:
        np.testing.assert_array_equal(m[c + "_o"].to_numpy(float), m[c + "_n"].to_numpy(float))
    for c in ["Status", "BatteryType", "CornersOutOfSpec", "MissingCorners", "Date"]:
        assert (m[c + "_o"] == m[c + "_n"]).all(), c
    assert (m["CalendarWeek_o"] == m["CalendarWeek_n"].str[5:]).all()


def test_first_runs_and_fpy_match(both):
    _, o_first, new = both
    fr = metrics.first_runs(new, True)
    m = o_first.merge(fr, on="PartID", suffixes=("_o", "_n"))
    assert len(m) == len(o_first)
    assert (m["Status_o"] == m["Status_n"]).all()
    assert metrics.fpy(fr) == pytest.approx((o_first["Status"] == "PASS").mean() * 100)


def test_centroid_and_yaw_match(both):
    _, o_first, new = both
    fr = metrics.first_runs(new, True)
    o = ref.add_centroid_and_rotation(o_first)
    n = geometry.add_centroid_and_rotation(fr)
    m = o.merge(n, on="PartID", suffixes=("_o", "_n"))
    for c in ["Centroid_X", "Centroid_Y", "Rotation_Angle"]:
        np.testing.assert_allclose(m[c + "_o"].to_numpy(float), m[c + "_n"].to_numpy(float), atol=1e-9)


def test_squareness_matches(both):
    o_sum, _, new = both
    recs = []
    for _, row in o_sum.iterrows():
        if row[DEV].isna().any():
            continue
        nom = ref.get_nominal_coordinates(row["BatteryType"])
        P = {c: (nom[c + "_X"] + row[c + "_X"], nom[c + "_Y"] + row[c + "_Y"]) for c in ["FL", "FR", "RL", "RR"]}
        N = {c: (nom[c + "_X"], nom[c + "_Y"]) for c in ["FL", "FR", "RL", "RR"]}

        def d(p, a, b):
            return np.hypot(p[b][0] - p[a][0], p[b][1] - p[a][1])

        dd = abs((d(P, "FL", "RR") - d(P, "FR", "RL")) - (d(N, "FL", "RR") - d(N, "FR", "RL")))
        wa = (d(P, "FL", "FR") - d(N, "FL", "FR")) - (d(P, "RL", "RR") - d(N, "RL", "RR"))
        la = (d(P, "FL", "RL") - d(N, "FL", "RL")) - (d(P, "FR", "RR") - d(N, "FR", "RR"))
        ang = (ref.calculate_corner_angle(P["FL"], P["FR"], P["RL"])
               - ref.calculate_corner_angle(N["FL"], N["FR"], N["RL"]))
        status, detail = ref.evaluate_deformation(dd, ang, wa, la, 1.5)
        recs.append({"PartID": row["PartID"], "RunNum": row["RunNum"], "st": status, "det": detail, "dd": dd})
    o_sq = pd.DataFrame(recs)
    n_sq = geometry.squareness(new, 1.5)
    m = o_sq.merge(n_sq, on=["PartID", "RunNum"])
    assert len(m) == len(o_sq) == len(n_sq)
    assert (m["st"] == m["SquarenessStatus"]).all()
    assert (m["det"] == m["RootCause"]).all()
    np.testing.assert_allclose(m["dd"], m["DeltaDiag"], atol=1e-9)


@pytest.mark.parametrize("dx,dy,yaw", [(0.5, -0.3, 0.05), (-1.2, 0.7, -0.12)])
def test_simulation_matches(both, dx, dy, yaw):
    _, o_first, new = both
    fr = metrics.first_runs(new, True)
    o = ref.simulate_compensation(o_first, dx, dy, yaw, 3.0)
    n = geometry.simulate_compensation(fr, dx, dy, yaw, 3.0)
    m = o.merge(n, on="PartID", suffixes=("_o", "_n"))
    assert (m["Status_Sim_o"] == m["Status_Sim_n"]).all()
    for c in DEV:
        np.testing.assert_allclose(m[c + "_Sim_o"].to_numpy(float), m[c + "_Sim_n"].to_numpy(float), atol=1e-9)
    og = ref.build_sim_geometry(o)[["PartID", "Rotation_Angle"]]
    ng = pd.concat([n[["PartID"]], geometry.sim_geometry(n)], axis=1)
    mm = og.merge(ng, on="PartID")
    np.testing.assert_allclose(mm["Rotation_Angle"].to_numpy(float), mm["Rotation_Angle_Sim"].to_numpy(float), atol=1e-9)


def test_corner_offsets_match():
    for t in ("Type S", "Type M"):
        a = ref.compute_corner_offsets(t, 0.5, -0.3, 0.05)
        b = geometry.corner_offsets(t, 0.5, -0.3, 0.05)
        np.testing.assert_allclose(a.iloc[:, 2:].to_numpy(float), b.iloc[:, 2:].to_numpy(float))


def test_original_loses_rows_when_first_time_is_am_new_code_does_not():
    recs = make_records(n_modules=150, start="2026-06-01 06:00", days=60, seed=5)
    i = recs.index[recs["Time"].str.endswith("am")][0]
    recs = pd.concat([recs.loc[[i]], recs.drop(index=i)], ignore_index=True)
    data = to_export_bytes(recs, "xlsx")
    o_raw = ref.original_pipeline(_Upload(data, "s.xlsx"))[0]
    lost = int(o_raw["ParsedDate"].isna().sum())
    assert lost > len(recs) * 0.2  # the original silently loses a large share
    ld = loading.build_dataset(data, "s.xlsx")
    assert ld.health["rows_without_valid_time"] == 0
