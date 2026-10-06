import pandas as pd
import pytest

from qc import loading
from synthetic import make_records, to_export_bytes


def _records():
    return make_records(n_modules=80, start="2026-06-01 06:00", days=40, seed=1)


def test_reads_xlsx_and_csv_the_same():
    recs = _records()
    a = loading.build_dataset(to_export_bytes(recs, "xlsx"), "f.xlsx")
    b = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv")
    assert a.health["module_runs"] == b.health["module_runs"] > 80
    pd.testing.assert_frame_equal(a.summary, b.summary, check_dtype=False)


def test_semicolon_csv_with_decimal_comma():
    recs = _records()
    text = "Corner Cell Deviations for All\n\n" + recs.to_csv(index=False, sep=";", decimal=",")
    a = loading.build_dataset(text.encode(), "f.csv")
    b = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv")
    assert a.health["module_runs"] == b.health["module_runs"]


def test_bad_file_raises_a_clear_error():
    with pytest.raises(loading.DataFormatError):
        loading.build_dataset(b"a,b\n1,2\n", "x.csv")


def test_daylight_saving_never_drops_rows():
    recs = make_records(n_modules=60, start="2026-10-20 06:00", days=170, seed=2,
                        extra_times=["2026-10-25 02:30", "2027-03-28 02:15"])
    ld = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv")
    assert ld.health["rows_without_valid_time"] == 0
    assert ld.health["dst_ambiguous_rows"] >= 4
    assert ld.health["dst_nonexistent_rows"] >= 4
    assert ld.summary["Date"].notna().all()


def test_calendar_weeks_sort_across_new_year():
    recs = make_records(n_modules=60, start="2026-12-10 06:00", days=40, seed=3)
    ld = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv")
    w = ld.summary[["WeekKey", "CalendarWeek"]].drop_duplicates().sort_values("WeekKey")
    labels = w["CalendarWeek"].tolist()
    assert labels == sorted(labels)
    assert any(lbl.startswith("2027-") for lbl in labels)
    assert any(lbl.startswith("2026-") for lbl in labels)


def test_exact_duplicate_rows_do_not_create_a_fake_run():
    recs = _records()
    dup = pd.concat([recs.iloc[[0]], recs], ignore_index=True)
    a = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv")
    b = loading.build_dataset(to_export_bytes(dup, "csv"), "f.csv")
    assert b.health["duplicate_rows_removed"] == 1
    assert a.health["module_runs"] == b.health["module_runs"]


def test_lookalike_part_ids_are_flagged_and_can_be_merged():
    recs = _records()
    pid = recs["Part ID"].iloc[0]
    alt = pid.replace("O", "0", 1)
    extra = recs[recs["Part ID"] == pid].copy()
    extra["Part ID"] = alt
    extra["Time"] = "Dec 1, 2026 9:00am"
    both = pd.concat([recs, extra], ignore_index=True)
    a = loading.build_dataset(to_export_bytes(both, "csv"), "f.csv")
    assert (min(pid, alt), max(pid, alt)) in [tuple(sorted(p)) for p in a.health["similar_part_ids"]]
    b = loading.build_dataset(to_export_bytes(both, "csv"), "f.csv", merge_lookalike_ids=True)
    assert b.health["modules"] == a.health["modules"] - 1


def test_all_zero_readings_are_missing_by_default():
    recs = _records()
    recs.loc[3, ["X deviation", "Y deviation", "Z deviation"]] = 0.0
    on = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv")
    off = loading.build_dataset(to_export_bytes(recs, "csv"), "f.csv", zero_as_missing=False)
    assert on.health["zero_readings"] == 1
    miss_on = on.summary[[c for c in on.summary.columns if c[-2:] in ("_X", "_Y")]].isna().sum().sum()
    miss_off = off.summary[[c for c in off.summary.columns if c[-2:] in ("_X", "_Y")]].isna().sum().sum()
    assert miss_on == miss_off + 2


def test_runs_count_up_when_a_corner_repeats():
    parts = ["A", "A", "A", "A", "A", "A", "B", "B"]
    feats = ["c1", "c2", "c3", "c4", "c1", "c2", "c1", "c2"]
    assert loading.infer_runs(parts, feats) == [1, 1, 1, 1, 2, 2, 1, 1]


def test_symmetric_limit_from_file():
    assert loading.symmetric_xy_limit({"X": (-3.0, 3.0), "Y": (-3.0, 3.0)}) == 3.0
    assert loading.symmetric_xy_limit({"X": (-3.0, 3.0), "Y": (-2.0, 2.0)}) is None
    assert loading.symmetric_xy_limit({}) is None
