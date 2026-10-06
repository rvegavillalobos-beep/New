"""Synthetic measurement exports in the same layout as the real one
("Corner Cell Deviations for All"). Used by the "demo data" button and tests.
The numbers are invented; only the format matches the real export."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

FEATURES = {
    "Type S": ["72_L0324_aa", "72_R0301_aa", "72_L0324_da", "72_R0302_da"],
    "Type M": ["72_L0324_aa", "72_R0301_aa", "72_L0324_dj", "72_R0301_dj"],
}
PREFIX = {"Type S": "FO88995230126B", "Type M": "FO88995310126B"}

# Per-corner systematic bias (mm) and yaw-like skew used to make the data
# behave like a real process: corners are biased differently, so centering
# the average is not the same as maximizing the pass rate.
BIAS = {
    "Type S": {"FL": (0.6, -0.4), "FR": (-1.4, 0.9), "RL": (0.2, -1.0), "RR": (-0.9, 0.3)},
    "Type M": {"FL": (0.3, 0.6), "FR": (-1.6, 1.3), "RL": (0.5, -1.7), "RR": (-1.2, -0.2)},
}


def make_records(
    n_modules: int = 400,
    start: str = "2026-09-01 06:00",
    days: int = 120,
    seed: int = 7,
    repeat_share: float = 0.35,
    missing_share: float = 0.04,
    noise: float = 1.6,
    extra_times: list[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    rows = []
    corners = ["FL", "FR", "RL", "RR"]
    times = sorted(t0 + pd.to_timedelta(rng.uniform(0, days * 24 * 60, n_modules), unit="min"))
    if extra_times:
        times = sorted(list(times[: n_modules - len(extra_times)]) + [pd.Timestamp(t) for t in extra_times])
    for i, t in enumerate(times):
        btype = "Type S" if rng.random() < 0.65 else "Type M"
        pid = f"{PREFIX[btype]}{100 + i:03d}N{rng.integers(1, 5):07d}11296112"
        drift = 0.6 * np.sin(i / max(n_modules, 1) * 3.0)  # slow process drift
        n_runs = 1 + (rng.random() < repeat_share) + (rng.random() < repeat_share / 3)
        for r in range(n_runs):
            when = t + pd.Timedelta(minutes=int(rng.integers(5, 600)) * r)
            yaw = rng.normal(0.0, 0.0006)  # rad, shared by the 4 corners
            shift = rng.normal(0, 0.35, 2)
            for c, feat in zip(corners, FEATURES[btype]):
                bx, by = BIAS[btype][c]
                lever_x = {"FL": -560, "FR": 560, "RL": -560, "RR": 520}[c]
                lever_y = {"FL": 650, "FR": 650, "RL": -650, "RR": -650}[c]
                x = bx + drift + shift[0] - yaw * lever_x + rng.normal(0, noise * 0.5) - 0.8 * r
                y = by + shift[1] + yaw * lever_y + rng.normal(0, noise * 0.5) + 0.3 * r
                if rng.random() < missing_share:
                    x = y = np.nan
                rows.append({
                    "Time": when.strftime("%b %-d, %Y %-I:%M%p").replace("AM", "am").replace("PM", "pm"),
                    "Part ID": pid, "Featurename": feat,
                    "X deviation": x, "X lower limit": -3, "X upper limit": 3,
                    "Y deviation": y, "Y lower limit": -3, "Y upper limit": 3,
                    "Z deviation": rng.normal(-0.2, 0.4), "Z lower limit": -1.25, "Z upper limit": 1.21,
                })
    df = pd.DataFrame(rows)
    # Export is newest first, like the real one.
    return df.iloc[::-1].reset_index(drop=True)


def to_export_bytes(df: pd.DataFrame, fmt: str = "xlsx") -> bytes:
    """Write with the two title rows above the header, like the real export."""
    buf = io.BytesIO()
    if fmt == "csv":
        text = "Corner Cell Deviations for All\n\n" + df.to_csv(index=False)
        return text.encode()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([["Corner Cell Deviations for All"], [None]]).to_excel(
            w, index=False, header=False, startrow=0
        )
        df.to_excel(w, index=False, startrow=2)
    return buf.getvalue()


def demo_bytes(seed: int = 7) -> bytes:
    return to_export_bytes(make_records(n_modules=420, start="2026-06-01 06:00", days=120, seed=seed), "xlsx")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_export.xlsx"
    recs = make_records()
    open(out, "wb").write(to_export_bytes(recs, "csv" if out.endswith(".csv") else "xlsx"))
    print(f"wrote {out}: {len(recs)} rows")
