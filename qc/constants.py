"""Shared constants: nominal geometry, tolerances and the visual vocabulary."""

CORNERS = ["FL", "FR", "RL", "RR"]
CORNER_LABELS = {
    "FL": "Front Left",
    "FR": "Front Right",
    "RL": "Rear Left",
    "RR": "Rear Right",
}
CORNER_INDEX = {1: "FL", 2: "FR", 3: "RL", 4: "RR"}

BATTERY_TYPES = ["Type S", "Type M"]

# Nominal corner coordinates [mm] in the CAD reference system.
# Left corners carry negative Y.
NOMINALS = {
    "Type S": {
        "FL": (2290.48, -559.4),
        "FR": (2290.48, 558.9),
        "RL": (997.28, -559.4),
        "RR": (997.28, 511.1),
    },
    "Type M": {
        "FL": (2290.48, -559.4),
        "FR": (2290.48, 558.9),
        "RL": (609.31, -583.3),
        "RR": (609.31, 535.0),
    },
}

# Squareness root-cause thresholds (kept from the original app).
ANGULAR_DEV_TOL = 0.15  # deg
DIM_DELTA_TOL = 0.8  # mm

STATUSES = ["PASS", "FAIL", "INCOMPLETE"]

# ---------------------------------------------------------------------------
# Colors. Identity colors follow the entity everywhere in the app; status
# colors are reserved for status and never reused for a series.
# Validated with the dataviz palette validator (light surface).
# ---------------------------------------------------------------------------
TYPE_COLORS = {"Type S": "#2a78d6", "Type M": "#eb6834"}
AXIS_COLORS = {"X": "#4a3aa7", "Y": "#1baf7a"}
STATUS_COLORS = {
    "PASS": "#0ca30c",
    "FAIL": "#d03b3b",
    "INCOMPLETE": "#898781",
    "SQUARE OK": "#0ca30c",
    "DEFORMED": "#d03b3b",
}
CAUSE_ORDER = [
    "Parallelogram Tilt",
    "Trapezoidal Width",
    "Trapezoidal Length",
    "Combined Asymmetry",
]
CAUSE_COLORS = {
    "Parallelogram Tilt": "#4a3aa7",
    "Trapezoidal Width": "#1baf7a",
    "Trapezoidal Length": "#eda100",
    "Combined Asymmetry": "#e87ba4",
}
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
LIMIT_COLOR = "#d03b3b"
NOMINAL_COLOR = "#16a34a"  # nominal outlines: green dashed, as in the original app
TARGET_COLOR = "#0ca30c"
HIGHLIGHT_COLOR = "#00b8d9"

# Why a NOK module failed (placement vs shape decomposition).
NOK_CAUSE_ORDER = ["Placement", "Shape + placement", "Shape", "Both"]
NOK_CAUSE_COLORS = {
    "Placement": "#2a78d6",
    "Shape + placement": "#eda100",
    "Shape": "#e87ba4",
    "Both": "#4a3aa7",
}
MA_COLOR = "#f59e0b"  # moving average: amber dashed, as in the original app

# Compensation candidates (identity colors for the three options).
OPTION_COLORS = {
    "No compensation": "#898781",
    "Median-based": "#52514e",
    "Optimized": "#4a3aa7",
    "Manual": "#e87ba4",
}
