"""Verbatim copy of the ORIGINAL app's logic (RICKHARDV/streamlit_app.py),
used only by the regression tests to prove the new code reproduces it."""
import numpy as np
import pandas as pd


def determine_battery_type(part_id, feature_names):
    """Determines whether the battery is Type M or Type S by evaluating PartID and feature set."""
    p_id = str(part_id).upper().strip()
    if isinstance(feature_names, (list, set, pd.Series)):
        f_combined = " ".join([str(f).upper().strip() for f in feature_names])
    else:
        f_combined = str(feature_names).upper().strip()

    if (
        "_DJ" in p_id or "_DJ" in f_combined
        or "_DI" in p_id or "_DI" in f_combined
    ):
        return "Type M"
    if (
        "_M" in p_id or " M" in p_id or "TYPE M" in p_id
        or "TYPEM" in p_id or p_id.endswith("M")
    ):
        return "Type M"
    if "_DA" in p_id or "_DA" in f_combined:
        return "Type S"
    if "_S" in p_id or " S" in p_id or "TYPE S" in p_id or "TYPES" in p_id:
        return "Type S"
    return "Type S"


def extract_corner_index(feature_name, part_id):
    f = str(feature_name).lower().strip()
    if not f:
        return 0
    is_type_m = determine_battery_type(part_id, feature_name) == "Type M"

    if "l0324_aa" in f or "l324_aa" in f:
        return 1
    if "r0301_aa" in f or "r301_aa" in f:
        return 2

    if is_type_m:
        if any(k in f for k in ["l0324_dj", "l324_dj", "l0324_di", "l324_di"]):
            return 3
        if any(k in f for k in ["r0301_dj", "r301_dj", "r0301_di", "r301_di"]):
            return 4
    else:
        if "l0324_da" in f or "l324_da" in f:
            return 3
        if "r0302_da" in f or "r302_da" in f:
            return 4

    if "fl" in f or "c1" in f:
        return 1
    elif "fr" in f or "c2" in f:
        return 2
    elif "rl" in f or "c3" in f:
        return 3
    elif "rr" in f or "c4" in f or "r302" in f or "r301" in f or "r00301" in f:
        return 4
    return 0


def get_nominal_coordinates(bat_type):
    # NOTE: Sign of FL_Y and RL_Y corrected to negative (left-side corners
    # must sit on the opposite side of the Y axis relative to FR/RR),
    # matching the CAD reference system where left corners carry negative Y.
    if str(bat_type).upper() == "TYPE S":
        return {
            "FL_X": 2290.48, "FL_Y": -559.4,
            "FR_X": 2290.48, "FR_Y": 558.9,
            "RL_X": 997.28, "RL_Y": -559.4,
            "RR_X": 997.28, "RR_Y": 511.1,
        }
    else:
        return {
            "FL_X": 2290.48, "FL_Y": -559.4,
            "FR_X": 2290.48, "FR_Y": 558.9,
            "RL_X": 609.31, "RL_Y": -583.3,
            "RR_X": 609.31, "RR_Y": 535.0,
        }


def calculate_corner_angle(a, b, c):
    v_ab = np.array(b) - np.array(a)
    v_ac = np.array(c) - np.array(a)
    dot_prod = np.dot(v_ab, v_ac)
    mag_ab = np.linalg.norm(v_ab)
    mag_ac = np.linalg.norm(v_ac)
    if mag_ab == 0 or mag_ac == 0:
        return 0.0
    cos_theta = np.clip(dot_prod / (mag_ab * mag_ac), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def evaluate_deformation(delta_diags, angle_fl_dev, diff_ancho, diff_largo, max_diag_tol):
    angular_dev_tol = 0.15
    dim_delta_tol = 0.8

    if delta_diags > max_diag_tol:
        status = "DEFORMED"
        if abs(angle_fl_dev) > angular_dev_tol and abs(diff_ancho) < dim_delta_tol:
            detail = f"PARALLELOGRAM DISTORTION (Tilt: {angle_fl_dev:+.2f}°)"
        elif abs(diff_ancho) >= dim_delta_tol:
            detail = f"TRAPEZOIDAL WIDTH VARIATION (Delta: {diff_ancho:+.2f} mm)"
        elif abs(diff_largo) >= dim_delta_tol:
            detail = f"TRAPEZOIDAL LENGTH VARIATION (Delta: {diff_largo:+.2f} mm)"
        else:
            detail = f"COMBINED ASYMMETRY (Diagonal Delta: {delta_diags:.2f} mm)"
    else:
        status = "SQUARE OK"
        detail = "Geometry within acceptable tolerance"

    return status, detail


def add_centroid_and_rotation(df):
    """
    Given a dataframe with FL_X/Y, FR_X/Y, RL_X/Y, RR_X/Y deviations and a
    BatteryType column, returns a COPY with added columns:
      Centroid_X, Centroid_Y, Vector_Magnitude, Rotation_Angle
    This reuses exactly the same logic already used in the Vector Drift tab,
    so it can be shared safely with the Compensation Calculator tab.
    """
    df_out = df.copy()
    df_out["Centroid_X"] = df_out[["FL_X", "FR_X", "RL_X", "RR_X"]].mean(axis=1)
    df_out["Centroid_Y"] = df_out[["FL_Y", "FR_Y", "RL_Y", "RR_Y"]].mean(axis=1)
    df_out["Vector_Magnitude"] = np.sqrt(df_out["Centroid_X"] ** 2 + df_out["Centroid_Y"] ** 2)

    rotation_list = []
    for _, row in df_out.iterrows():
        if pd.isna(row["FL_X"]) or pd.isna(row["FR_X"]) or pd.isna(row["RL_X"]) or pd.isna(row["RR_X"]):
            rotation_list.append(np.nan)
            continue

        nom = get_nominal_coordinates(row["BatteryType"])
        f_nom_x, f_nom_y = (nom["FL_X"] + nom["FR_X"]) / 2, (nom["FL_Y"] + nom["FR_Y"]) / 2
        r_nom_x, r_nom_y = (nom["RL_X"] + nom["RR_X"]) / 2, (nom["RL_Y"] + nom["RR_Y"]) / 2
        angle_nom = np.degrees(np.arctan2(f_nom_y - r_nom_y, f_nom_x - r_nom_x))

        fl_x_act, fl_y_act = nom["FL_X"] + row["FL_X"], nom["FL_Y"] + row["FL_Y"]
        fr_x_act, fr_y_act = nom["FR_X"] + row["FR_X"], nom["FR_Y"] + row["FR_Y"]
        rl_x_act, rl_y_act = nom["RL_X"] + row["RL_X"], nom["RL_Y"] + row["RL_Y"]
        rr_x_act, rr_y_act = nom["RR_X"] + row["RR_X"], nom["RR_Y"] + row["RR_Y"]

        f_act_x, f_act_y = (fl_x_act + fr_x_act) / 2, (fl_y_act + fr_y_act) / 2
        r_act_x, r_act_y = (rl_x_act + rr_x_act) / 2, (rl_y_act + rr_y_act) / 2
        angle_act = np.degrees(np.arctan2(f_act_y - r_act_y, f_act_x - r_act_x))

        diff_angle = (angle_act - angle_nom + 180) % 360 - 180
        rotation_list.append(diff_angle)

    df_out["Rotation_Angle"] = rotation_list
    return df_out


def order_corners_convex(points_dict):
    """
    points_dict: {"FL": (x, y), "FR": (x, y), "RL": (x, y), "RR": (x, y)}
    Returns the corner names ordered by angle around the centroid,
    guaranteeing a simple polygon (no self-intersection), even if the
    Exaggeration Factor flips the relative order of two very close corners
    (such as FL/FR in this dataset, nominally separated by only 0.5 mm).
    """
    cx = np.mean([p[0] for p in points_dict.values()])
    cy = np.mean([p[1] for p in points_dict.values()])

    def angle(name):
        x, y = points_dict[name]
        return np.arctan2(y - cy, x - cx)

    return sorted(points_dict.keys(), key=angle)


CORNER_NAMES = ["FL", "FR", "RL", "RR"]


def compute_robust_bias(df_window, method="median"):
    """
    Computes the robust central tendency (median by default) of Centroid_X,
    Centroid_Y and Rotation_Angle for a filtered dataframe window.
    Also returns dispersion indicators (std, IQR, MAD) to flag unstable
    processes / low-confidence recommendations.
    """
    result = {}
    for col in ["Centroid_X", "Centroid_Y", "Rotation_Angle"]:
        series = df_window[col].dropna()
        if series.empty:
            result[col] = {"value": np.nan, "std": np.nan, "iqr": np.nan, "mad": np.nan, "n": 0}
            continue

        central = series.mean() if method == "mean" else series.median()
        std_val = series.std()
        q75, q25 = np.percentile(series, [75, 25])
        iqr_val = q75 - q25
        mad_val = np.median(np.abs(series - series.median()))

        result[col] = {"value": central, "std": std_val, "iqr": iqr_val, "mad": mad_val, "n": len(series)}
    return result


def rigid_transform_point(x, y, pivot_x, pivot_y, dx, dy, theta_deg):
    """
    Applies a rigid 2D roto-translation to point (x, y):
      1. Express point relative to pivot.
      2. Rotate by theta_deg (degrees) around the pivot.
      3. Translate back to global coords and apply compensation offset (dx, dy).
    No scale, no shear, no per-corner independent deformation.
    """
    theta_rad = np.radians(theta_deg)
    x_rel = x - pivot_x
    y_rel = y - pivot_y
    x_rot = x_rel * np.cos(theta_rad) - y_rel * np.sin(theta_rad)
    y_rot = x_rel * np.sin(theta_rad) + y_rel * np.cos(theta_rad)
    x_new = x_rot + pivot_x + dx
    y_new = y_rot + pivot_y + dy
    return x_new, y_new


def get_nominal_center(nom):
    """
    Pivot choice: nominal global center of the module = average of the
    4 nominal corners (FL, FR, RL, RR). Used consistently for corner-offset
    derivation and historical simulation.
    """
    cx = (nom["FL_X"] + nom["FR_X"] + nom["RL_X"] + nom["RR_X"]) / 4.0
    cy = (nom["FL_Y"] + nom["FR_Y"] + nom["RL_Y"] + nom["RR_Y"]) / 4.0
    return cx, cy


def compute_corner_offsets(bat_type, rec_x, rec_y, rec_yaw):
    """
    Derives the equivalent per-corner displacement of applying the
    applied rigid compensation (rec_x, rec_y, rec_yaw) to the nominal
    polygon of a given BatteryType. These are NOT independently optimized;
    they are a direct consequence of the rigid roto-translation.
    """
    nom = get_nominal_coordinates(bat_type)
    pivot_x, pivot_y = get_nominal_center(nom)

    offsets = []
    for corner in CORNER_NAMES:
        nx, ny = nom[f"{corner}_X"], nom[f"{corner}_Y"]
        cx, cy = rigid_transform_point(nx, ny, pivot_x, pivot_y, rec_x, rec_y, rec_yaw)
        offsets.append({
            "BatteryType": bat_type,
            "Corner": corner,
            "Equivalent_Offset_X_mm": round(cx - nx, 3),
            "Equivalent_Offset_Y_mm": round(cy - ny, 3),
        })
    return pd.DataFrame(offsets)


def simulate_compensation(df_piece, rec_x, rec_y, rec_yaw, spec_limit_val):
    """
    Simulates the historical effect of applying the rigid compensation
    (rec_x, rec_y, rec_yaw) to every piece in df_piece. Recomputes simulated
    per-corner deviations, CornersOutOfSpec_Sim and Status_Sim using the same
    PASS/FAIL/INCOMPLETE rules already used in the app. Operates on a COPY;
    never touches the original dataframes.
    """
    sim_records = []

    for _, row in df_piece.iterrows():
        bat_type = row["BatteryType"]
        nom = get_nominal_coordinates(bat_type)
        pivot_x, pivot_y = get_nominal_center(nom)

        sim_row = row.to_dict()
        corners_out_of_spec_sim = 0
        missing_corners_sim = 0

        for corner in CORNER_NAMES:
            dev_x, dev_y = row.get(f"{corner}_X"), row.get(f"{corner}_Y")

            if pd.isna(dev_x) or pd.isna(dev_y):
                missing_corners_sim += 1
                sim_row[f"{corner}_X_Sim"] = np.nan
                sim_row[f"{corner}_Y_Sim"] = np.nan
                continue

            nx, ny = nom[f"{corner}_X"], nom[f"{corner}_Y"]
            act_x, act_y = nx + dev_x, ny + dev_y

            comp_x, comp_y = rigid_transform_point(act_x, act_y, pivot_x, pivot_y, rec_x, rec_y, rec_yaw)

            new_dev_x = comp_x - nx
            new_dev_y = comp_y - ny
            sim_row[f"{corner}_X_Sim"] = new_dev_x
            sim_row[f"{corner}_Y_Sim"] = new_dev_y

            if abs(new_dev_x) > spec_limit_val or abs(new_dev_y) > spec_limit_val:
                corners_out_of_spec_sim += 1

        is_complete_sim = missing_corners_sim == 0
        if not is_complete_sim:
            status_sim = "INCOMPLETE"
        elif corners_out_of_spec_sim > 0:
            status_sim = "FAIL"
        else:
            status_sim = "PASS"

        sim_row["CornersOutOfSpec_Sim"] = corners_out_of_spec_sim
        sim_row["MissingCorners_Sim"] = missing_corners_sim
        sim_row["Status_Sim"] = status_sim
        sim_records.append(sim_row)

    return pd.DataFrame(sim_records)


def build_sim_geometry(df_sim):
    """
    Takes the output of simulate_compensation() (with *_Sim columns) and
    builds a dataframe reusing the FL_X/FL_Y/... column names but filled
    with the simulated (compensated) deviations, so add_centroid_and_rotation()
    can be reused to get Centroid_X, Centroid_Y and Rotation_Angle for the
    *compensated* scenario, without duplicating logic.
    """
    df_geo = df_sim.copy()
    for corner in CORNER_NAMES:
        df_geo[f"{corner}_X"] = df_sim[f"{corner}_X_Sim"]
        df_geo[f"{corner}_Y"] = df_sim[f"{corner}_Y_Sim"]
    return add_centroid_and_rotation(df_geo)


def original_pipeline(uploaded_file, spec_limit=3.0, exclude_incomplete=True):
    if True:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, skiprows=2)
        else:
            df_raw = pd.read_excel(uploaded_file, skiprows=2)

        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        time_col = [c for c in df_raw.columns if "time" in c.lower()][0]
        part_col = [c for c in df_raw.columns if "part" in c.lower()][0]
        feat_col = [c for c in df_raw.columns if "feature" in c.lower()][0]
        x_dev_col = [c for c in df_raw.columns if "x" in c.lower() and "deviation" in c.lower()][0]
        y_dev_col = [c for c in df_raw.columns if "y" in c.lower() and "deviation" in c.lower()][0]

        # Timezone conversion (Germany -> Mexico City)
        df_raw["ParsedDate"] = pd.to_datetime(df_raw[time_col], errors="coerce")
        df_raw["ParsedDate"] = (
            df_raw["ParsedDate"]
            .dt.tz_localize("Europe/Berlin", ambiguous="NaT")
            .dt.tz_convert("America/Mexico_City")
            .dt.tz_localize(None)
        )
        df_raw["CalendarWeek"] = (
            "CW" + df_raw["ParsedDate"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

        df_raw["BatteryType"] = df_raw.apply(
            lambda row: determine_battery_type(row[part_col], row[feat_col]), axis=1
        )
        df_raw["CornerIndex"] = df_raw.apply(
            lambda row: extract_corner_index(row[feat_col], row[part_col]), axis=1
        )
        df_raw["X_Val"] = pd.to_numeric(df_raw[x_dev_col], errors="coerce")
        df_raw["Y_Val"] = pd.to_numeric(df_raw[y_dev_col], errors="coerce")

        df_raw = df_raw.sort_values(by="ParsedDate").reset_index(drop=True)

        # ----- Run inference -----
        # NOTE: BaseKey is based ONLY on PartID (not PartID + Date). This
        # ensures a given physical module keeps a single, continuous run
        # history across ALL the days it was measured. Previously, including
        # the date in the key caused the run counter to reset back to 1
        # every time the same module was re-measured on a different day,
        # producing multiple "Run 1" records for the same physical module.
        base_keys = []
        current_runs = []
        run_tracker = {}
        mod_corner_history = {}

        for _, r_item in df_raw.iterrows():
            p_val = r_item[part_col]
            base_key = str(p_val)
            f_name = r_item[feat_col]

            if base_key not in run_tracker:
                run_tracker[base_key] = 1
                mod_corner_history[base_key] = f_name
            else:
                if f_name in mod_corner_history[base_key]:
                    run_tracker[base_key] += 1
                    mod_corner_history[base_key] = f_name
                else:
                    mod_corner_history[base_key] += f";{f_name}"

            base_keys.append(base_key)
            current_runs.append(run_tracker[base_key])

        df_raw["BaseKey"] = base_keys
        df_raw["CurrentRun"] = current_runs

        # ----- Build df_summary -----
        modules_data = []
        grouped_runs = df_raw.groupby(["BaseKey", "CurrentRun"])

        for (b_key, c_run), group in grouped_runs:
            first_row = group.iloc[0]
            full_dt = first_row["ParsedDate"]
            cal_week = first_row["CalendarWeek"]
            p_val = first_row[part_col]
            bat_type = determine_battery_type(p_val, group[feat_col])

            corners = {1: (np.nan, np.nan), 2: (np.nan, np.nan), 3: (np.nan, np.nan), 4: (np.nan, np.nan)}
            for _, r_item in group.iterrows():
                c_idx = r_item["CornerIndex"]
                if c_idx in [1, 2, 3, 4]:
                    corners[c_idx] = (r_item["X_Val"], r_item["Y_Val"])

            corners_out_of_spec = 0
            missing_corners = 0
            for c_idx in [1, 2, 3, 4]:
                cx, cy = corners[c_idx]
                if pd.isna(cx) or pd.isna(cy):
                    missing_corners += 1
                else:
                    if abs(cx) > spec_limit or abs(cy) > spec_limit:
                        corners_out_of_spec += 1

            is_complete = missing_corners == 0
            if not is_complete:
                status = "INCOMPLETE"
            elif corners_out_of_spec > 0:
                status = "FAIL"
            else:
                status = "PASS"

            modules_data.append({
                "Date": full_dt,
                "CalendarWeek": cal_week,
                "PartID": p_val,
                "BaseKey": b_key,
                "BatteryType": bat_type,
                "RunNum": c_run,
                "FL_X": corners[1][0], "FL_Y": corners[1][1],
                "FR_X": corners[2][0], "FR_Y": corners[2][1],
                "RL_X": corners[3][0], "RL_Y": corners[3][1],
                "RR_X": corners[4][0], "RR_Y": corners[4][1],
                "CornersOutOfSpec": corners_out_of_spec,
                "MissingCorners": missing_corners,
                "IsComplete": is_complete,
                "Status": status,
            })

        df_summary = pd.DataFrame(modules_data)
        df_summary = df_summary.sort_values(by=["Date", "RunNum"], ascending=True).reset_index(drop=True)

        cols = [
            "Date", "CalendarWeek", "PartID", "BatteryType", "RunNum",
            "FL_X", "FL_Y", "FR_X", "FR_Y", "RL_X", "RL_Y", "RR_X", "RR_Y",
            "CornersOutOfSpec", "MissingCorners", "IsComplete", "Status",
        ]
        df_summary = df_summary[cols]

        # ----- Dataset toggles -----
        df_analysis = df_summary.copy()

        first_run_records = []
        for _, group in df_summary.groupby("PartID"):
            r1 = group[group["RunNum"] == 1]
            rec = r1.iloc[0].copy() if not r1.empty else group.iloc[0].copy()
            if exclude_incomplete and rec["MissingCorners"] > 0:
                rec["Status"] = "FAIL"
            first_run_records.append(rec)

        df_first_valid = (
            pd.DataFrame(first_run_records) if first_run_records else pd.DataFrame(columns=df_summary.columns)
        )
    return df_raw, df_summary, df_analysis, df_first_valid
