"""Data loading, validation, cleaning and feature engineering.

The cleaning and feature-engineering rules are taken from the midterm notebook
(Tasks 4 and 5) so that the app and the notebook treat data the same way.
"""
import numpy as np
import pandas as pd

# ---- Schema (exact columns of the midterm dataset) ---------------------------
REQUIRED_COLUMNS = ["Reservation_ID", "Facility_Type", "User_Type",
                    "Participants", "Lead_Time", "Usage_Status"]
OPTIONAL_COLUMNS = ["Actual_Attendance", "Cancellation_Days_Before"]
NUMERIC_COLUMNS = ["Participants", "Lead_Time", "Actual_Attendance",
                   "Cancellation_Days_Before"]
ALLOWED_STATUS = {"Used", "Cancelled", "No-Show"}
LEAD_ORDER = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]

# Fields recorded AFTER the outcome. They are never used for prediction (leakage).
LEAKAGE_COLUMNS = ["Usage_Status", "Actual_Attendance",
                   "Attendance_Ratio", "Cancellation_Days_Before"]


def lead_time_category(days):
    """Same bins as the midterm notebook (Task 5)."""
    if pd.isna(days):
        return np.nan
    if 0 <= days <= 2:
        return "Last Minute"
    if 3 <= days <= 7:
        return "Short Lead"
    if 8 <= days <= 14:
        return "Medium Lead"
    if days > 14:
        return "Early Booking"
    return np.nan


def _strip_text(df):
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c]
        df[c] = s.where(s.isna(), s.astype(str).str.strip())
    return df


def standardise_columns(df):
    """Trim column names and accept 'User_Category' as an alias of 'User_Type'."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "User_Type" not in df.columns and "User_Category" in df.columns:
        df["User_Type"] = df["User_Category"]
    if "Lead_Time" not in df.columns and "Lead_Time_Days" in df.columns:
        df["Lead_Time"] = df["Lead_Time_Days"]
    return df


def validate_schema(df):
    """Return (errors, warnings). Errors block the upload, warnings do not."""
    errors, warnings = [], []
    if df is None or df.shape[1] == 0:
        return ["The file has no columns."], warnings
    if len(df) == 0:
        return ["The file has headers but no rows (empty dataset)."], warnings

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing)
                      + ". Required: " + ", ".join(REQUIRED_COLUMNS) + ".")
        return errors, warnings

    for c in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        coerced = pd.to_numeric(df[c], errors="coerce")
        bad = int((coerced.isna() & df[c].notna()).sum())
        if bad:
            warnings.append(f"Column '{c}' has {bad} non-numeric value(s); they will be treated as missing.")

    status = df["Usage_Status"].dropna().astype(str).str.strip()
    unknown = sorted(set(status) - ALLOWED_STATUS)
    if unknown:
        warnings.append("Unrecognised Usage_Status value(s): " + ", ".join(unknown)
                        + ". Allowed: Used, Cancelled, No-Show. These rows will be excluded as Not Completed cannot be derived reliably.")
    for c in [c for c in OPTIONAL_COLUMNS if c not in df.columns]:
        warnings.append(f"Optional column '{c}' is absent (not needed for prediction).")
    return errors, warnings


def quality_summary(df):
    """Null / duplicate / invalid-value summary shown on the Dataset page."""
    d = standardise_columns(df)
    nulls = d.isna().sum().rename("Missing values").to_frame()
    nulls["Missing %"] = (nulls["Missing values"] / max(len(d), 1) * 100).round(2)
    summary = {
        "rows": int(len(d)),
        "columns": int(d.shape[1]),
        "exact_duplicates": int(d.duplicated().sum()),
        "duplicate_ids": int(d.duplicated("Reservation_ID").sum()) if "Reservation_ID" in d else 0,
    }
    inv = {}
    if "Participants" in d:
        p = pd.to_numeric(d["Participants"], errors="coerce")
        inv["Participants <= 0"] = int((p <= 0).sum())
        inv["Participants > 500"] = int((p > 500).sum())
    if "Lead_Time" in d:
        lt = pd.to_numeric(d["Lead_Time"], errors="coerce")
        inv["Lead_Time < 0"] = int((lt < 0).sum())
        inv["Lead_Time > 365"] = int((lt > 365).sum())
    return summary, nulls, pd.Series(inv, name="Records", dtype="int64").to_frame()


def clean_dataset(raw):
    """Clean + engineer features (midterm Tasks 4 and 5). Returns (df_clean, log)."""
    df = standardise_columns(raw)
    n_raw = len(df)

    df = df.drop_duplicates().copy()
    dup_removed = n_raw - len(df)

    df = _strip_text(df)
    for c in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # invalid numeric values -> missing
    df.loc[(df["Participants"] <= 0) | (df["Participants"] > 500), "Participants"] = np.nan
    df.loc[(df["Lead_Time"] < 0) | (df["Lead_Time"] > 365), "Lead_Time"] = np.nan

    # normalise Usage_Status spelling, then drop rows whose target cannot be derived
    status_map = {"used": "Used", "cancelled": "Cancelled", "canceled": "Cancelled",
                  "no-show": "No-Show", "no show": "No-Show", "noshow": "No-Show"}
    df["Usage_Status"] = df["Usage_Status"].astype("object").map(
        lambda v: status_map.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)

    before = len(df)
    df = df.dropna(subset=["Usage_Status", "Participants"]).copy()
    invalid_dropped = before - len(df)

    df["Lead_Time_Category"] = df["Lead_Time"].apply(lead_time_category)
    df["Reservation_Outcome"] = np.where(df["Usage_Status"] == "Used", "Completed", "Not Completed")
    if "Actual_Attendance" in df.columns:
        den = df["Participants"].where(df["Participants"] > 0)
        df["Attendance_Ratio"] = df["Actual_Attendance"] / den
    else:
        df["Attendance_Ratio"] = np.nan

    df = df.reset_index(drop=True)
    log = {"original_rows": n_raw, "duplicates_removed": dup_removed,
           "invalid_or_missing_dropped": invalid_dropped, "final_rows": len(df)}
    return df, log


# ----------------------------------------------------------------------------------
# Streamlit page (Module 1)
# ----------------------------------------------------------------------------------
SAMPLE_PATH = "data/sample_facility_reservations.csv"


def read_csv_safely(file):
    """Read an uploaded CSV and return (df, error_message)."""
    try:
        return pd.read_csv(file), None
    except pd.errors.EmptyDataError:
        return None, "The file is empty - it contains no data to read."
    except UnicodeDecodeError:
        return None, "The file is not valid UTF-8 text. Save it as 'CSV UTF-8' and try again."
    except pd.errors.ParserError as e:
        return None, f"The file could not be parsed as CSV: {e}"
    except Exception as e:
        return None, f"The file could not be read: {e}"


def render_dataset_page():
    import streamlit as st

    st.write("Load the reservation data used by the Data Analysis and Dashboard pages. "
             "Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
             + ". Optional: " + ", ".join(f"`{c}`" for c in OPTIONAL_COLUMNS) + ".")
    st.caption("Use synthetic or properly authorised data only. Never upload real student records to a public deployment.")

    options = ["Bundled sample dataset", "Upload my own CSV"]
    choice = st.radio("Data source", options, horizontal=True,
                      index=0 if st.session_state.get("source_choice", options[0]) == options[0] else 1)
    st.session_state["source_choice"] = choice
    if choice == "Upload my own CSV":
        up = st.file_uploader("Upload CSV", type="csv")
        if up is not None:
            df, err = read_csv_safely(up)
            if err:
                st.error(err)
            else:
                df = standardise_columns(df)
                errors, warns = validate_schema(df)
                for e in errors:
                    st.error(e)
                for w in warns:
                    st.warning(w)
                if not errors:
                    st.session_state["raw_df"] = df
                    st.session_state["source_name"] = up.name
                    st.success(f"Loaded **{up.name}** ({len(df)} rows).")
        else:
            st.info("Waiting for a CSV file. Until then the previously loaded data stays active.")
    else:
        if "raw_df" not in st.session_state or not str(st.session_state.get("source_name", "")).startswith("sample"):
            st.session_state["raw_df"] = pd.read_csv(SAMPLE_PATH)
            st.session_state["source_name"] = "sample_facility_reservations.csv (synthetic)"

    raw = st.session_state.get("raw_df")
    if raw is None:
        st.stop()
    st.markdown(f"**Active dataset:** {st.session_state.get('source_name')}")

    st.subheader("Data preview")
    st.dataframe(raw.head(20))
    summary, nulls, invalid = quality_summary(raw)
    c = st.columns(4)
    c[0].metric("Rows", summary["rows"])
    c[1].metric("Columns", summary["columns"])
    c[2].metric("Exact duplicate rows", summary["exact_duplicates"])
    c[3].metric("Duplicate Reservation_IDs", summary["duplicate_ids"])

    a, b = st.columns(2)
    a.markdown("**Null summary**")
    a.dataframe(nulls)
    b.markdown("**Invalid-value checks (before cleaning)**")
    b.dataframe(invalid)
    st.markdown("**Data types**")
    st.dataframe(raw.dtypes.astype(str).rename("dtype").to_frame())

    st.subheader("Cleaning result (same rules as the midterm notebook)")
    clean, log = clean_dataset(raw)
    if clean.empty:
        st.error("No usable rows remain after cleaning (all rows have missing Usage_Status or invalid Participants).")
        return
    c = st.columns(4)
    c[0].metric("Original rows", log["original_rows"])
    c[1].metric("Duplicates removed", log["duplicates_removed"])
    c[2].metric("Invalid / missing dropped", log["invalid_or_missing_dropped"])
    c[3].metric("Final clean rows", log["final_rows"])
    st.caption("Duplicates removed; Participants <= 0 or > 500 and Lead_Time < 0 set to missing; rows without "
               "Usage_Status or Participants dropped; Lead_Time_Category, Reservation_Outcome and "
               "Attendance_Ratio derived. Outcome fields are never used for prediction.")
    st.dataframe(clean.head(20))
    st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode(),
                       "clean_facility_reservations.csv", "text/csv")
