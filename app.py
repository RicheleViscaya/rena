"""Campus Facility Reservation Monitoring - SAD predictive system (Streamlit)."""
import pandas as pd
import streamlit as st

from modules.dashboard import render_dashboard
from modules.data_processing import SAMPLE_PATH, clean_dataset, render_dataset_page
from modules.exploratory_analysis import render_eda
from modules.midterm_page import render_midterm
from modules.prediction import (load_metadata, load_midterm_results, load_model,
                                render_prediction)
from modules.system_design import render_system_design

st.set_page_config(page_title="SAD Predictive System", page_icon=":classical_building:", layout="wide")

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design", "Midterm Work"]
menu = st.sidebar.selectbox("Module", PAGES)
role = st.sidebar.selectbox("Signed in as (role)", ["Facility Administrator", "Facility User", "Analyst"],
                            help="Only the Facility Administrator can record final decisions.")
st.sidebar.caption("Predictions are decision-support outputs, not automatic authorisation to reject a "
                   "reservation or deny student services.")


@st.cache_data
def cached_clean(raw):
    return clean_dataset(raw)


def active_clean_data():
    """Cleaned dataset currently loaded (bundled sample until the user loads another)."""
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = pd.read_csv(SAMPLE_PATH)
        st.session_state["source_name"] = "sample_facility_reservations.csv (synthetic)"
    clean, _ = cached_clean(st.session_state["raw_df"])
    return clean


try:
    meta = load_metadata()
    midterm = load_midterm_results()
except FileNotFoundError as e:
    st.error(f"A required file is missing: {e}. Run `python train_model.py` to create the model files.")
    st.stop()

st.title("Campus Facility Reservation Monitoring")

if menu == "Home":
    st.markdown("""
**Goal:** predict whether a facility reservation will end in a **cancellation or no-show**, so the
Facility Administrator can follow up *before* the slot is wasted.

This app is the Streamlit deployment of my Midterm Laboratory Examination: the same cleaning rules,
the same Decision Tree (`max_depth=4`) and the same pre-booking predictors
(`Lead_Time`, `Participants`, `Facility_Type`, `User_Type`, `Lead_Time_Category`).
""")
    c = st.columns(4)
    m = meta["metrics"]
    c[0].metric("Model", f"Decision Tree v{meta['model_version']}")
    c[1].metric("Test accuracy", f"{m['accuracy']:.2f}")
    c[2].metric("Recall (Not Completed)", f"{m['recall']:.2f}")
    c[3].metric("Training rows", meta["train_rows"] + meta["test_rows"])
    st.markdown("""
| Page | What it does |
|---|---|
| **Dataset** | Upload a CSV, check the schema, see null / duplicate / invalid counts and the cleaned data |
| **Data Analysis** | Descriptive statistics, class distribution and interactive charts |
| **Prediction** | Enter a reservation, get the predicted outcome, probability, risk level and follow-up |
| **Dashboard** | KPIs, risk / review queue, interpretation and the administrator decision log |
| **System Design** | Requirements, actors, updated use case diagram and decision workflow |
| **Midterm Work** | The original midterm notebook, results and figures |
""")
    if meta.get("is_sample_data"):
        st.warning("Demo mode: the model and bundled data are a synthetic stand-in built from the midterm's recorded "
                   "statistics. Replace them with the original midterm CSV (see README) for final submission.")
    st.info("Predictions support, and never replace, a human decision.")

elif menu == "Dataset":
    st.header("1. Dataset Upload and Validation")
    render_dataset_page()

elif menu == "Data Analysis":
    st.header("2. Exploratory Data Analysis")
    st.caption(f"Active dataset: {st.session_state.get('source_name', 'bundled sample')}")
    render_eda(active_clean_data())

elif menu == "Prediction":
    st.header("3. Predictive Modeling")
    render_prediction(meta, midterm)

elif menu == "Dashboard":
    st.header("4. Decision-Support Dashboard")
    st.caption(f"Active dataset: {st.session_state.get('source_name', 'bundled sample')}  |  Role: {role}")
    try:
        render_dashboard(active_clean_data(), load_model(), meta, role)
    except Exception as e:
        st.error(f"The dashboard could not be built: {e}")

elif menu == "System Design":
    st.header("5. SAD Integration")
    render_system_design()

else:
    st.header("Midterm Work")
    render_midterm(midterm, meta)
