"""Model loading and inference. The model is trained once (train_model.py) and loaded here."""
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from modules.data_processing import lead_time_category

MODEL_PATH = "models/trained_model.joblib"
META_PATH = "models/model_metadata.json"
MIDTERM_PATH = "models/midterm_results.json"

LIMITATIONS = [
    "Decision-support only: a prediction never authorises rejecting a reservation or denying student services.",
    "Small training set (a few hundred reservations) and a shallow Decision Tree (max_depth=4): probabilities are coarse and can be overfitted.",
    "Only five pre-booking inputs are used. Exam periods, weather, events and a user's cancellation history are not captured.",
    "Categories never seen in training are ignored, so predictions for new facility or user types are unreliable.",
    "The model reflects past behaviour and may encode bias against user groups; a human administrator must review every flagged case.",
]


@st.cache_resource
def load_model():
    """Load the saved preprocessing + model pipeline once per server process."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_midterm_results():
    with open(MIDTERM_PATH) as f:
        return json.load(f)


def build_input_row(lead_time, participants, facility_type, user_type):
    """One-row DataFrame with the exact training feature names and order."""
    return pd.DataFrame([{
        "Lead_Time": float(lead_time),
        "Participants": float(participants),
        "Facility_Type": facility_type,
        "User_Type": user_type,
        "Lead_Time_Category": lead_time_category(lead_time),
    }])


def validate_inputs(lead_time, participants, facility_type, user_type, meta):
    errors = []
    lo, hi = meta["ranges"]["Lead_Time"]
    if lead_time is None or not (lo <= lead_time <= hi):
        errors.append(f"Lead time must be between {lo} and {hi} days.")
    lo, hi = meta["ranges"]["Participants"]
    if participants is None or not (lo <= participants <= hi):
        errors.append(f"Participants must be between {lo} and {hi}.")
    if facility_type not in meta["categories"]["Facility_Type"]:
        errors.append("Facility type is not one the model was trained on.")
    if user_type not in meta["categories"]["User_Type"]:
        errors.append("User type is not one the model was trained on.")
    return errors


def risk_band(p_not_completed, low=0.35, high=0.55):
    if p_not_completed >= high:
        return "High"
    if p_not_completed >= low:
        return "Medium"
    return "Low"


FOLLOW_UP = {
    "High": "Administrator to contact the requester, request re-confirmation 24h before the event, and keep a waitlist option ready.",
    "Medium": "Send an automated reminder and re-confirmation prompt; review again if no reply.",
    "Low": "No action needed beyond the standard confirmation message.",
}


def predict_one(row, model, meta, low=0.35, high=0.55):
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p_nc = float(proba[classes.index(meta["positive_label"])])
    label = classes[int(proba.argmax())]
    return {
        "label": label,
        "p_not_completed": p_nc,
        "p_completed": 1 - p_nc,
        "band": risk_band(p_nc, low, high),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def score_dataframe(df, model, meta, low=0.35, high=0.55):
    """Score a cleaned dataframe using ONLY pre-booking features (no leakage fields)."""
    X = df[meta["features"]]
    classes = list(model.classes_)
    p = model.predict_proba(X)[:, classes.index(meta["positive_label"])]
    out = df.copy()
    out["Risk_Probability"] = p
    out["Risk_Level"] = [risk_band(v, low, high) for v in p]
    out["Predicted_Outcome"] = model.predict(X)
    out["Recommended_Follow_Up"] = out["Risk_Level"].map(FOLLOW_UP)
    return out


SAMPLE_SCENARIOS = [
    ("Last-minute lab booking by a student", 0, 25, "Lab", "Student"),
    ("Early auditorium booking by faculty", 20, 100, "Auditorium", "Faculty"),
    ("Same-week sports hall, external group", 4, 50, "Sports Hall", "External"),
    ("Small conference room, faculty, 10 days ahead", 10, 10, "Conference Room", "Faculty"),
    ("Large last-minute sports hall booking", 1, 100, "Sports Hall", "Student"),
]


# ----------------------------------------------------------------------------------
# Streamlit page (Module 3)
# ----------------------------------------------------------------------------------
def _metrics_block(meta, midterm):
    import plotly.express as px
    m = meta["metrics"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    c[1].metric("Precision (Not Completed)", f"{m['precision']:.3f}")
    c[2].metric("Recall (Not Completed)", f"{m['recall']:.3f}")
    c[3].metric("F1 (Not Completed)", f"{m['f1']:.3f}")
    st.caption(f"Held-out test set: {meta['test_rows']} reservations. "
               f"Always predicting the majority class would score {meta['majority_baseline']:.3f} accuracy.")
    a, b = st.columns(2)
    cm = meta["confusion_matrix"]
    fig = px.imshow(cm["matrix"], text_auto=True, x=[f"Pred: {l}" for l in cm["labels"]],
                    y=[f"Actual: {l}" for l in cm["labels"]], color_continuous_scale="Blues",
                    title="Confusion matrix (test set)")
    a.plotly_chart(fig)
    imp = (pd.Series(meta["feature_importances"]).sort_values(ascending=False).head(6)
           .rename_axis("Feature").reset_index(name="Importance").iloc[::-1])
    b.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top predictors"))
    mm = midterm["metrics"]
    st.markdown(
        f"**Midterm notebook (as submitted):** accuracy {mm['accuracy']}, precision {mm['precision']}, "
        f"recall {mm['recall']}, F1 {mm['f1']}. Compare these with the model currently deployed above "
        "(see the Midterm Work page for the full record).")


def render_prediction(meta, midterm):
    try:
        model = load_model()
    except Exception as e:  # missing or incompatible model file
        st.error("The saved model could not be loaded. Check that models/trained_model.joblib exists and "
                 f"that scikit-learn matches the training version ({meta.get('sklearn_version')}). Details: {e}")
        return

    st.info(f"Model **v{meta['model_version']}** - {meta['algorithm']} - trained {meta['trained_at']} "
            f"on `{meta['training_file']}` (scikit-learn {meta['sklearn_version']}). "
            "The model is loaded from disk once and is never retrained on page refresh.")
    if meta.get("is_sample_data"):
        st.warning("This deployed model was trained on the **synthetic stand-in dataset**, not the original midterm CSV. "
                   "Re-run `train_model.py` with your real file to use your own data.")

    tab1, tab2, tab3 = st.tabs(["Predict a reservation", "Five sample predictions", "Model metrics & limitations"])

    with tab1:
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            facility = c1.selectbox("Facility type", meta["categories"]["Facility_Type"])
            user = c2.selectbox("User type", meta["categories"]["User_Type"])
            participants = c1.number_input("Expected participants", min_value=-10, max_value=1000, value=25, step=1,
                                           help="Valid range 1-500. Out-of-range values are rejected.")
            lead = c2.number_input("Lead time (days between booking and event)", min_value=-30, max_value=1000,
                                   value=3, step=1, help="Valid range 0-365.")
            go = st.form_submit_button("Predict")
        if go:
            errs = validate_inputs(lead, participants, facility, user, meta)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                row = build_input_row(lead, participants, facility, user)
                res = predict_one(row, model, meta)
                st.session_state["last_prediction"] = res
                color = {"High": "red", "Medium": "orange", "Low": "green"}[res["band"]]
                st.markdown(f"### Predicted outcome: **{res['label']}**  -  :{color}[{res['band']} risk]")
                a, b = st.columns(2)
                a.metric("Probability: Not Completed", f"{res['p_not_completed'] * 100:.1f}%")
                b.metric("Probability: Completed", f"{res['p_completed'] * 100:.1f}%")
                st.markdown(f"**Recommended human follow-up:** {FOLLOW_UP[res['band']]}")
                st.caption(f"Model version {meta['model_version']} - prediction timestamp {res['timestamp']} - "
                           "human review required before any action.")
                st.markdown("**Model input row**")
                st.dataframe(row, hide_index=True)

    with tab2:
        st.write("Five ready-made scenarios, handy for screenshots and for checking consistency with the notebook.")
        if st.button("Run the five sample predictions"):
            rows = []
            for name, lt, pt, fac, usr in SAMPLE_SCENARIOS:
                if fac not in meta["categories"]["Facility_Type"] or usr not in meta["categories"]["User_Type"]:
                    continue
                r = predict_one(build_input_row(lt, pt, fac, usr), model, meta)
                rows.append({"Scenario": name, "Facility": fac, "User": usr, "Participants": pt, "Lead time": lt,
                             "Prediction": r["label"], "P(Not Completed) %": round(r["p_not_completed"] * 100, 1),
                             "Risk": r["band"], "Timestamp": r["timestamp"], "Model": f"v{meta['model_version']}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    with tab3:
        _metrics_block(meta, midterm)
        st.subheader("Limitations")
        for item in LIMITATIONS:
            st.markdown("- " + item)
