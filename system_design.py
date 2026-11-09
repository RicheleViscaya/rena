"""Module 5 - SAD integration: requirements, actors, use case diagram, decision workflow."""
import pandas as pd
import streamlit as st

FUNCTIONAL = [
    ("FR-DM-01", "Short lead-time reservations (0-2 days) show the highest cancellation / no-show rate (EDA, Part III).",
     "The system shall evaluate cancellation risk at submission time with the Decision Tree model and flag high-risk reservations.",
     "Prediction page, Dashboard risk bands"),
    ("FR-DM-02", "High-risk reservations need a human touchpoint before the event.",
     "The system shall present a re-confirmation follow-up recommendation for reservations flagged as high-risk.",
     "Prediction result, Dashboard review queue"),
    ("FR-DM-03", "Invalid participant counts and out-of-range lead times appeared in the raw data (Part II).",
     "The system shall validate participant and lead-time inputs and reject zero, negative or out-of-range values.",
     "Prediction input validation, Dataset validation"),
    ("FR-DM-04", "Model recall is low, so flagged cases need administrator judgement (Part IV).",
     "The system shall let the Facility Administrator record a final decision and note for each flagged reservation and export the log.",
     "Dashboard decision log"),
    ("FR-DM-05", "Data arrives as batch CSV exports from the reservation system.",
     "The system shall accept a CSV upload, check its schema, and report null, duplicate and invalid-value counts before analysis.",
     "Dataset page"),
]
NON_FUNCTIONAL = [
    ("NFR-01", "Transparency", "Every prediction shall display the model version, timestamp, probability and known limitations."),
    ("NFR-02", "Performance", "The trained model shall be loaded once per server session (cached) so a prediction returns in under 2 seconds without retraining."),
    ("NFR-03", "Privacy and security", "The app shall use only synthetic or authorised data, no credentials in the repository, and restrict decision recording to the Facility Administrator role."),
]
ACTORS = [
    ("Facility User (Student / Faculty / External)", "Submits reservations; may view the risk result and confirmation prompt for their own booking.", "View own result only", "No"),
    ("Facility Administrator", "Reviews the risk queue, contacts requesters, decides the final action.", "Yes - all results and dashboard", "Yes"),
    ("Predictive Model (system actor)", "Scores each reservation from pre-booking fields.", "Produces results", "No"),
]
WORKFLOW_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef5ff", fontname="Helvetica"];
  A [label="Reservation data\\n(CSV / system export)"];
  B [label="Validate schema\\n& clean data"];
  C [label="Feature engineering\\n(Lead_Time_Category)"];
  D [label="Saved pipeline\\npredicts risk"];
  E [label="Risk band\\nLow / Medium / High", fillcolor="#fff4d6"];
  F [label="Review queue\\n(Facility Administrator)", fillcolor="#e6f7e6"];
  G [label="Human decision\\n+ note recorded", fillcolor="#e6f7e6"];
  H [label="Follow-up action\\n(reminder, re-confirm, waitlist)"];
  A -> B -> C -> D -> E -> F -> G -> H;
  G -> A [style=dashed, label="new data / feedback"];
}
"""


def render_system_design():
    st.subheader("Functional requirements (evidence-based)")
    st.dataframe(pd.DataFrame(FUNCTIONAL, columns=["ID", "Finding / evidence", "Requirement", "Implemented in this app"]),
                 hide_index=True)
    st.subheader("Non-functional requirements")
    st.dataframe(pd.DataFrame(NON_FUNCTIONAL, columns=["ID", "Category", "Requirement"]),
                 hide_index=True)
    st.subheader("Actors and permissions")
    st.dataframe(pd.DataFrame(ACTORS, columns=["Actor", "Responsibility", "Can view results", "Records final decision"]),
                 hide_index=True)
    st.subheader("Updated use case diagram")
    st.image("assets/use_case_diagram.png",
             caption="Green ellipses are new since the midterm: dataset upload/validation and administrator review.")
    st.subheader("Analytical process / decision workflow")
    st.graphviz_chart(WORKFLOW_DOT)
    st.info("Predictions are decision-support outputs. They are not an automatic authorisation to reject a "
            "reservation or deny student services; the Facility Administrator makes and records the final decision.")
