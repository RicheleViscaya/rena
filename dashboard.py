"""Module 4 - Decision-support dashboard: KPIs, risk/review queue, interpretation, human follow-up."""
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.prediction import FOLLOW_UP, score_dataframe

DECISIONS = ["Confirmed - no action", "Reminder sent", "Re-confirmation requested",
             "Escalated to facility manager", "Waitlist offered", "Reservation released (human decision)"]
BAND_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}


def render_dashboard(df, model, meta, role):
    if df is None or df.empty:
        st.warning("No cleaned data available. Load a dataset on the Dataset page first.")
        return

    st.caption("Risk levels come from the saved Decision Tree using pre-booking fields only "
               "(no outcome fields). They support, and never replace, a human decision.")
    c1, c2 = st.columns(2)
    high = c1.slider("High-risk threshold (probability of Not Completed)", 0.30, 0.90, 0.55, 0.05)
    low = c2.slider("Medium-risk threshold", 0.10, float(high), min(0.35, float(high)), 0.05)

    scored = score_dataframe(df, model, meta, low=low, high=high)

    # ---- KPIs ------------------------------------------------------------------
    nc_rate = (scored["Reservation_Outcome"] == "Not Completed").mean() * 100
    n_high = int((scored["Risk_Level"] == "High").sum())
    n_med = int((scored["Risk_Level"] == "Medium").sum())
    k = st.columns(5)
    k[0].metric("Reservations analysed", f"{len(scored):,}")
    k[1].metric("Observed completion rate", f"{100 - nc_rate:.1f}%")
    k[2].metric("Observed cancel / no-show rate", f"{nc_rate:.1f}%")
    k[3].metric("High-risk reservations", n_high)
    k[4].metric("Medium-risk reservations", n_med)

    # ---- utilisation / risk charts -----------------------------------------------
    st.subheader("Facility risk and utilisation")
    c1, c2 = st.columns(2)
    fac = (scored.groupby("Facility_Type")
           .agg(Reservations=("Reservation_ID", "count"),
                Observed_Not_Completed_pct=("Reservation_Outcome", lambda s: round((s == "Not Completed").mean() * 100, 1)),
                Avg_Predicted_Risk=("Risk_Probability", lambda s: round(s.mean() * 100, 1)))
           .reset_index())
    c1.plotly_chart(px.bar(fac, x="Facility_Type", y=["Observed_Not_Completed_pct", "Avg_Predicted_Risk"],
                           barmode="group", title="Observed vs predicted risk by facility (%)"))
    mix = scored.groupby(["Facility_Type", "Risk_Level"]).size().reset_index(name="Count")
    c2.plotly_chart(px.bar(mix, x="Facility_Type", y="Count", color="Risk_Level", color_discrete_map=BAND_COLORS,
                           category_orders={"Risk_Level": ["High", "Medium", "Low"]},
                           title="Risk level mix by facility"))
    st.dataframe(fac, hide_index=True)

    # ---- interpretation -----------------------------------------------------------
    st.subheader("Interpretation")
    top = fac.sort_values("Avg_Predicted_Risk", ascending=False).iloc[0]
    st.markdown(
        f"- **{n_high}** of {len(scored)} reservations ({n_high / len(scored) * 100:.1f}%) fall in the High band; "
        f"{n_med} are Medium.\n"
        f"- **{top['Facility_Type']}** has the highest average predicted risk ({top['Avg_Predicted_Risk']}%).\n"
        f"- Model recall for 'Not Completed' is {meta['metrics']['recall'] * 100:.0f}% on the held-out test set, "
        f"so many real cancellations will **not** be flagged. Treat Low as 'not flagged', not 'safe'."
    )

    # ---- review queue ---------------------------------------------------------------
    st.subheader("Risk / review queue")
    levels = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High"])
    queue = (scored[scored["Risk_Level"].isin(levels)]
             .sort_values("Risk_Probability", ascending=False)
             [["Reservation_ID", "Facility_Type", "User_Type", "Participants", "Lead_Time",
               "Risk_Probability", "Risk_Level", "Recommended_Follow_Up"]].copy())
    queue["Risk_Probability"] = (queue["Risk_Probability"] * 100).round(1)
    queue = queue.rename(columns={"Risk_Probability": "Risk %"})
    st.dataframe(queue, hide_index=True)
    st.download_button("Download review queue (CSV)", queue.to_csv(index=False).encode(),
                       "review_queue.csv", "text/csv")

    st.markdown("**Recommended human follow-up by risk level**")
    for band in ["High", "Medium", "Low"]:
        st.markdown(f"- **{band}**: {FOLLOW_UP[band]}")

    # ---- human decision log (role-gated) -----------------------------------------------
    st.subheader("Administrator review - record final decision")
    if "decision_log" not in st.session_state:
        st.session_state["decision_log"] = []
    if role != "Facility Administrator":
        st.info(f"Signed in as **{role}**. Only the **Facility Administrator** can record final decisions. "
                "Switch the role in the sidebar to try the workflow.")
    elif queue.empty:
        st.info("The review queue is empty for the selected risk levels.")
    else:
        with st.form("decision_form", clear_on_submit=True):
            rid = st.selectbox("Reservation", queue["Reservation_ID"].tolist())
            decision = st.selectbox("Final decision", DECISIONS)
            note = st.text_input("Reason / note (required)")
            submitted = st.form_submit_button("Record decision")
        if submitted:
            if not note.strip():
                st.error("A note is required so every decision is traceable.")
            else:
                row = queue[queue["Reservation_ID"] == rid].iloc[0]
                st.session_state["decision_log"].append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Reservation_ID": rid, "Model_Risk_%": row["Risk %"],
                    "Risk_Level": row["Risk_Level"], "Decision": decision,
                    "Decided_By": role, "Note": note.strip(), "Model_Version": meta["model_version"]})
                st.success(f"Decision recorded for {rid}.")
    log = pd.DataFrame(st.session_state["decision_log"])
    if not log.empty:
        st.dataframe(log, hide_index=True)
        st.download_button("Download decision log (CSV)", log.to_csv(index=False).encode(),
                           "decision_log.csv", "text/csv")
    st.caption("The decision log lives in this browser session only. Download it to keep a record.")
