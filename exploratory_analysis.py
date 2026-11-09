"""Module 2 - Exploratory Data Analysis (statistics, class distribution, interactive Plotly charts)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import LEAD_ORDER

COLORS = {"Completed": "#2ca02c", "Not Completed": "#d62728"}


def render_eda(df):
    if df is None or df.empty:
        st.warning("No cleaned data available. Load a dataset on the Dataset page first.")
        return

    # ---- interactive filters ------------------------------------------------
    with st.expander("Filters (apply to every chart below)", expanded=False):
        c1, c2 = st.columns(2)
        fac = c1.multiselect("Facility type", sorted(df["Facility_Type"].dropna().unique()),
                             default=sorted(df["Facility_Type"].dropna().unique()))
        usr = c2.multiselect("User type", sorted(df["User_Type"].dropna().unique()),
                             default=sorted(df["User_Type"].dropna().unique()))
    d = df[df["Facility_Type"].isin(fac) & df["User_Type"].isin(usr)]
    if d.empty:
        st.warning("The selected filters leave no rows. Widen the filters.")
        return

    # ---- descriptive statistics --------------------------------------------
    st.subheader("1. Descriptive statistics")
    num_cols = [c for c in ["Lead_Time", "Participants", "Actual_Attendance",
                            "Cancellation_Days_Before", "Attendance_Ratio"] if c in d.columns]
    c1, c2 = st.columns(2)
    c1.markdown("**Numerical variables**")
    c1.dataframe(d[num_cols].describe().T.round(2))
    c2.markdown("**Categorical variables**")
    c2.dataframe(d[["Facility_Type", "User_Type", "Usage_Status", "Lead_Time_Category",
                    "Reservation_Outcome"]].astype(str).describe().T)

    # ---- class distribution -------------------------------------------------
    st.subheader("2. Class distribution (target: Reservation_Outcome)")
    dist = d["Reservation_Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Count")
    dist["Percent"] = (dist["Count"] / dist["Count"].sum() * 100).round(2)
    c1, c2 = st.columns([1, 1])
    fig = px.pie(dist, names="Outcome", values="Count", hole=0.45, color="Outcome",
                 color_discrete_map=COLORS, title="Completed vs Not Completed")
    c1.plotly_chart(fig)
    c2.dataframe(dist, hide_index=True)
    status = d["Usage_Status"].value_counts().rename_axis("Usage_Status").reset_index(name="Count")
    c2.plotly_chart(px.bar(status, x="Usage_Status", y="Count", color="Usage_Status",
                           title="Raw Usage_Status (Used / Cancelled / No-Show)"))

    # ---- interactive charts ---------------------------------------------------
    st.subheader("3. Interactive charts")
    t1, t2, t3, t4 = st.tabs(["Outcome by facility", "Lead time vs outcome",
                              "Non-completion by lead-time category", "Participants & user type"])
    with t1:
        by_fac = d.groupby(["Facility_Type", "Reservation_Outcome"]).size().reset_index(name="Count")
        st.plotly_chart(px.bar(by_fac, x="Facility_Type", y="Count", color="Reservation_Outcome",
                               barmode="group", color_discrete_map=COLORS,
                               title="Reservation outcomes by facility type"))
    with t2:
        st.plotly_chart(px.box(d, x="Reservation_Outcome", y="Lead_Time", color="Reservation_Outcome",
                               points="all", color_discrete_map=COLORS,
                               title="Lead time (days) by outcome"))
    with t3:
        cat = d.dropna(subset=["Lead_Time_Category"]).copy()
        cat["Not_Completed"] = (cat["Reservation_Outcome"] == "Not Completed").astype(int)
        rate = (cat.groupby("Lead_Time_Category")["Not_Completed"].agg(["mean", "count"])
                .reindex([c for c in LEAD_ORDER if c in set(cat["Lead_Time_Category"])]).reset_index())
        rate["Non-completion rate (%)"] = (rate["mean"] * 100).round(1)
        st.plotly_chart(px.bar(rate, x="Lead_Time_Category", y="Non-completion rate (%)", text="Non-completion rate (%)",
                               hover_data=["count"], title="Non-completion rate by lead-time category"))
    with t4:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(d, x="Participants", color="Reservation_Outcome", barmode="overlay",
                                     nbins=20, color_discrete_map=COLORS,
                                     title="Participants distribution"))
        ut = (d.groupby("User_Type")["Reservation_Outcome"].apply(lambda s: (s == "Not Completed").mean() * 100)
              .round(1).reset_index(name="Non-completion rate (%)"))
        c2.plotly_chart(px.bar(ut, x="User_Type", y="Non-completion rate (%)",
                               title="Non-completion rate by user type"))

    # ---- outcome rate tables + dynamic findings -------------------------------
    st.subheader("4. Outcome rates and findings")
    c1, c2 = st.columns(2)
    c1.markdown("**Outcome rate by user type (%)**")
    c1.dataframe((pd.crosstab(d["User_Type"], d["Reservation_Outcome"], normalize="index") * 100).round(2))
    c2.markdown("**Outcome rate by facility type (%)**")
    c2.dataframe((pd.crosstab(d["Facility_Type"], d["Reservation_Outcome"], normalize="index") * 100).round(2))

    nc = (d["Reservation_Outcome"] == "Not Completed")
    findings = [f"Overall non-completion rate is **{nc.mean() * 100:.1f}%** ({int(nc.sum())} of {len(d)} reservations)."]
    lm = d[d["Lead_Time_Category"] == "Last Minute"]
    if len(lm):
        findings.append(f"Last-minute bookings (0-2 days) have a non-completion rate of **{(lm['Reservation_Outcome'] == 'Not Completed').mean() * 100:.1f}%** (n={len(lm)}).")
    worst = (d.groupby("Facility_Type")["Reservation_Outcome"].apply(lambda s: (s == "Not Completed").mean()).sort_values(ascending=False))
    if len(worst):
        findings.append(f"Highest non-completion by facility: **{worst.index[0]}** ({worst.iloc[0] * 100:.1f}%).")
    st.markdown("**Key findings (computed from the data currently loaded)**")
    for f in findings:
        st.markdown("- " + f)
