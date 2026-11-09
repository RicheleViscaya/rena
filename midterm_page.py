"""Midterm Work page - makes the submitted midterm notebook and results accessible inside the app."""
import base64
import json

import pandas as pd
import streamlit as st

NOTEBOOK = "midterm/Midterm_Notebook.ipynb"
FIGURES = [
    ("assets/midterm_cell3_fig5.png", "Part III - Exploratory data analysis (4 charts)"),
    ("assets/midterm_cell4_fig3.png", "Task 7 - Decision Tree structure"),
    ("assets/midterm_cell6_fig0.png", "Task 10 - Original use case diagram"),
    ("assets/midterm_cell7_fig1.png", "Task 11 - Administrator dashboard wireframe"),
]
CELL_TITLES = [
    "Task 1 - Load and inspect the dataset",
    "Setup, data dictionary and data-quality audit (Parts I-II)",
    "Tasks 4-5 - Cleaning and feature engineering",
    "Part III - Exploratory data analysis",
    "Part IV - Decision Tree model and evaluation (Tasks 7-8)",
    "Part V - Requirements and use case (Tasks 9-10)",
    "Task 10 - Use case diagram (matplotlib)",
    "Part VI - Decision-support dashboard wireframe (Tasks 11-12)",
]


@st.cache_data
def _load_notebook():
    with open(NOTEBOOK, encoding="utf-8") as f:
        return json.load(f)


def _show_outputs(outputs):
    for o in outputs:
        kind = o.get("output_type")
        if kind == "stream" and o.get("name") == "stdout":
            st.text("".join(o["text"]))
        elif kind == "display_data" or kind == "execute_result":
            data = o.get("data", {})
            if "image/png" in data:
                st.image(base64.b64decode(data["image/png"]))
            elif "text/plain" in data:
                txt = "".join(data["text/plain"])
                if not txt.startswith("<Figure"):
                    st.text(txt)


def render_midterm(midterm, meta):
    st.write("This page keeps your submitted **Midterm Laboratory Examination** inside the deployed app, "
             "so the app and the notebook can be read side by side.")
    tabs = st.tabs(["Results summary", "Original notebook", "Figures", "Download"])

    with tabs[0]:
        mm = midterm["metrics"]
        ds = midterm["dataset"]
        st.markdown("**Problem:** predict whether a campus facility reservation will be **Completed** or "
                    "**Not Completed** (cancelled or no-show) using only information known at booking time.")
        c = st.columns(4)
        c[0].metric("Raw rows", ds["raw_rows"])
        c[1].metric("Duplicates removed", ds["duplicates_removed"])
        c[2].metric("Invalid / missing dropped", ds["invalid_or_missing_dropped"])
        c[3].metric("Clean rows", ds["final_rows"])
        st.markdown(f"**Predictors:** {', '.join(midterm['predictors'])}  \n**Model:** {midterm['algorithm']}")
        c = st.columns(4)
        c[0].metric("Accuracy", mm["accuracy"])
        c[1].metric("Precision", mm["precision"])
        c[2].metric("Recall", mm["recall"])
        c[3].metric("F1", mm["f1"])
        cm = midterm["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm["matrix"], index=[f"Actual: {l}" for l in cm["labels"]],
                                  columns=[f"Pred: {l}" for l in cm["labels"]]))
        st.markdown("**Top predictors (midterm):** " + ", ".join(f"{n} ({v})" for n, v in midterm["top_predictors"]))
        st.markdown("**EDA findings**")
        for f in midterm["eda_findings"]:
            st.markdown("- " + f)
        st.markdown(f"**Limitation:** {midterm['limitation']}  \n**Improvement:** {midterm['improvement']}")
        st.markdown("**Midterm vs deployed model**")
        dm = meta["metrics"]
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Midterm notebook": [mm["accuracy"], mm["precision"], mm["recall"], mm["f1"]],
            f"Deployed model (v{meta['model_version']}, {meta['training_file']})":
                [round(dm["accuracy"], 4), round(dm["precision"], 4), round(dm["recall"], 4), round(dm["f1"], 4)],
        }), hide_index=True)
        if meta.get("is_sample_data"):
            st.warning("The deployed model currently uses the synthetic stand-in data, so its numbers differ from the "
                       "midterm. Re-run `train_model.py` on your original CSV; the two columns should then match closely.")

    with tabs[1]:
        nb = _load_notebook()
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            title = CELL_TITLES[i] if i < len(CELL_TITLES) else f"Cell {i + 1}"
            with st.expander(f"Cell {i + 1}: {title}"):
                st.code("".join(cell["source"]), language="python")
                if cell.get("outputs"):
                    st.markdown("**Saved output**")
                    _show_outputs(cell["outputs"])

    with tabs[2]:
        for path, cap in FIGURES:
            try:
                st.image(path, caption=cap)
            except Exception:
                st.caption(f"{cap}: image not found at {path}")

    with tabs[3]:
        with open(NOTEBOOK, "rb") as f:
            st.download_button("Download original midterm notebook (.ipynb)", f.read(),
                               "Midterm_Notebook.ipynb", "application/x-ipynb+json")
