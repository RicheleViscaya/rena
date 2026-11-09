# Campus Facility Reservation Monitoring - Streamlit deployment

Deployment of the **Midterm Laboratory Examination** (Section A: predict cancellation / no-show).
Course: Systems Analysis and Design, BS Information Systems.

* **Target:** `Reservation_Outcome` (Completed / Not Completed)
* **Predictors (pre-booking only):** `Lead_Time`, `Participants`, `Facility_Type`, `User_Type`, `Lead_Time_Category`
* **Model:** `DecisionTreeClassifier(max_depth=4, random_state=42)`, 80/20 stratified split, saved as one
  preprocessing + model pipeline (`models/trained_model.joblib`). It is loaded once and never retrained on refresh.
* **Leakage fields excluded:** `Usage_Status`, `Actual_Attendance`, `Attendance_Ratio`, `Cancellation_Days_Before`

## Repository layout
```
app.py                       navigation + page routing
requirements.txt
modules/data_processing.py   Module 1  upload, schema checks, cleaning, features
modules/exploratory_analysis.py  Module 2  statistics + interactive charts
modules/prediction.py        Module 3  model loading, inference, metrics, limitations
modules/dashboard.py         Module 4  KPIs, review queue, decision log
modules/system_design.py     Module 5  requirements, actors, diagram, workflow
modules/midterm_page.py      the original midterm notebook, results and figures inside the app
models/trained_model.joblib  saved pipeline          models/model_metadata.json  metrics + categories
models/midterm_results.json  results copied from the submitted notebook
assets/use_case_diagram.png  updated use case diagram
midterm/Midterm_Notebook.ipynb  original midterm notebook
data/sample_facility_reservations.csv  SYNTHETIC stand-in data
train_model.py               rebuilds the saved pipeline from a CSV
```

## Use your ORIGINAL midterm data (important)
The bundled data/model are a synthetic stand-in built from the statistics printed in the midterm notebook.
To use your real file:
```
cp /path/to/facility_reservations.csv data/facility_reservations.csv
python train_model.py            # writes models/trained_model.joblib, model_metadata.json, assets/decision_tree.png
```
Then commit the new files. The app reads categories and metrics from `model_metadata.json`, so it adapts automatically.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push to GitHub, open https://share.streamlit.io, **Create app**, choose repo + branch, main file `app.py`, Deploy.
`scikit-learn` is pinned to the version that created the joblib file; if you retrain in another environment, update the pin.

Predictions are decision-support outputs, not automatic authorisation to reject a reservation or deny student services.
