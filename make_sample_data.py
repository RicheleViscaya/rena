"""Generate the SYNTHETIC stand-in dataset (data/sample_facility_reservations.csv).

It mirrors the schema and the summary statistics printed in the midterm notebook
(510 rows, 10 duplicate rows, 50 invalid/missing records, 450 usable rows,
183 'Not Completed'). It is NOT the original midterm CSV. Replace it with your
real facility_reservations.csv and re-run train_model.py to use your own data.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500
facility = rng.choice(["Lab", "Sports Hall", "Auditorium", "Conference Room"], N, p=[.38, .25, .20, .17])
user = rng.choice(["Student", "Faculty", "External"], N, p=[.50, .30, .20])
participants = rng.choice([10, 25, 50, 100], N, p=[.28, .34, .28, .10]).astype(float)
lead = rng.choice([0, 4, 10, 20], N, p=[.34, .30, .26, .10]).astype(int)

# invalid / missing records: 20 missing + 15 negative + 15 zero participants = 50
idx = rng.permutation(N)
bad = idx[:50]
participants[bad[:20]] = np.nan
participants[bad[20:35]] = -5
participants[bad[35:]] = 0
good = np.setdiff1d(np.arange(N), bad)

# weak, realistic signal: last-minute and student bookings fail slightly more often
p_fail = 0.36 + 0.10 * (lead <= 2) + 0.05 * (user == "Student") - 0.07 * (user == "External") + 0.04 * (participants == 100)
fail = rng.random(N) < p_fail
# force exactly 183 'Not Completed' among the 450 usable rows
cur = fail[good].sum()
if cur > 183:
    flip = rng.choice(good[fail[good]], cur - 183, replace=False); fail[flip] = False
elif cur < 183:
    flip = rng.choice(good[~fail[good]], 183 - cur, replace=False); fail[flip] = True
status = np.where(fail, rng.choice(["Cancelled", "No-Show"], N, p=[.6, .4]), "Used")
status[bad[:0]] = status[bad[:0]]

attend = np.where(status == "Used",
                  np.round(np.nan_to_num(participants, nan=10) * rng.choice([0, .4, .9, 1.0], N, p=[.05, .25, .5, .2])), np.nan)
cancel_days = np.where(status == "Cancelled", rng.integers(0, 6, N), np.nan)

df = pd.DataFrame({
    "Reservation_ID": [f"RES-{1000 + i}" for i in range(N)],
    "Facility_Type": facility, "User_Type": user,
    "Participants": participants, "Lead_Time": lead, "Usage_Status": status,
    "Actual_Attendance": attend, "Cancellation_Days_Before": cancel_days})
df.loc[rng.choice(good, 12, replace=False), "Lead_Time"] = -2          # invalid lead times
dups = df.loc[rng.choice(good, 10, replace=False)]                    # 10 exact duplicate rows
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/sample_facility_reservations.csv", index=False)
print(df.shape)
