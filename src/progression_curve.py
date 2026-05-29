import matplotlib.pyplot as plt

# =========================
# Simulated Predictions
# =========================

predictions = [

    'Open',
    'Open',
    'no_yawn',
    'Open',

    'yawn',
    'yawn',

    'Open',
    'yawn',

    'Closed',
    'Closed',
    'Closed',

    'Open',
    'no_yawn'
]

# =========================
# Convert to Fatigue Scores
# =========================

fatigue_scores = []

for prediction in predictions:

    if prediction in ['Open', 'no_yawn']:

        fatigue_scores.append(0)

    elif prediction == 'yawn':

        fatigue_scores.append(1)

    elif prediction == 'Closed':

        fatigue_scores.append(2)

# =========================
# Time Intervals
# =========================

time_intervals = list(
    range(1, len(fatigue_scores) + 1)
)

# =========================
# Plot Fatigue Curve
# =========================

plt.figure(figsize=(12,5))

plt.plot(
    time_intervals,
    fatigue_scores,
    marker='o'
)

plt.yticks(
    [0,1,2],
    ['Alert',
     'Mild Fatigue',
     'Severe Fatigue']
)

plt.xlabel("Time Interval")
plt.ylabel("Fatigue Level")

plt.title(
    "Driver Fatigue Progression Curve"
)

plt.grid(True)

# Save Graph
plt.savefig(
    "outputs/fatigue_progression_curve.png"
)

plt.show()

# =========================
# Transition Detection
# =========================

print("\nFatigue Transition Analysis:\n")

for i, score in enumerate(fatigue_scores):

    if score == 0:

        state = "Alert"

    elif score == 1:

        state = "Mild Fatigue"

    else:

        state = "Severe Fatigue"

    print(
        f"Time {i+1} ---> {state}"
    )