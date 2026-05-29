# =========================
# Driver Fatigue Logic
# =========================

def fatigue_level(prediction):

    if prediction in ['Open', 'no_yawn']:
        return "Alert"

    elif prediction == 'yawn':
        return "Mild Fatigue"

    elif prediction == 'Closed':
        return "Severe Fatigue"

    else:
        return "Unknown"


# =========================
# Sample Predictions
# =========================

sample_predictions = [
    'Open',
    'no_yawn',
    'yawn',
    'Closed',
    'Open',
    'Closed'
]

print("\nDriver Fatigue Detection Results:\n")

for prediction in sample_predictions:

    fatigue = fatigue_level(prediction)

    print(
        f"Prediction: {prediction} ---> Fatigue Level: {fatigue}"
    )