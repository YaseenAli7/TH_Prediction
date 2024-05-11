import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load the model
with open('tuned_TH_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load bootstrap predictions
bootstrap_predictions_df = pd.read_csv('bootstrap_predictions.csv')

# Define the page layout
st.title("Therapeutic Hypothermia Outcome Prediction")

# Input fields
GA = st.number_input("Gestational Age (weeks)", min_value=0.0, value=40.0, step=1.0)
creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=1.9, step=0.1)
creatinine_unit = st.radio("Creatinine Unit", ("mg/dL", "µmol/L"))
PNA = st.number_input("Postnatal Age (days)", min_value=0, value=3, step=1)
BW = st.number_input("Body Weight (grams)", min_value=0, value=1800, step=10)

if creatinine_unit == 'µmol/L':
    creatinine /= 88.4  # Convert from µmol/L to mg/dL

# Calculate interaction terms
GA_BW_interaction = GA * BW
GA_Creatinine_interaction = GA * creatinine
PNA_Creatinine_interaction = PNA * creatinine

input_df = pd.DataFrame({
    'GA (weeks)': [GA],
    'BW (grams)': [BW],
    'PNA (days)': [PNA],
    'creatinine (mg/dL)': [creatinine],
    'GA_BW_interaction': [GA_BW_interaction],
    'GA_Creatinine_interaction': [GA_Creatinine_interaction],
    'PNA_Creatinine_interaction': [PNA_Creatinine_interaction]
})

# Descriptive labels and graph labels
labels = [
    "This infant may require therapeutic hypothermia for the best possible outcome and is likely to survive it without issue.",
    "If therapeutic hypothermia is induced upon this infant, survival of the infant is likely, although there is risk of acute kidney injury developing.",
    "Fatality may result during therapeutic hypothermia treatment of this infant.",
    "There is a risk of acute kidney injury as well as death while this infant undergoes therapeutic hypothermia treatment.",
    "This patient seems to be stable and therapeutic hypothermia may not be necessary at all."
]

graph_label = [
    'TH-treated NE neonate surviving without AKI',
    'TH-treated NE neonate surviving with AKI',
    'TH-treated NE neonate dies without AKI',
    'TH-treated NE neonate dies with AKI',
    'Hospitalized (Control neonate, non-NE)'
]

if st.button("Predict"):
    # Model prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df).flatten()
    
    # Calculate mean and 95% CI of prediction probabilities for the specific outcome
    predicted_probs = bootstrap_predictions_df.iloc[:, prediction]
    mean_prob = np.mean(predicted_probs)
    std_prob = np.std(predicted_probs)
    n = len(predicted_probs)
    ci_lower = mean_prob - 1.96 * (std_prob / np.sqrt(n))
    ci_upper = mean_prob + 1.96 * (std_prob / np.sqrt(n))
    
    # Odds of Death given AKI
    odds_death_given_AKI = prediction_proba[3] / prediction_proba[1] if prediction_proba[1] > 0 else float('inf')

    st.subheader(f"Prediction: {labels[prediction]}")
    st.text(f"95% Confidence Interval for Outcome: ({ci_lower:.2f}, {ci_upper:.2f})")
    st.text(f"Odds of Death given AKI: {odds_death_given_AKI:.2f}")

    # Confidence chart without legend
    fig = px.bar(
        x=graph_label,
        y=prediction_proba,
        title="Prediction Confidence Levels",
        labels={'x': 'Outcome', 'y': 'Probability'},
        color=graph_label,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(showlegend=False, height=700, width=750, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig)
