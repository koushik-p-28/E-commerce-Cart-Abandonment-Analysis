import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("final_model.pkl")

# App title
st.title("E-commerce Cart Abandonment Predictor")

# App description
st.markdown("This app predicts whether a customer session will result in cart abandonment or a completed purchase based on user behavior.")

# Sidebar for input
st.sidebar.header("Session Details")
pages_visited = st.sidebar.slider("Pages Visited", 1, 50, 10)
time_on_site = st.sidebar.slider("Time on Site (seconds)", 30, 1800, 300)
cart_value = st.sidebar.slider("Cart Value (USD)", 0.0, 1000.0, 150.0, step=10.0)

# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[pages_visited, time_on_site, cart_value]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    # Output
    if prediction == 1:
        st.error("Prediction: Likely to Abandon Cart")
        st.info(f"Estimated Risk of Abandonment: {proba[1]:.2%}")
    else:
        st.success("Prediction: Likely to Complete Purchase")
        st.info(f"Estimated Probability of Purchase: {proba[0]:.2%}")

