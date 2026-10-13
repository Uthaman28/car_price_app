import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('car_model.pkl', 'rb'))

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the car details below to predict its estimated price!")

# Create form for user inputs
with st.form("car_form"):
    year = st.number_input("Car Manufacturing Year:", min_value=2000, max_value=2026, value=2018)
    kms = st.number_input("Kilometers Driven:", min_value=0, max_value=500000, value=30000)
    owner = st.selectbox("Owner Type:", [1, 2, 3])
    present_price = st.number_input("Present Showroom Price (Lakhs):", min_value=0.1, max_value=100.0, value=8.0)
    
    fuel_type = st.selectbox("Fuel Type:", ["Petrol", "Diesel", "Other"])
    transmission = st.selectbox("Transmission:", ["Manual", "Automatic"])
    
    submit_button = st.form_submit_button(label="Predict Price")

# Prediction logic
if submit_button:
    # Encoding for Fuel Type and Transmission
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    trans_manual = 1 if transmission == "Manual" else 0
    
    # Prepare feature array
    features = np.array([[year, kms, owner, present_price, fuel_diesel, trans_manual]])
    
    # Make prediction
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ Estimated Car Price: **₹ {output} Lakhs**")
