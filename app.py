import streamlit as st
import pickle
import numpy as np

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the car details below to predict its estimated price!")

# Form for User Inputs
with st.form("car_form"):
    # Brand and Model Inputs
    brand = st.selectbox("Car Brand:", ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Other"])
    car_model = st.text_input("Car Model (e.g., Swift, City, Innova):", value="Swift")
    
    year = st.number_input("Car Manufacturing Year:", min_value=2000, max_value=2026, value=2018)
    kms = st.number_input("Kilometers Driven:", min_value=0, max_value=500000, value=30000)
    owner = st.selectbox("Owner Type:", [1, 2, 3])
    present_price = st.number_input("Present Showroom Price (Lakhs):", min_value=0.1, max_value=100.0, value=8.0)
    
    fuel_type = st.selectbox("Fuel Type:", ["Petrol", "Diesel", "Other"])
    transmission = st.selectbox("Transmission:", ["Manual", "Automatic"])
    
    # Car Complaints/Issues Input
    has_complaints = st.radio("Major Mechanical Issues / Past Accidents?", ["No", "Yes"])
    
    submit_button = st.form_submit_button(label="Predict Price")

# Prediction logic
if submit_button:
    # Load model
    try:
        model = pickle.load(open('car_model.pkl', 'rb'))
        
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        trans_manual = 1 if transmission == "Manual" else 0
        complaint_val = 1 if has_complaints == "Yes" else 0
        
        # Prepare feature array
        features = np.array([[year, kms, owner, present_price, fuel_diesel, trans_manual, complaint_val]])
        
        # Make prediction
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        
        # Adjust price slightly if complaints exist
        if complaint_val == 1:
            output = round(output * 0.85, 2) # Reduce 15% price for complaints
            
        st.success(f"🏷️ Estimated Price for **{brand} {car_model}**: **₹ {output} Lakhs**")
        
        if complaint_val == 1:
            st.warning("⚠️ Note: Price is lower due to reported issues/accidents.")
            
    except Exception as e:
        st.error("Please train and update the model with the new features.")

