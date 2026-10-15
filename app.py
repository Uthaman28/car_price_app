import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load Model, Scaler & Dataset
model = pickle.load(open('car_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
df = pd.read_csv('cars24data.csv')

st.title("🚗 Cars24 Price Prediction App")
st.write("Enter your car details below to get the estimated market price in Indian Rupees (INR).")

with st.form("car_form"):
    models_list = sorted(df['Model Name'].dropna().unique())
    selected_model_name = st.selectbox("Car Model Name:", models_list)
    
    manufacturing_year = st.number_input("Manufacturing Year:", min_value=2000, max_value=2026, value=2018)
    engine_capacity = st.number_input("Engine Capacity (CC):", min_value=600, max_value=5000, value=1197)
    km_driven = st.number_input("Total KM Driven:", min_value=100, max_value=500000, value=30000)
    
    spare_key = st.selectbox("Spare Key Available?", ["Yes", "No"])
    transmission = st.selectbox("Transmission Type:", sorted(df['Transmission'].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type:", sorted(df['Fuel type'].dropna().unique()))
    
    ownership = st.selectbox("Ownership (Number of Previous Owners):", [1, 2, 3, 4])
    imperfections = st.number_input("Imperfections Count:", min_value=0, max_value=50, value=2)
    repainted_parts = st.number_input("Repainted Parts Count:", min_value=0, max_value=20, value=1)
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # Input DataFrame creation
    input_data = pd.DataFrame([[
        selected_model_name, manufacturing_year, engine_capacity, 
        spare_key, transmission, km_driven, ownership, 
        fuel_type, imperfections, repainted_parts
    ]], columns=[
        'Model Name', 'Manufacturing_year', 'Engine capacity', 
        'Spare key', 'Transmission', 'KM driven', 'Ownership', 
        'Fuel type', 'Imperfections', 'Repainted Parts'
    ])
    
    # Label Encoding matching original dataset categories
    for col in ['Model Name', 'Spare key', 'Transmission', 'Fuel type']:
        le = LabelEncoder()
        le.fit(df[col].dropna().astype(str))
        input_data[col] = input_data[col].astype(str)
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Prediction (Directly in INR)
    prediction = model.predict(input_scaled)[0]
    prediction_inr = round(prediction, 2)
    
    st.success(f"🏷️ Estimated Car Price: **₹ {prediction_inr:,.2f}**")
