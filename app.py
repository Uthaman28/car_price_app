import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load Model & Dataset
model = pickle.load(open('car_model.pkl', 'rb'))
df = pd.read_csv('cardekho_dataset.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

st.title("🚗 CarDekho Accurate Price Prediction App")
st.write("Enter your car details below to get the exact estimated market price in Indian Rupees (INR).")

with st.form("car_form"):
    brand_list = sorted(df['brand'].dropna().unique())
    selected_brand = st.selectbox("Select Brand:", brand_list)
    
    # Filter models based on selected brand
    filtered_models = sorted(df[df['brand'] == selected_brand]['model'].dropna().unique())
    selected_model = st.selectbox("Select Model:", filtered_models)
    
    vehicle_age = st.number_input("Vehicle Age (Years):", min_value=0, max_value=25, value=5)
    km_driven = st.number_input("KM Driven:", min_value=100, max_value=500000, value=40000)
    
    seller_type = st.selectbox("Seller Type:", sorted(df['seller_type'].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type:", sorted(df['fuel_type'].dropna().unique()))
    transmission_type = st.selectbox("Transmission Type:", sorted(df['transmission_type'].dropna().unique()))
    
    mileage = st.number_input("Mileage (km/l or km/kg):", min_value=5.0, max_value=50.0, value=18.5)
    engine = st.number_input("Engine Capacity (CC):", min_value=600, max_value=5000, value=1197)
    max_power = st.number_input("Max Power (bhp):", min_value=30.0, max_value=400.0, value=82.0)
    seats = st.selectbox("Seats:", [2, 4, 5, 6, 7, 8])
    
    car_name = selected_brand + " " + selected_model
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    input_data = pd.DataFrame([[
        car_name, selected_brand, selected_model, vehicle_age, 
        km_driven, seller_type, fuel_type, transmission_type, 
        mileage, engine, max_power, seats
    ]], columns=[
        'car_name', 'brand', 'model', 'vehicle_age', 
        'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 
        'mileage', 'engine', 'max_power', 'seats'
    ])
    
    # Label encoding matching dataset categories
    for col in ['car_name', 'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_data[col] = input_data[col].astype(str)
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    prediction = model.predict(input_data)[0]
    prediction_inr = round(prediction, 2)
    
    st.success(f"🏷️ Estimated Car Price: **₹ {prediction_inr:,.2f}**")
