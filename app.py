import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open('car_model.pkl', 'rb'))

st.title("🚗 Advanced Used Car Price Prediction App")
st.write("Enter the car details below to get an accurate estimated price!")

with st.form("car_form"):
    # Dropdowns based on your dataset
    make = st.selectbox("Car Brand (Make):", ['Honda', 'Ford', 'BMW', 'Audi', 'Toyota'])
    model_name = st.selectbox("Car Model:", ['Model A', 'Model B', 'Model C', 'Model D', 'Model E'])
    
    year = st.number_input("Car Manufacturing Year:", min_value=2000, max_value=2026, value=2015)
    engine_size = st.number_input("Engine Size (L):", min_value=1.0, max_value=5.0, value=2.5)
    mileage = st.number_input("Mileage (Driven KM):", min_value=0, max_value=300000, value=50000)
    
    fuel_type = st.selectbox("Fuel Type:", ['Petrol', 'Diesel', 'Electric'])
    transmission = st.selectbox("Transmission:", ['Manual', 'Automatic'])
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # Creating a dummy dataframe matching training columns to handle encoding cleanly
    input_data = pd.DataFrame([[make, model_name, year, engine_size, mileage, fuel_type, transmission]],
                              columns=['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission'])
    
    # We need dummy columns matching the trained model's columns structure
    # A simple way in Streamlit is to load base columns or structure manual encoding:
    # Let's map manually based on standard pandas get_dummies logic used during training:
    
    # Simulating the encoded array columns mapping
    # To keep it robust with get_dummies:
    full_df = pd.read_csv('Car_Price_Prediction.csv')
    full_df_encoded = pd.get_dummies(full_df.drop('Price', axis=1), drop_first=True)
    
    input_encoded = pd.concat([input_data, full_df_encoded], ignore_index=True)
    input_encoded = pd.get_dummies(input_encoded, columns=['Make', 'Model', 'Fuel Type', 'Transmission'], drop_first=True)
    
    # Take only the first row (our input row)
    final_input = input_encoded.iloc[[0]].fillna(0)
    
    # Prediction
    prediction = model.predict(final_input)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ Estimated Car Price: **₹ {output:,}**")
