import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load Trained Model & Dataset
model = pickle.load(open('car_model.pkl', 'rb'))
df = pd.read_csv('Car_Price_Prediction.csv')

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict the estimated market price.")

with st.form("car_form"):
    makes = sorted(df['Make'].dropna().unique())
    selected_make = st.selectbox("Car Brand (Make):", makes)
    
    filtered_models = sorted(df[df['Make'] == selected_make]['Model'].dropna().unique())
    selected_model = st.selectbox("Car Model:", filtered_models)
    
    year = st.number_input("Manufacturing Year:", min_value=1990, max_value=2026, value=2018)
    engine_size = st.number_input("Engine Size (in Litres):", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    mileage = st.number_input("Total Mileage / KM Driven:", min_value=0, max_value=500000, value=50000)
    
    fuel_types = sorted(df['Fuel Type'].dropna().unique())
    selected_fuel = st.selectbox("Fuel Type:", fuel_types)
    
    transmissions = sorted(df['Transmission'].dropna().unique())
    selected_trans = st.selectbox("Transmission:", transmissions)
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    input_data = pd.DataFrame([[selected_make, selected_model, year, engine_size, mileage, selected_fuel, selected_trans]],
                              columns=['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission'])
    
    for col in ['Make', 'Model', 'Fuel Type', 'Transmission']:
        le = LabelEncoder()
        le.fit(df[col].dropna().astype(str))
        input_data[col] = input_data[col].astype(str)
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    prediction = model.predict(input_data)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ Estimated Car Price: **$ {output:,}**")
