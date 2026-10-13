import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('car_model.pkl', 'rb'))

st.title("🚗 Advanced Used Car Price Prediction App")
st.write("Enter the car details below to get an accurate estimated price!")

with st.form("car_form"):
    make = st.selectbox("Car Brand (Make):", ['Audi', 'BMW', 'Ford', 'Honda', 'Toyota'])
    model_name = st.selectbox("Car Model:", ['Model A', 'Model B', 'Model C', 'Model D', 'Model E'])
    
    year = st.number_input("Car Manufacturing Year:", min_value=2000, max_value=2026, value=2015)
    engine_size = st.number_input("Engine Size (L):", min_value=1.0, max_value=5.0, value=2.5)
    mileage = st.number_input("Mileage (Driven KM):", min_value=0, max_value=300000, value=50000)
    
    fuel_type = st.selectbox("Fuel Type:", ['Diesel', 'Electric', 'Petrol'])
    transmission = st.selectbox("Transmission:", ['Automatic', 'Manual'])
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # Model കൃത്യമായി പ്രതീക്ഷിക്കുന്ന അതേ ഓർഡറിലുള്ള Columns
    input_data = {
        'Year': [year],
        'Engine Size': [engine_size],
        'Mileage': [mileage],
        'Make_BMW': [1 if make == 'BMW' else 0],
        'Make_Ford': [1 if make == 'Ford' else 0],
        'Make_Honda': [1 if make == 'Honda' else 0],
        'Make_Toyota': [1 if make == 'Toyota' else 0],
        'Model_Model B': [1 if model_name == 'Model B' else 0],
        'Model_Model C': [1 if model_name == 'Model C' else 0],
        'Model_Model D': [1 if model_name == 'Model D' else 0],
        'Model_Model E': [1 if model_name == 'Model E' else 0],
        'Fuel Type_Electric': [1 if fuel_type == 'Electric' else 0],
        'Fuel Type_Petrol': [1 if fuel_type == 'Petrol' else 0],
        'Transmission_Manual': [1 if transmission == 'Manual' else 0]
    }
    
    final_input = pd.DataFrame(input_data)
    
    # Prediction
    prediction = model.predict(final_input)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ Estimated Car Price: **₹ {output:,}**")
