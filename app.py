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
    # 1. User input dataframe ഉണ്ടാക്കുന്നു
    input_data = pd.DataFrame([[make, model_name, year, engine_size, mileage, fuel_type, transmission]],
                              columns=['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission'])
    
    # 2. Original dataset-മായി ചേർത്ത് ഒരേപോലെ get_dummies ചെയ്യുന്നു (Columns match ചെയ്യാൻ)
    df_train = pd.read_csv('Car_Price_Prediction.csv')
    df_combined = pd.concat([input_data, df_train.drop('Price', axis=1)], ignore_index=True)
    
    df_encoded = pd.get_dummies(df_combined, columns=['Make', 'Model', 'Fuel Type', 'Transmission'], drop_first=True)
    
    # 3. User input ചെയ്ത ആദ്യത്തെ row മാത്രം എടുത്തു പ്രെഡിക്റ്റ് ചെയ്യുന്നു
    final_input = df_encoded.iloc[[0]]
    
    prediction = model.predict(final_input)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ Estimated Car Price: **₹ {output:,}**")
