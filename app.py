import streamlit as st
import pickle
import numpy as np

# മോഡൽ ലോഡ് ചെയ്യുക
model = pickle.load(open('car_model.pkl', 'rb'))

st.title("🚗 Used Car Price Prediction App")
st.write("കാറിന്റെ വിവരങ്ങൾ നൽകി അതിന്റെ വില എത്രയെന്ന് അറിയാം!")

# യൂസർ ഇൻപുട്ടുകൾ സ്വീകരിക്കാൻ Form ഉണ്ടാക്കുക
with st.form("car_form"):
    year = st.number_input("കാർ നിർമ്മിച്ച വർഷം (Year):", min_value=2000, max_value=2026, value=2018)
    kms = st.number_input("ഓടിയ കിലോമീറ്റർ (Kilometers Driven):", min_value=0, max_value=500000, value=30000)
    owner = st.selectbox("ഉടമസ്ഥത (Owner Type):", [1, 2, 3])
    present_price = st.number_input("ഷോറൂം വില / നിലവിലെ വില (Lakhs):", min_value=0.1, max_value=100.0, value=8.0)
    
    fuel_type = st.selectbox("ഇന്ധനം (Fuel Type):", ["Petrol", "Diesel", "Other"])
    transmission = st.selectbox("ഗിയർ ബോക്സ് (Transmission):", ["Manual", "Automatic"])
    
    submit_button = st.form_submit_button(label="വില പ്രവചിക്കുക (Predict Price)")

# പ്രവചനം നടത്തുന്ന ഭാഗം
if submit_button:
    # Fuel Type-നും Transmission-നും ഉള്ള Encoding
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    trans_manual = 1 if transmission == "Manual" else 0
    
    # അറേ തയ്യാറാക്കുക
    features = np.array([[year, kms, owner, present_price, fuel_diesel, trans_manual]])
    
    # പ്രവചനം
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ ഈ കാറിന്റെ আনুമാനിക വില: **₹ {output} ലക്ഷം**")