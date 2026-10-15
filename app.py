import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="CarDekho Price Prediction",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Lovable-inspired Modern CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    h1 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    p.subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 25px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        margin-top: 25px;
    }
    .prediction-card h3 {
        color: #eff6ff;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    .prediction-card h2 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0px;
    }
    .stButton>button {
        width: 100%;
        background: #0f172a;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #1e293b;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.3);
        transform: translateY(-2px);
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #cbd5e1;
    }
    .stNumberInput input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache Model & Data
@st.cache_resource
def load_model():
    return pickle.load(open('car_model.pkl', 'rb'))

@st.cache_data
def load_data():
    df = pd.read_csv('cardekho_dataset.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

model = load_model()
df = load_data()

# App Header
st.markdown("<h1>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered accurate car valuation in Indian Rupees (INR)</p>", unsafe_allow_html=True)

# Input Layout using columns for a clean card-like dashboard look
col1, col2 = st.columns(2)

with col1:
    brand_list = sorted(df['brand'].dropna().unique())
    selected_brand = st.selectbox("Select Brand", brand_list)
    
    filtered_models = sorted(df[df['brand'] == selected_brand]['model'].dropna().unique())
    selected_model = st.selectbox("Select Model", filtered_models)
    
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=25, value=3)
    km_driven = st.number_input("KM Driven", min_value=100, max_value=500000, value=25000)
    seller_type = st.selectbox("Seller Type", sorted(df['seller_type'].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_type'].dropna().unique()))

with col2:
    transmission_type = st.selectbox("Transmission Type", sorted(df['transmission_type'].dropna().unique()))
    mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=50.0, value=19.5)
    engine = st.number_input("Engine Capacity (CC)", min_value=600, max_value=5000, value=1197)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=400.0, value=85.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8], index=2)

car_name = selected_brand + " " + selected_model

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button & Output Card
if st.button("Calculate Market Price"):
    input_data = pd.DataFrame([[
        car_name, selected_brand, selected_model, vehicle_age, 
        km_driven, seller_type, fuel_type, transmission_type, 
        mileage, engine, max_power, seats
    ]], columns=[
        'car_name', 'brand', 'model', 'vehicle_age', 
        'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 
        'mileage', 'engine', 'max_power', 'seats'
    ])
    
    for col in ['car_name', 'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        input_data[col] = input_data[col].astype(str)
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    prediction = model.predict(input_data)[0]
    prediction_inr = round(prediction, 2)
    
    # Lovable styled gradient result card
    st.markdown(f"""
        <div class="prediction-card">
            <h3>Estimated Car Value</h3>
            <h2>₹ {prediction_inr:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)
