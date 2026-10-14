import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Trained model ലോഡ് ചെയ്യുന്നു
model = pickle.load(open('car_model.pkl', 'rb'))

# 2. ഡാറ്റാസെറ്റ് ലോഡ് ചെയ്യുന്നു (Small dataset used to avoid size limit)
df = pd.read_csv('used_cars_small.csv')

st.title("🚗 Advanced Real Used Car Price Prediction App")
st.write("Select car details below to get an accurate estimated price!")

with st.form("car_form"):
    # ബ്രാൻഡുകൾ ലിസ്റ്റ് ചെയ്യുന്നു
    brands = sorted(df['brand'].dropna().unique())
    selected_brand = st.selectbox("Car Brand (Brand):", brands)
    
    # ആ ബ്രാൻഡിലുള്ള മോഡലുകൾ മാത്രം ഫിൽട്ടർ ചെയ്യുന്നു
    filtered_models = sorted(df[df['brand'] == selected_brand]['model'].dropna().unique())
    selected_model = st.selectbox("Car Model:", filtered_models)
    
    year = st.number_input("Manufacturing Year:", min_value=1990, max_value=2026, value=2018)
    mileage = st.number_input("Mileage (Driven Miles):", min_value=0, max_value=300000, value=40000)
    
    fuel_types = sorted(df['fuel_type'].dropna().unique())
    selected_fuel = st.selectbox("Fuel Type:", fuel_types)
    
    transmissions = sorted(df['transmission'].dropna().unique())
    selected_trans = st.selectbox("Transmission:", transmissions)
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    # User നൽകിയ വിവരങ്ങൾ ഡാറ്റാ ഫ്രെയിം രൂപത്തിലാക്കുന്നു
    input_data = pd.DataFrame([[selected_brand, selected_model, year, mileage, selected_fuel, selected_trans]],
                              columns=['brand', 'model', 'model_year', 'milage', 'fuel_type', 'transmission'])
    
    # Label encoding പ്രോസസ്സ് ആപ്പിനും ബാധകമാക്കുന്നു
    for col in ['brand', 'model', 'fuel_type', 'transmission']:
        le = LabelEncoder()
        # Full dataset-ലെ values വെച്ച് fit ചെയ്യുന്നു
        le.fit(df[col].dropna().astype(str))
        
        # User input-ൽ ഉള്ള value യ്ക്കുള്ള transform സുരക്ഷിതമാക്കാൻ 
        # unseen values വന്നാൽ ഹാൻഡിൽ ചെയ്യാനുള്ള കോഡ്
        input_data[col] = input_data[col].astype(str)
        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    # Prediction നടത്തുന്നു
    prediction = model.predict(input_data)
    output = round(prediction[0], 2)
    
    st.success(f"🏷️ Estimated Car Price: **$ {output:,}**")
