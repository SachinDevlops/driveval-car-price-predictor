import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="DriveVal", page_icon="🚘", layout="centered")

# Load saved artifacts
model = joblib.load('car_model.pkl')
model_columns = joblib.load('model_columns.pkl')
brands = joblib.load('brands.pkl')

# App Header
st.title("🚘 DriveVal: Used Car Price Estimator")
st.write("Enter vehicle specifications below to estimate fair market resale value.")

# Input Form
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Manufacturer / Brand", brands)
    year = st.slider("Manufacturing Year", 1995, 2020, 2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=800000, value=50000, step=2000)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

with col2:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.selectbox(
        "Owner History", 
        ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
    )

# Prediction Logic
if st.button("Predict Resale Price", use_container_width=True):
    input_data = pd.DataFrame({
        'brand': [brand],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })
    
    # Align categorical encoding with the model schema
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Market Value: **₹ {max(20000, prediction):,.2f}**")