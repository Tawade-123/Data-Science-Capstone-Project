import streamlit as st
import numpy as np
import joblib

# Load the trained model, scaler, and label encoder
model = joblib.load('ML_Section/best_model.pkl')
scaler = joblib.load('ML_Section/scaler.pkl')
label_encoder = joblib.load('ML_Section/label_encoder.pkl')

# Dictionaries for feature encoding/decoding
fuel_map = {1: 'Diesel', 2: 'Petrol', 3: 'CNG', 4: 'LPG', 5: 'Electric'}
seller_type_map = {1: 'Individual', 2: 'Dealer', 3: 'Trustmark Dealer'}
transmission_map = {1: 'Manual', 2: 'Automatic'}
owner_map = {
    1: 'First Owner', 
    2: 'Second Owner', 
    3: 'Third Owner', 
    4: 'Fourth & Above Owner', 
    5: 'Test Drive Car'
}
selling_price_range_map = {
    1: 'Less than 100000',
    2: '100001 - 1000000', 
    3: '1000001 - 2000000', 
    4: '2000001 - 3000000', 
    5: '3000001 - 4000000', 
    6: '4000001 - 5000000', 
    7: '5000001 - 6000000', 
    8: '6000001 - 7000000', 
    9: '7000001 - 8000000', 
    10: '8000001 - 9000000'
}

# Reverse mappings for input conversion
fuel_reverse_map = {v: k for k, v in fuel_map.items()}
seller_type_reverse_map = {v: k for k, v in seller_type_map.items()}
transmission_reverse_map = {v: k for k, v in transmission_map.items()}
owner_reverse_map = {v: k for k, v in owner_map.items()}


# Streamlit app
st.title('Car Selling Price Prediction')
st.write('This app predicts the selling price range of a car based on its features.')

# Input form
year = st.number_input('Year of Manufacture', min_value=1992, max_value=2020, step=1)
km_driven = st.number_input('Kilometers Driven', min_value=0, step=999999)

fuel = st.selectbox('Fuel Type', list(fuel_reverse_map.keys()))
seller_type = st.selectbox('Seller Type', list(seller_type_reverse_map.keys()))
transmission = st.selectbox('Transmission Type', list(transmission_reverse_map.keys()))
owner = st.selectbox('Number of Previous Owners', list(owner_reverse_map.keys()))

if st.button('Predict Selling Price'):
    try:
        # Pre-process the input data
        fuel_encoded = fuel_reverse_map[fuel]
        seller_type_encoded = seller_type_reverse_map[seller_type]
        transmission_encoded = transmission_reverse_map[transmission]
        owner_encoded = owner_reverse_map[owner]

        # Combine the processed features
        features = np.array([[year, km_driven, fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded]])

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features)

        # Predict using the loaded model
        prediction = model.predict(scaled_features)

        # Decode the label back to user-friendly format
        predicted_label = selling_price_range_map[prediction[0]]

        # Display result
        st.success(f'Predicted Selling Price Range: {predicted_label}')
    except Exception as e:
        st.error(f'Error in prediction: {str(e)}')
