import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Title of the app
st.title('Diabetes Prediction App')

# Function to get user input
def user_input():
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=0, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=125, value=70, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20, step=1)
    insulin = st.number_input('Insulin', min_value=0, max_value=850, value=30, step=1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.2)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.07, max_value=2.5, value=0.5, step=0.02)
    age = st.number_input('Age', min_value=19, max_value=80, value=30, step=1)
    
    # Create a dictionary of inputs
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input()

# Display user input
st.subheader('User Input:')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display prediction
if st.button('Predict'):
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.warning('The model predicts that the individual has diabetes',icon="⚠️")
    else:
        st.success('The model predicts that the individual does not have diabetes',icon="✅")

# Display prediction probability
# st.subheader('Prediction Probability:')
st.write(f'Probability of having diabetes: {prediction_proba[0][1]:.2f}')
st.write(f'Probability of not having diabetes: {prediction_proba[0][0]:.2f}')

