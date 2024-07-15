import streamlit as st
import pandas as pd
import joblib

# Load the trained and preprocessor model
model = joblib.load('nb.joblib')
sc = joblib.load('scaler.joblib')

# Title of the app
st.title('Welcome to Breast Cancer Prediction App')

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.write('### Enter the features to predict the likelihood of breast cancer:')
# Create input fields for all features
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, min_value=0.0, format="%.4f")

# Convert the input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Standardize the input data using the scaler
input_df_scaled = sc.transform(input_df)

# Display user input
# st.subheader('User Input:')
# st.write(input_df)

# Display prediction
if st.button('Predict'):
    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.warning(f'The individual is malignant with probability {prediction_proba[0][1]*100:.3f}%',icon="⚠️")
    else:
        st.success(f'The individual is benign with probability {prediction_proba[0][1]*100:.3f}%',icon="✅")


