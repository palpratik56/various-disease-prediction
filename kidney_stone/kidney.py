import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
dt = joblib.load('dt.joblib')

st.title('Kidney Stone Prediction')

gr = st.number_input('Gravity',min_value=1.005, max_value=1.034, step=0.002)
ph = st.number_input('PH',min_value=4.75, max_value=7.95, step=0.4)
os = st.number_input('Osmo',min_value=187.00, max_value=1236.00, step=7.0)
con = st.number_input('Condition',min_value=5.0, max_value=39.0, step=0.5)
ur = st.number_input('Urea',min_value=10, max_value=620, step=8)
cal = st.number_input('Calcium',min_value=0.17, max_value=13.00, step=0.9)

input = pd.DataFrame(
    {'gravity': gr, 'ph': ph,'osmo': os, 'cond': con, 'urea': ur, 'calc':cal }, 
    index=[0] )

# Display user input
st.subheader('User Input:')
st.write(input)

# sc_input = sc.transform(input)

y_pri=dt.predict(input)
prediction_proba = dt.predict_proba(input)

if st.button('Predict'):
    st.subheader('Prediction:')
    if y_pri[0]==0:
        st.success(f'No risk with probability {(prediction_proba[0][0]*100):.3f}%')
    else:
        st.warning(f'Risk of stone with probability {(prediction_proba[0][1]*100):.3f}%')
