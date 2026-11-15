import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load the pretrained model
with open('pipeline_employee.pkl', 'rb') as file:
    model = dill.load(file)

feature_dict_employee = load('feature_dict_employee.pkl')
 
# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction


st.title('Employee Churn Prediction App')
st.subheader('Created by Muhammad Shahid')

# Display categorical features
st.subheader('Categorical Features')
categorical_input = feature_dict_employee.get('CATEGORICAL')
categorical_input_vals={}
for i, col in enumerate(categorical_input.get('Column Name').values()):
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i],key=col)

# Load numerical features
numerical_input = feature_dict_employee.get('NUMERICAL')

# Display numerical features
st.subheader('Numerical Feature')
st.markdown(':red[Please Enter Age from 22 to 41]')
numerical_input = feature_dict_employee.get('NUMERICAL')
numerical_input_vals={}
for col in  numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col,key=col,value=0,step=1,format='%d')

# Combine numerical and categorical input dicts
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))

input_data = pd.DataFrame.from_dict(input_data,orient='index').T

# Churn Prediction
if st.button('Predict'):
    prediction = predict_churn(input_data)[0]
    prediction_dict = {1: "Yes", 0: "No"}
    translation_dict = {1 : "Expected", 0 : "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    prediction_label = prediction_dict.get(prediction)
    st.write(f'The Prediction is **{prediction_label}**, Hence employee is **{prediction_translate}** to churn.')