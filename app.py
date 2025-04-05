import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import pickle

#load the traianed model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoder objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('OneHot_encoder.pkl', 'rb') as f:
    OneHot_encoder = pickle.load(f)

with open('Label_encoder_gender.pkl', 'rb') as f:
    Label_encoder_gender = pickle.load(f)

##streamlit app
st.title("Customer Churn Prediction")
st.subheader("Predicting whether a customer will leave the bank or not")
st.write("Please enter the following details:")

#user input
geograpgy =  st.selectbox('Geography',  OneHot_encoder.categories_[0])
gender = st.selectbox('Gender', Label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10) 
num_of_products = st.slider('Number of Products',1,4) 
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Active Member', [0, 1])


##prepare input data
input_data = pd.DataFrame({
    'Geography': [geograpgy],
    'CreditScore': [credit_score],
    'Gender' : [Label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

##one hot encode geography
geo_encoded = OneHot_encoder.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OneHot_encoder.get_feature_names_out(['Geography']))

#combine the encoded geography with the input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

#drop the original geography column
input_data = input_data.drop(columns=['Geography'])

#scale the input data
input_data_scaled = scaler.transform(input_data)

#make prediction
prediction = model.predict(input_data_scaled)  
prediction_prob = prediction[0][0] 


if prediction_prob  > 0.5:
    st.success("The customer is likely to leave the bank.") 
else:
    st.success("The customer is likely to stay with the bank.")
