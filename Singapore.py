import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout='wide')

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

def back():
    st.session_state['page'] = 'Home'

def change():
    st.session_state['page'] = 'Model'

@st.cache_resource
def data():
    df = pd.read_csv("ResaleFlatPrice.csv")

    return df

@st.cache_resource
def loading_model():
    with open("Singapore_model.pkl",'rb') as f:
        model = pickle.load(f)
    with open("Singapore_town_ohe.pkl",'rb') as f:
        town_ohe = pickle.load(f)
    with open("Singapore_storey_le.pkl",'rb') as f:
        storey_le = pickle.load(f)
    with open("singapore_flat_le.pkl",'rb') as f:
        flat_le = pickle.load(f)

    return model,town_ohe,storey_le,flat_le

c1,c2,c3 = st.columns([0.2,0.7,0.1])
c2.title(":orange[Singapore House Resale Price Prediction]")

df = data()
model,town_ohe,storey_le,flat_le=loading_model()

if st.session_state['page'] == 'Home':
    co1,co2 = st.columns([0.1,0.9])
    co2.text("""The objective of this project is to develop a machine learning model and deploy it as a user-friendly web
application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data
of resale flat transactions.
""")
    with co2:
        w1 = st.container(border=True)
        w1.text("""
    To achieve these objectives, the solution involves:""")

        w1.write(":violet[Data Preprocessing]") 
        w1.text("Handling missing values, outlier detection, and skewness treatment.")
        w1.write(":violet[Feature Engineering]") 
        w1.text("        Creating informative features and encoding categorical variables.")
        w1.write(":violet[Model Building]") 
        w1.text("        Training and evaluating tree-based models for both regression and classification tasks.")
        w1.write(":violet[Model Deployment]")
        w1.text("""       Developing a Streamlit application for real-time predictions, enabling users to input relevant data
and receive accurate selling prices or lead statuses.""")

    co2.text("""This Project aims to assist both potential buyers and sellers in estimating the resale value of a flat.
""")
    co2.button("Launch ML Model",on_click=change)


if st.session_state['page'] == 'Model':
    c1.button("Back",on_click=back)
    q1,q2 = st.columns([0.1,0.9])
    town = st.selectbox("Select town",df['town'].unique())
    flat_type = st.selectbox("Select flat type",df['flat_type'].unique())
    storey_range = st.selectbox("Select storey range",df['storey_range'].unique())
    floor_sq = st.number_input(f"Enter Area, Min={df['floor_area_sqm'].min()}, Max={df['floor_area_sqm'].max()}", value=None, placeholder="Type a number...")
    year = st.number_input(f"Select Lease Star year,Range = {df['lease_commence_date'].min()} - {df['lease_commence_date'].max()}",value=None,placeholder='Type a number...')
    m = 'Formula = (Lease start year + 99) - Current year'
    remain = st.number_input("Enter remaining lease year",help=m, value=None, placeholder="Type a number...")
    v = st.button("Predict")
    if v:
        result = np.array([[town,flat_type,storey_range,floor_sq,year,remain]])
        t = town_ohe.transform(result[:,[0]]).toarray()
        f_t = flat_le.transform(result[:,[1]])
        f_t = f_t.reshape(-1,1)
        s_t = storey_le.transform(result[:,[2]])
        s_t = s_t.reshape(-1,1)
        features = np.concatenate([t,f_t,s_t,result[:,[3,4,5]]],axis=1)
        c = model.predict(features)
        y = round(*c,2)
        st.success(f"The Resale Value is: {y}")
