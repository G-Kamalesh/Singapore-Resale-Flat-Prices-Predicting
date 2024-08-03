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
def load_data():
    df = pd.read_csv("K:\data\ResaleflatpricesbasedonregistrationdatefromJan2017onwards (1).csv")
    return df

@st.cache_resource
def loading_model():
    with open(r"K:\data\ML Model\Singapore LE\Singapore_model.pkl",'rb') as f:
        model = pickle.load(f)
    with open(r"K:\data\ML Model\Singapore LE\scaler.pkl",'rb') as f:
        scaler = pickle.load(f)
    with open(r"K:\data\ML Model\Singapore LE\flat_type_le.pkl",'rb') as f:
        flat_type_le = pickle.load(f)
    with  open(r"K:\data\ML Model\Singapore LE\block_le.pkl",'rb') as f:
        block_le = pickle.load(f)
    with open(r"K:\data\ML Model\Singapore LE\street_le.pkl",'rb') as f:
        street_le = pickle.load(f)
    with open(r"K:\data\ML Model\Singapore LE\town_le.pkl",'rb') as f:
        town_le = pickle.load(f)
    with open(r"K:\data\ML Model\Singapore LE\flat_model_le.pkl",'rb') as f:
        flat_model_le = pickle.load(f)

    return model,scaler,flat_type_le,block_le,street_le,town_le,flat_model_le

c1,c2,c3 = st.columns([0.2,0.7,0.1])
c2.title(":orange[Singapore House Resale Price Prediction]")

df = load_data()
model,scaler,flat_type_le,block_le,street_le,town_le,flat_model_le = loading_model()

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
        w1.text("Creating informative features and encoding categorical variables.")
        w1.write(":violet[Model Building]") 
        w1.text("Training and evaluating tree-based models for regression task.")
        w1.write(":violet[Model Deployment]")
        w1.text("""Developing a Streamlit application for real-time predictions, enabling users to input relevant data
and receive accurate selling price.""")

    co2.text("""This Project aims to assist both potential buyers and sellers in estimating the resale value of a flat.
""")
    with co2:
        m1,m2,m3,m4,m5 = st.columns([0.2,0.2,0.2,0.2,0.5],gap='small')
        m2.link_button("Linkedin Profile","https://www.linkedin.com/in/g-kamaleashwar-28a2802ba/")
        m3.link_button("Hugging Face","https://huggingface.co/spaces/kamalesh-g/Singapore-RealEstate-Streamlit")
        m4.button("Launch ML Model",on_click=change)


if st.session_state['page'] == 'Model':
    c1.button("Back",on_click=back)
    q1,q2 = st.columns([0.1,0.9])

    town_unique = df['town'].unique()
    town = st.selectbox("Select town", town_unique)
    town_filtered = df[df['town'] == town]
    
    ft_unique = town_filtered['flat_type'].unique()
    flat_type = st.selectbox("Select flat type",ft_unique)
    
    ftm_unique = town_filtered['flat_model'].unique()
    flat_model = st.selectbox("Select flat model",ftm_unique)

    block_unique = town_filtered['block'].unique()
    block = st.selectbox("Select block",block_unique)

    street_unique = town_filtered['street_name'].unique()
    street = st.selectbox("Select street name",df['street_name'].unique())
    
    area = st.number_input(f"Enter Area, Min=31.00", value=None, placeholder="Type a number...")
    start_year = st.number_input(f"Enter Lease Star year",value=None,placeholder='Type a number...')

    end_year = st.number_input("Enter remaining lease year", value=None, placeholder='Formula = (Lease start year + 99) - Current year')
    
    v = st.button("Predict")
    if v:
        age = end_year - start_year
        array = np.array([[town,flat_type,block,street,area,flat_model,end_year,age]])
        t = town_le.transform(array[:,0])
        f_t = flat_type_le.transform(array[:,1])
        f_m = flat_model_le.transform(array[:,5])
        b = block_le.transform(array[:,2])
        s = street_le.transform(array[:,3])
        a = np.log(array[:,4].astype(float))
        combined_features = np.column_stack([t, f_t, b, s, a, f_m, array[:, [6, 7]].astype(float)])
        features = scaler.transform(combined_features)
        c = model.predict(features)
        y = np.exp(c)
        st.success(f"The Resale Value is: S$ {y[0]:.2f}")

