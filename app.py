import streamlit as st
import numpy as np
import pandas as pd
import pickle

Smoker = pickle.load(open('model.pkl', 'rb'))

st.title('Prediction model')

# total_bill	tip	sex	day	time	size

total_bill = st.number_input('total_bill')
tip = st.number_input('tip')
sex = st.text_input('sex')
day = st.text_input('day')
time = st.text_input('time')
size = st.number_input('size')


if st.button('Predict'):
    data = {'total_bill': total_bill, 'tip': tip, 'sex': sex, 'day': day, 'time': time, 'size': size}
    df = pd.DataFrame([data])
    y_pred = Smoker.predict(df)
    st.write('Predicted Survival:', y_pred[0])
