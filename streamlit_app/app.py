import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import make_prediction

# Set Streamlit app configuration
st.set_page_config(page_title="Yulu Bike Demand Prediction", layout="centered")
st.title("🚲 Yulu Bike Demand Prediction")

st.markdown("""
This app predicts the **rental demand** for Yulu bikes based on environmental and calendar-related factors. 
Please adjust the values below to simulate input features:
""")

# Input sliders and selectors
hour = st.slider("Hour of the day", 0, 23, 9)
temp = st.slider("Temperature (°C)", -20, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 60)
wind = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
visibility = st.slider("Visibility (10m)", 100, 2000, 1500)
dew_point = st.slider("Dew Point Temp (°C)", -25, 30, 10)
solar = st.slider("Solar Radiation (MJ/m2)", 0.0, 3.0, 0.5)
rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
snowfall = st.slider("Snowfall (cm)", 0.0, 10.0, 0.0)

holiday = st.radio("Is it a Holiday?", ("Yes", "No"))
functioning = st.radio("Is it a Functioning Day?", ("Yes", "No"))

year = st.selectbox("Year", [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)
weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 1)

season = st.radio("Season", ("Spring", "Summer", "Winter"))
season_spring = 1 if season == "Spring" else 0
season_summer = 1 if season == "Summer" else 0
season_winter = 1 if season == "Winter" else 0

# Encode Holiday and Functioning Day
holiday_flag = 1 if holiday == "Yes" else 0
functioning_flag = 1 if functioning == "Yes" else 0

# Create input DataFrame
input_df = pd.DataFrame([{ 
    'Hour': hour,
    'Temperature(°C)': temp,
    'Humidity(%)': humidity,
    'Wind speed (m/s)': wind,
    'Visibility (10m)': visibility,
    'Dew point temperature(°C)': dew_point,
    'Solar Radiation (MJ/m2)': solar,
    'Rainfall(mm)': rainfall,
    'Snowfall (cm)': snowfall,
    'Holiday': holiday_flag,
    'Functioning Day': functioning_flag,
    'Year': year,
    'Month': month,
    'Day': day,
    'Weekday': weekday,
    'season_Spring': season_spring,
    'season_Summer': season_summer,
    'season_Winter': season_winter
}])

# Make prediction on button click
if st.button("Predict Rented Bike Count"):
    prediction = make_prediction(input_df)
    st.success(f"Predicted Rented Bike Count: {int(prediction[0])}")

    # Load model safely for importance plot
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_bike_model.pkl'))
    model = joblib.load(model_path)
    booster = model.get_booster()

    importance_dict = booster.get_score(importance_type='gain')
    imp_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['Gain']).reset_index()
    imp_df.columns = ['Feature', 'Gain']
    imp_df['Gain %'] = 100 * imp_df['Gain'] / imp_df['Gain'].sum()
    imp_df = imp_df.sort_values(by='Gain %', ascending=False).head(10)

    # Plot top 10 features
    st.subheader("Top 10 Feature Importances (by Gain)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df, y='Feature', x='Gain %', palette='viridis', ax=ax)
    ax.set_title("Top 10 Features by Gain", fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    st.pyplot(fig)
