import sys
import os
sys.path.append(os.path.abspath("src"))

from predict import make_prediction
import pandas as pd

sample = pd.DataFrame({
    'Hour': [10],
    'Temperature(°C)': [45],
    'Humidity(%)': [55],
    'Wind speed (m/s)': [1.5],
    'Visibility (10m)': [2000],
    'Dew point temperature(°C)': [12.2],
    'Solar Radiation (MJ/m2)': [0.5],
    'Rainfall(mm)': [0.0],
    'Snowfall (cm)': [0.0],
    'Holiday': [0],
    'Functioning Day': [1],
    'Year': [2025],
    'Month': [6],
    'Day': [15],
    'Weekday': [4],
    'season_Spring': [0],
    'season_Summer': [1],
    'season_Winter': [0]
})

result = make_prediction(sample)
print("🔮 Predicted Rented Bike Count:", int(result[0]))
