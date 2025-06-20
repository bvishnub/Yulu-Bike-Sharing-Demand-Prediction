# Yulu Bike Sharing Demand Prediction 🚲

This project aims to accurately predict bike-sharing demand for **Yulu Bikes** using various regression-based machine learning models. It also includes an interactive **Streamlit web application**, publicly deployed, that allows users to input custom conditions and get real-time demand predictions.

---

## 🎯 Objective

To predict Yulu bike-sharing demand using features such as weather, time, and holidays — optimizing resource allocation, enhancing user experience, and improving operational efficiency.

---

## 🏢 Business Context

Urban mobility providers like **Yulu Bike** must manage fleets in response to unpredictable demand. Using data-driven forecasting, they can:

- 🔁 **Optimize fleet distribution** using features like temperature, humidity, time of day, and holidays.
- 😃 **Improve customer satisfaction** by reducing stockouts and over-availability.
- ⚙️ **Streamline operations** and reduce unnecessary maintenance or idle times.
- 🌦 **Adapt to environmental shifts** in real-time using predictive analytics.

---

## 🗂️ Project Structure

```
├── models/
│   ├── xgboost_bike_model.pkl
│   └── features.pkl
├── notebook/
│   ├── eda.ipynb
│   └── model_building.ipynb
├── processed_data/
│   └── bike_cleaned_data.csv
├── raw_data/
│   └── SeoulBikeData.csv
├── src/
│   ├── predict.py
│   ├── test_predict.py
│   └── __init__.py
├── streamlit_app/
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Model Performance Summary

| Model                          | Dataset | R² Score         | RMSE                  | MAE                  | Notes                             |
|-------------------------------|---------|------------------|------------------------|----------------------|-----------------------------------|
| **Linear Regression**         | Train   | 0.279            | 1.345                  | 0.820                | Basic model                       |
|                               | Test    | 0.289            | 1.312                  | 0.799                |                                   |
| **Ridge Regression**          | Test    | 0.289            | 1.312                  | 0.799                | Best alpha = 10                   |
| **Lasso Regression**          | Test    | 0.289            | 1.312                  | 0.799                | Best alpha = 0.001                |
| **Random Forest (Tuned)**     | Train   | 0.885            | 218.237                | 125.974              | After hyperparameter tuning       |
|                               | Test    | 0.814            | 278.602                | 159.769              |                                   |
| **Gradient Boosting (GBM)**   | Train   | 0.843            | 254.946                | 150.259              | Base model                        |
|                               | Test    | 0.828            | 267.519                | 156.875              |                                   |
| **XGBoost (Base Model)**      | Train   | 0.980            | 91.027                 | 50.303               |                                   |
|                               | Test    | 0.944            | 152.782                | 83.187               |                                   |
| **XGBoost (Tuned Model)**     | Train   | 0.985            | 79.733                 | 44.947               | Final chosen model                |
|                               | Test    | 0.947            | 148.334                | 77.339               | Best overall performance          |

---

## 📱 Streamlit Web App

Use the deployed interactive app to test demand predictions live based on inputs like temperature, humidity, hour, etc.

🌐 **Live App**: [Launch Application](https://yulu-bike-demand-predictor.streamlit.app/)  
🧠 **Model**:  XGBoost  
📁 **Input**: User-provided custom data 

---

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/yulu-bike-sharing-demand-prediction.git
cd yulu-bike-sharing-demand-prediction

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app/app.py
```

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, Random Forest, Gradient Boosting
- Streamlit for app deployment
- Joblib for saving models
- Matplotlib & Seaborn for visualization

---

## ✅ Conclusion

The final XGBoost model provides highly accurate predictions with a Test R² of 0.947, making it the optimal choice for production. With this solution, Yulu Bike can improve decision-making, enhance user experience, and scale operations more effectively. The deployed Streamlit app allows anyone to test the model instantly with real-time inputs.

---

