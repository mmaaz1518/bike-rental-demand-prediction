# 🚲 Bike Rental Demand Prediction

## Project Overview
This project uses Machine Learning to predict hourly bike rental demand based on weather conditions, time, and other relevant factors.

The application is built using Python, Pandas, Scikit-learn, and Streamlit.

## Problem Statement
Bike rental demand changes according to the time of day, weather, season, and whether a day is a working day or a holiday. This project aims to estimate the number of bike rentals for a given set of conditions.

## Dataset
The project uses the Bike Sharing Dataset from the UCI Machine Learning Repository.

The target variable is `cnt`, which represents the total number of bike rentals in an hour.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn for data analysis

## Machine Learning Model
Linear Regression is used to predict hourly bike rental demand.

## Application Features
- Select year, month, hour, season, and weather.
- Enter temperature, humidity, and wind speed.
- Generate an estimated hourly bike rental count.

## How to Run Locally

1. Install Python.
2. Install the required libraries using:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

## Model Evaluation
Model performance should be reported using MAE, MSE, RMSE, and R² after evaluating the model on a separate test dataset.

## Project Status
The Streamlit application runs locally and produces bike rental predictions. Online deployment and final model evaluation are pending.
