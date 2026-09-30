# Smart Electricity Usage Prediction

A machine learning project for predicting household electricity consumption using environmental and time-based features.

## Overview

This project uses the **Appliances Energy Prediction** dataset to develop a Linear Regression model for estimating household appliance energy consumption.

The project demonstrates a basic machine learning workflow:

- Data loading and preprocessing
- Date/time feature extraction
- Feature selection
- Train/test data splitting
- Linear Regression model training
- Model prediction
- Performance evaluation

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Linear Regression

## Dataset

The project uses the **Appliances Energy Prediction** dataset.

The dataset contains household environmental measurements such as:

- Temperature
- Relative humidity
- Visibility
- Pressure
- Windspeed
- Humidity
- Irradiance

The target variable is `Appliances`, representing appliance energy consumption.

The dataset itself is not included in this repository.

## Features

### Time-based Features

- Hour
- Day
- Month
- Day of week

### Environmental Features

- Temperature
- Relative humidity
- Visibility
- Pressure
- Windspeed
- Humidity
- Irradiance

## Model

A **Linear Regression** model is used to predict appliance energy consumption.

The dataset is divided into:

- 80% training data
- 20% testing data

## Evaluation Metrics

The model is evaluated using:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

## Project Structure

```text
smart-electricity-predictor/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
