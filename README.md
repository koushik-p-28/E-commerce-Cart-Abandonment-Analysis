# E-commerce Cart Abandonment Prediction  

This project analyzes customer session data to predict whether users are likely to abandon their shopping carts. It uses a Random Forest model trained on behavioral session features and is deployed via a Streamlit web app.

---

## Overview

This Streamlit web app predicts whether an online shopping session will likely result in **cart abandonment** or **successful purchase**, based on user behavior like time on site, pages visited, and cart value.

Built as part of an AI/ML internship project, this app demonstrates a complete machine learning workflow — from data preprocessing to model deployment — with a focus on real-world e-commerce impact.

---

## Project Objective

- Analyze customer behavior and session data  
- Build predictive models to flag likely cart abandoners  
- Deploy an interactive app for real-time predictions  

---

## Features

- Predicts **likelihood of cart abandonment** based on:
  - Pages visited
  - Time spent on site
  - Cart value in USD
- Provides **estimated risk percentage**
- Clean and interactive UI via **Streamlit**
- Powered by a trained **Random Forest Classifier**

---

## Machine Learning Workflow

- **1. Data Preprocessing:**  
  Missing values check, outlier detection, feature selection

- **2. Exploratory Data Analysis:**  
  Visualizations using seaborn & matplotlib to understand behavior patterns

- **3. Model Development:**  
  Logistic Regression and Random Forest (with class imbalance handling)

- **4. Evaluation & Tuning:**  
  ROC curve, classification report, cross-validation, feature importance

- **5. Clustering:**  
  KMeans + PCA for behavioral segmentation

- **6. Deployment:**  
  Streamlit app for live prediction

---

## Model Details

- Algorithm: Random Forest Classifier  
- Evaluation Metrics: Accuracy, F1-score, ROC AUC  
- Feature importance: `time_on_site` > `pages_visited` > `cart_value`  
- Model saved using `joblib` as `final_model.pkl`

---

## Dataset Description

| Column         | Description                               |
|----------------|-------------------------------------------|
| `session_id`   | Unique identifier for each session        |
| `pages_visited`| Number of pages visited in the session    |
| `time_on_site` | Total time spent on site (in seconds)     |
| `cart_value`   | Total cart value in USD                   |
| `abandoned`    | Target variable (1 = abandoned, 0 = purchase) |

---

## Streamlit App

🔗 [Live Web App](https://e-commerce-cart-abandonment-analysis.streamlit.app/)

---

## Project Structure

```
📁 cart-abandonment-app/
├── shopping_abandonment.csv # Dataset
├── final_model.pkl # Trained model file
├── streamlit_app.py # Deployed app code
├── CartAbandonmentAnalysis.ipynb # Full ML notebook
├── Final_Report.pdf # Formal project report
├── README.md # This file
└── requirements.txt # Python dependencies
```

