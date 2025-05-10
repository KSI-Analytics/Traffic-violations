# Traffic Violation Predictions
## Traffic Violation Predictions is a machine learning web app built to analyze and predict various traffic-related risks using over 1.6 million traffic stop records from Montgomery County, Maryland (2012 to 2025). The application provides insights into high-risk periods, accident likelihood, and potential violations by location. It is designed to support data-driven decisions in public safety and enforcement planning.

# Project Overview
## This project uses a large-scale dataset to explore and model three main traffic safety tasks:
- High-Risk Time Prediction by Location
- Accident Occurrence Prediction
- Violation Probability Estimation by Coordinates
- Each task is powered by a custom-trained machine learning model and integrated into a lightweight web application for easy user interaction.

# Dataset Details
- Source: Data.gov
- Region: Montgomery County, Maryland
- Date Range: 2012 – 2025
- Size: ~1.6 million records (~809 MB CSV)

# Key Features:
- Stop time and location (latitude, longitude)
- Violation type
- Driver and vehicle details
- Accident, injury, and damage indicators

# Data Preprocessing Highlights
- Standardized column names for readability and consistency
- Extracted temporal features (hour, day of week, month)
- Cleaned and validated spatial coordinates
- Converted binary indicators to Boolean types
- Handled missing values via imputation or row removal
- Encoded categorical features and applied feature scaling
- Removed duplicate records to prevent training bias

# Machine Learning Models
## 1. High-Risk Period Prediction by Location
### Model: Random Forest Classifier
### Objective: Predict whether a stop occurs during a high-risk time block based on spatial and temporal features.
- Stop locations clustered into 50 zones using K-Means
- Labeled time blocks as "high-risk" if in top 25% of violation frequency
- Features: location cluster, hour, day of week, month
- Output: Top 10 high-risk zones ranked by predicted probability

## 2. Accident Prediction Model
### Model: Random Forest Classifier (with SMOTE for class balancing)
### Objective: Predict whether a traffic stop results in an accident.
- Features include time, driver demographics, vehicle type, and violation context
- Applied label encoding, one-hot encoding, and standardization
- Balanced class distribution using SMOTE to handle rare accident cases
- Output: Probability of accident for each stop record

## 3. Violation Probability Estimator by Location
### Model: Multiple Binary Logistic Regression Models
### Objective: Predict the probability of specific violation types based solely on location (latitude & longitude).
- One logistic model per violation type: accident, alcohol, personal injury, property damage, fatal, hazmat, seat belt
- Returns top 4 most likely violations for any coordinate input
- Enables geospatial risk assessment and planning

# How to Run the App
### This is a Flask-based web app. To launch locally:
### Step 1: Clone the repository
- git clone <your-repo-url>

### Step 2: Install dependencies
- pip install -r requirements.txt

### Step 3: Run the app
- python app.py
- Then open http://localhost:5000 in your browser.
