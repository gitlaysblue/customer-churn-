# Customer Churn Prediction

![Churn Prediction](https://via.placeholder.com/800x400.png?text=Customer+Churn+Prediction)

## Project Overview

This project implements a machine learning solution to predict customer churn for a telecommunications company. By analyzing customer data and identifying patterns that lead to customer attrition, businesses can take proactive measures to retain valuable customers and improve overall satisfaction.

## Table of Contents
- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)
- [Developer Information](#developer-information)

## Project Description

Customer churn, or customer attrition, refers to the loss of clients or customers. For telecommunications companies, understanding why and when customers are likely to churn is crucial for implementing retention strategies and maintaining revenue streams.

This project:
1. Explores and analyzes telco customer data
2. Prepares and preprocesses the data for machine learning
3. Selects relevant features for prediction
4. Builds and compares multiple ML models
5. Evaluates model performance using appropriate metrics
6. Provides insights and recommendations based on the findings

## Technologies Used

- Python 3.8+
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for data visualization
- XGBoost for gradient boosting
- Jupyter Notebook for interactive development

## Dataset Description

The dataset contains information about telecom customers including:

- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, streaming, backup)
- Billing information (monthly charges, total charges)
- Churn status (whether the customer left the company)

The dataset is provided in CSV format and contains approximately 7,000 customer records.

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Original dataset
│   ├── cleaned_telco_data.csv                  # Preprocessed dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb               # EDA and data insights
│   ├── 02_data_preprocessing.ipynb             # Data cleaning and preparation
│   ├── 03_feature_engineering.ipynb            # Feature selection and engineering
│   ├── 04_model_building.ipynb                 # ML model development
│   └── 05_model_evaluation.ipynb               # Performance evaluation
├── src/
│   ├── data_preprocessing.py                   # Data preprocessing functions
│   ├── feature_engineering.py                  # Feature engineering functions
│   ├── model_training.py                       # Model training pipeline
│   └── model_evaluation.py                     # Model evaluation functions
├── models/
│   ├── random_forest_model.pkl                 # Saved Random Forest model
│   ├── xgboost_model.pkl                       # Saved XGBoost model
│   └── logistic_regression_model.pkl           # Saved Logistic Regression model
├── requirements.txt                            # Project dependencies
├── README.md                                   # Project documentation
└── app.py                                      # Simple prediction application
```

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/laysblue/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

## Usage

1. Explore the Jupyter notebooks in the `notebooks/` directory to understand the analysis process
2. Run the data preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```
3. Train the models:
   ```bash
   python src/model_training.py
   ```
4. Evaluate model performance:
   ```bash
   python src/model_evaluation.py
   ```
5. For predictions on new data, use the app:
   ```bash
   python app.py
   ```

## Key Findings

- Contract type is the most important predictor of churn
- Customers with month-to-month contracts are more likely to churn
- Higher monthly charges correlate with increased churn probability
- Customers with fiber optic internet service tend to churn more often
- Tenure shows an inverse relationship with churn - longer-term customers are less likely to leave

## Model Performance

The final XGBoost model achieved:
- Accuracy: 0.82
- Precision: 0.78
- Recall: 0.67
- F1 Score: 0.72
- AUC-ROC: 0.85

## Future Improvements

- Collect and incorporate additional customer interaction data
- Implement a time-series analysis to predict when a customer might churn
- Develop a customer segmentation model to tailor retention strategies
- Deploy the model as a web service for real-time predictions
- Implement A/B testing for retention strategies based on model predictions

## Developer Information

This project was developed by **laysblue** as part of a machine learning project for genrative ai minor by intellipat . For questions or collaborations, please contact via GitHub.

---

