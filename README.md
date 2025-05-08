# Customer Churn Prediction Project 🎯

## Developer: laysblue

This project implements a machine learning solution for predicting customer churn in a telecommunications company. The model helps identify customers who are likely to discontinue services, enabling proactive retention strategies.

## Project Structure
```
customer_churn_prediction/
│
├── data/                    # Data directory
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
│
├── notebooks/              # Jupyter notebooks
│   └── EDA.ipynb          # Exploratory Data Analysis
│
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── feature_engineering.py  # Feature engineering code
│   ├── model.py           # Model training and evaluation
│   └── utils.py           # Utility functions
│
├── models/                 # Saved model files
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model training and hyperparameter tuning
- Model evaluation and performance metrics
- Prediction functionality

## Dataset
The project uses a telecommunications customer dataset with the following features:
- Customer demographics
- Service subscriptions
- Account information
- Usage patterns
- Churn status

## Models Used
- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone the repository
```bash
git clone [repository-url]
cd customer-churn-prediction
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Usage
1. Data Preprocessing:
```bash
python src/preprocessing.py
```

2. Train Model:
```bash
python src/model.py
```

## Results
- Model Accuracy: ~85%
- Precision: ~83%
- Recall: ~82%
- F1-Score: ~82%

## Contributing
Feel free to fork the project and submit pull requests for any improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: Telecommunications Customer Dataset
- Inspired by real-world churn prediction challenges
- Built as part of the Machine Learning Projects series 