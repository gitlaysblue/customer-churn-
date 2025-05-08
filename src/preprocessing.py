import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the customer churn dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    # Calculate missing values percentage
    missing_values = df.isnull().sum() / len(df) * 100
    
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables using Label Encoding
    """
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    return df, label_encoders

def scale_numerical_features(df):
    """
    Scale numerical features using StandardScaler
    """
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df, scaler

def prepare_data(df, target_column='Churn'):
    """
    Prepare data for modeling
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(filepath):
    """
    Complete preprocessing pipeline
    """
    # Load data
    df = load_data(filepath)
    if df is None:
        return None
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical variables
    df, label_encoders = encode_categorical_variables(df)
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler
    }

if __name__ == "__main__":
    # Example usage
    data = preprocess_pipeline("data/raw/customer_churn_data.csv")
    if data:
        print("Preprocessing completed successfully!")
        print(f"Training set shape: {data['X_train'].shape}")
        print(f"Testing set shape: {data['X_test'].shape}") 