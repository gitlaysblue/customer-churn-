import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_feature_importance(feature_importance, title="Feature Importance", figsize=(12, 6)):
    """
    Plot feature importance
    """
    plt.figure(figsize=figsize)
    feature_importance.head(10).plot(kind='barh')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", figsize=(8, 6)):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('models/roc_curve.png')
    plt.close()

def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

def analyze_feature_correlations(df, target_col='Churn'):
    """
    Analyze and plot feature correlations
    """
    # Calculate correlations
    correlations = df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('models/correlation_matrix.png')
    plt.close()
    
    # Get correlations with target variable
    target_correlations = correlations[target_col].sort_values(ascending=False)
    
    return target_correlations

def plot_feature_distributions(df, target_col='Churn', n_features=5):
    """
    Plot distributions of top n features by correlation with target
    """
    correlations = df.corr()[target_col].sort_values(ascending=False)
    top_features = correlations[1:n_features+1].index
    
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
    for i, feature in enumerate(top_features):
        if df[feature].dtype in ['int64', 'float64']:
            sns.histplot(data=df, x=feature, hue=target_col, ax=axes[i])
        else:
            sns.countplot(data=df, x=feature, hue=target_col, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} by {target_col}')
    
    plt.tight_layout()
    plt.savefig('models/feature_distributions.png')
    plt.close()

def save_model_report(model_results, feature_importance, output_file='models/model_report.txt'):
    """
    Save model performance report to a file
    """
    with open(output_file, 'w') as f:
        f.write("Model Performance Report\n")
        f.write("======================\n\n")
        
        f.write("Classification Report:\n")
        f.write(model_results['classification_report'])
        f.write("\n\n")
        
        f.write("ROC AUC Score:\n")
        f.write(f"{model_results['roc_auc_score']:.4f}\n\n")
        
        f.write("Top 10 Important Features:\n")
        for feature, importance in feature_importance.head(10).items():
            f.write(f"{feature}: {importance:.4f}\n")

def create_prediction_example(model, scaler, label_encoders):
    """
    Create an example of how to use the model for predictions
    """
    example_code = """
# Example of how to use the model for predictions
import joblib
import pandas as pd

def prepare_input(data):
    # Load preprocessing objects
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    label_encoders = joblib.load('models/label_encoders.joblib')
    
    # Prepare the input data (example)
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    # Scale numerical features
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    
    return df

# Example usage
sample_data = {
    'feature1': value1,
    'feature2': value2,
    # ... add all required features
}

# Prepare the input
prepared_input = prepare_input(sample_data)

# Make prediction
prediction = model.predict(prepared_input)
probability = model.predict_proba(prepared_input)[:, 1]

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Churn Probability: {probability[0]:.2f}")
"""
    
    with open('models/prediction_example.py', 'w') as f:
        f.write(example_code) 