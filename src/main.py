import os
import joblib
from preprocessing import preprocess_pipeline
from model import ChurnPredictor
from utils import (
    plot_feature_importance,
    plot_roc_curve,
    plot_confusion_matrix,
    analyze_feature_correlations,
    plot_feature_distributions,
    save_model_report,
    create_prediction_example
)

def create_directories():
    """
    Create necessary directories if they don't exist
    """
    directories = ['models', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_preprocessing_objects(data):
    """
    Save preprocessing objects for later use
    """
    joblib.dump(data['scaler'], 'models/scaler.joblib')
    joblib.dump(data['label_encoders'], 'models/label_encoders.joblib')

def main():
    print("Starting Customer Churn Prediction Pipeline...")
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    print("\nPreprocessing data...")
    data = preprocess_pipeline("data/raw/customer_churn_data.csv")
    if data is None:
        print("Error: Could not load and preprocess data.")
        return
    
    # Save preprocessing objects
    save_preprocessing_objects(data)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest model...")
    predictor.train_random_forest(data['X_train'], data['y_train'])
    rf_results = predictor.evaluate_model(data['X_test'], data['y_test'])
    
    # Train and evaluate XGBoost
    print("\nTraining XGBoost model...")
    predictor.train_xgboost(data['X_train'], data['y_train'])
    xgb_results = predictor.evaluate_model(data['X_test'], data['y_test'])
    
    # Compare models and save the best one
    if xgb_results['roc_auc_score'] > rf_results['roc_auc_score']:
        print("\nXGBoost model performed better. Saving XGBoost model...")
        best_results = xgb_results
    else:
        print("\nRandom Forest model performed better. Saving Random Forest model...")
        best_results = rf_results
    
    # Save the best model
    predictor.save_model("models/best_model.joblib")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_feature_importance(predictor.feature_importance)
    plot_roc_curve(data['y_test'], predictor.model.predict_proba(data['X_test'])[:, 1])
    plot_confusion_matrix(best_results['confusion_matrix'])
    
    # Save model report
    print("\nSaving model report...")
    save_model_report(best_results, predictor.feature_importance)
    
    # Create prediction example
    print("\nCreating prediction example...")
    create_prediction_example(predictor.model, data['scaler'], data['label_encoders'])
    
    print("\nPipeline completed successfully!")
    print(f"Best model ROC AUC Score: {best_results['roc_auc_score']:.4f}")

if __name__ == "__main__":
    main() 