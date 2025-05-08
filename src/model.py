import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
from preprocessing import preprocess_pipeline

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
    
    def train_random_forest(self, X_train, y_train):
        """
        Train a Random Forest model with hyperparameter tuning
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize base model
        rf = RandomForestClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='roc_auc',
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return self
    
    def train_xgboost(self, X_train, y_train):
        """
        Train an XGBoost model with hyperparameter tuning
        """
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Initialize base model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='roc_auc',
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return self
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba)
        }
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model
        """
        return joblib.load(filepath)

def main():
    # Load and preprocess data
    data = preprocess_pipeline("data/raw/customer_churn_data.csv")
    if data is None:
        return
    
    # Initialize and train model
    predictor = ChurnPredictor()
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    predictor.train_random_forest(data['X_train'], data['y_train'])
    rf_results = predictor.evaluate_model(data['X_test'], data['y_test'])
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    predictor.train_xgboost(data['X_train'], data['y_train'])
    xgb_results = predictor.evaluate_model(data['X_test'], data['y_test'])
    
    # Print results
    print("\nRandom Forest Results:")
    print(rf_results['classification_report'])
    print(f"ROC AUC Score: {rf_results['roc_auc_score']:.4f}")
    
    print("\nXGBoost Results:")
    print(xgb_results['classification_report'])
    print(f"ROC AUC Score: {xgb_results['roc_auc_score']:.4f}")
    
    # Save the best model
    predictor.save_model("models/best_model.joblib")
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    print(predictor.feature_importance.head(10))

if __name__ == "__main__":
    main() 