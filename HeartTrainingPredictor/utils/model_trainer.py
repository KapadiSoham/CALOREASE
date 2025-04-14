import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Train a Linear Regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        dict: Model, predictions, and metrics
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_train, train_preds, y_test, test_preds)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'metrics': metrics
    }

def train_ridge_regression(X_train, y_train, X_test, y_test, alpha=1.0):
    """
    Train a Ridge Regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        alpha: Regularization strength
        
    Returns:
        dict: Model, predictions, and metrics
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_train, train_preds, y_test, test_preds)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'metrics': metrics
    }

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None):
    """
    Train a Random Forest Regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        
    Returns:
        dict: Model, predictions, and metrics
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_train, train_preds, y_test, test_preds)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'metrics': metrics,
        'feature_importance': model.feature_importances_
    }

def train_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1):
    """
    Train a Gradient Boosting Regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        n_estimators: Number of boosting stages
        learning_rate: Shrinks the contribution of each tree
        
    Returns:
        dict: Model, predictions, and metrics
    """
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_train, train_preds, y_test, test_preds)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'metrics': metrics,
        'feature_importance': model.feature_importances_
    }

def train_svr(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf'):
    """
    Train a Support Vector Regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        C: Regularization parameter
        kernel: Kernel type
        
    Returns:
        dict: Model, predictions, and metrics
    """
    model = SVR(C=C, kernel=kernel)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_train, train_preds, y_test, test_preds)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'metrics': metrics
    }

def calculate_metrics(y_train, train_preds, y_test, test_preds):
    """
    Calculate regression metrics.
    
    Args:
        y_train, train_preds: Training data and predictions
        y_test, test_preds: Testing data and predictions
        
    Returns:
        dict: Dictionary of metrics
    """
    # Training metrics
    train_mse = mean_squared_error(y_train, train_preds)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)
    
    # Testing metrics
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }

@st.cache_resource
def train_all_models(_data_dict):
    """
    Train all regression models.
    
    Args:
        _data_dict: Dictionary containing processed data (underscore prefix to avoid hashing)
        
    Returns:
        dict: Dictionary of trained models and metrics
    """
    X_train_scaled = _data_dict['X_train_scaled']
    y_train = _data_dict['y_train']
    X_test_scaled = _data_dict['X_test_scaled']
    y_test = _data_dict['y_test']
    
    # Train models
    lr_results = train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    ridge_results = train_ridge_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    rf_results = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    gb_results = train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test)
    
    return {
        'Linear Regression': lr_results,
        'Ridge Regression': ridge_results,
        'Random Forest': rf_results,
        'Gradient Boosting': gb_results
    }

def predict_with_model(model, scaler, input_data, feature_names):
    """
    Make a prediction with a trained model.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        input_data: Input data as a dictionary
        feature_names: List of feature names
        
    Returns:
        float: Predicted value
    """
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def get_best_model(model_results):
    """
    Determine the best model based on test R2 score.
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        tuple: (best_model_name, best_model_results)
    """
    best_model_name = None
    best_r2 = -float('inf')
    
    for model_name, results in model_results.items():
        test_r2 = results['metrics']['test_r2']
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_name = model_name
    
    return best_model_name, model_results[best_model_name]
