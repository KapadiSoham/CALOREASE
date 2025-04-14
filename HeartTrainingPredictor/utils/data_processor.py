import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pandas.DataFrame): The input dataset
        
    Returns:
        pandas.DataFrame: The cleaned dataset
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Drop rows with missing values
    df_clean = df_clean.dropna()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle outliers using IQR method for numerical columns
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers (replace with boundaries)
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    return df_clean

def prepare_features_target(df, target_col='duration'):
    """
    Prepare features (X) and target (y) variables.
    
    Args:
        df (pandas.DataFrame): The input dataset
        target_col (str): The name of the target column
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset")
    
    # Select features (all columns except target and non-predictive columns)
    non_predictive = ['user_id']  # Columns that shouldn't be used for prediction
    drop_cols = [target_col] + [col for col in non_predictive if col in df.columns]
    
    # For gender, we want to exclude it from features based on user preference
    if 'gender' in df.columns:
        drop_cols.append('gender')
    
    X = df.drop(columns=drop_cols)
    
    # Select target
    y = df[target_col]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler

@st.cache_data
def process_data(df, target_col='duration', test_size=0.2, random_state=42):
    """
    Complete data processing pipeline.
    
    Args:
        df (pandas.DataFrame): The input dataset
        target_col (str): The name of the target column
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Processed data components
    """
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare features and target
    X, y = prepare_features_target(df_clean, target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'df_clean': df_clean,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
