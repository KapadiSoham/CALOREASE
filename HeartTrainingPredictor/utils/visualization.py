import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_correlation_heatmap(df):
    """
    Create a correlation heatmap.
    
    Args:
        df (pandas.DataFrame): The input dataset
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix"
    )
    
    return fig

def plot_feature_distributions(df):
    """
    Create histograms for each numerical feature.
    
    Args:
        df (pandas.DataFrame): The input dataset
        
    Returns:
        plotly.graph_objects.Figure: Histograms of features
    """
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a subplot for each numerical column
    fig = make_subplots(
        rows=len(numeric_cols), 
        cols=1,
        subplot_titles=[f"Distribution of {col}" for col in numeric_cols]
    )
    
    # Add histogram for each column
    for i, col in enumerate(numeric_cols):
        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 * len(numeric_cols),
        title_text="Feature Distributions",
        showlegend=False
    )
    
    return fig

def plot_pairplot(df, target_col='duration'):
    """
    Create a pairplot of the dataset.
    
    Args:
        df (pandas.DataFrame): The input dataset
        target_col (str): The name of the target column
        
    Returns:
        plotly.graph_objects.Figure: Pairplot
    """
    # Create a pairplot with Plotly
    fig = px.scatter_matrix(
        df,
        dimensions=[col for col in df.columns if col != target_col],
        color=target_col,
        title="Feature Relationships",
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=900
    )
    
    return fig

def plot_feature_importance(model_results, feature_names):
    """
    Create a feature importance plot.
    
    Args:
        model_results (dict): Results from a trained model
        feature_names (list): List of feature names
        
    Returns:
        plotly.graph_objects.Figure: Feature importance bar chart
    """
    # Check if the model has feature importance
    if 'feature_importance' not in model_results:
        return None
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model_results['feature_importance']
    }).sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance"
    )
    
    return fig

def plot_prediction_vs_actual(y_test, test_preds, model_name):
    """
    Create a scatter plot of predicted vs actual values.
    
    Args:
        y_test (pandas.Series): Actual values
        test_preds (numpy.ndarray): Predicted values
        model_name (str): Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot
    """
    # Create a DataFrame with actual and predicted values
    result_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_preds
    })
    
    # Create scatter plot
    fig = px.scatter(
        result_df,
        x='Actual',
        y='Predicted',
        title=f"Actual vs Predicted ({model_name})"
    )
    
    # Add perfect prediction line
    min_val = min(result_df['Actual'].min(), result_df['Predicted'].min())
    max_val = max(result_df['Actual'].max(), result_df['Predicted'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    
    return fig

def plot_residuals(y_test, test_preds, model_name):
    """
    Create a plot of residuals.
    
    Args:
        y_test (pandas.Series): Actual values
        test_preds (numpy.ndarray): Predicted values
        model_name (str): Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Residual plot
    """
    # Calculate residuals
    residuals = y_test - test_preds
    
    # Create a DataFrame with predictions and residuals
    result_df = pd.DataFrame({
        'Predicted': test_preds,
        'Residuals': residuals
    })
    
    # Create scatter plot
    fig = px.scatter(
        result_df,
        x='Predicted',
        y='Residuals',
        title=f"Residual Plot ({model_name})"
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Zero Residual",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Predicted Values",
        yaxis_title="Residuals"
    )
    
    return fig

def plot_metrics_comparison(model_results):
    """
    Create a bar chart comparing models.
    
    Args:
        model_results (dict): Dictionary of model results
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of model comparison
    """
    # Extract metrics for each model
    metrics_data = []
    
    for model_name, results in model_results.items():
        metrics = results['metrics']
        metrics_data.append({
            'Model': model_name,
            'Train RMSE': metrics['train_rmse'],
            'Test RMSE': metrics['test_rmse'],
            'Train R²': metrics['train_r2'],
            'Test R²': metrics['test_r2']
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["RMSE (lower is better)", "R² Score (higher is better)"]
    )
    
    # Add RMSE comparison
    fig.add_trace(
        go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Train RMSE'],
            name='Train RMSE',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Test RMSE'],
            name='Test RMSE',
            marker_color='darkblue'
        ),
        row=1, col=1
    )
    
    # Add R² comparison
    fig.add_trace(
        go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Train R²'],
            name='Train R²',
            marker_color='lightgreen'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Test R²'],
            name='Test R²',
            marker_color='darkgreen'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Model Performance Comparison",
        legend_title="Metric",
        height=500,
        width=900
    )
    
    return fig

def plot_heart_rate_vs_calories(df):
    """
    Create a scatter plot of heart rate vs calories.
    
    Args:
        df (pandas.DataFrame): The input dataset
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot
    """
    # Identify the heart rate and calories columns
    heart_rate_col = [col for col in df.columns if 'heart' in col.lower() and 'rate' in col.lower()]
    calories_col = [col for col in df.columns if 'calorie' in col.lower()]
    duration_col = 'duration' if 'duration' in df.columns else None
    
    # If columns are found, create the plot
    if heart_rate_col and calories_col:
        hr_col = heart_rate_col[0]
        cal_col = calories_col[0]
        
        fig = px.scatter(
            df,
            x=hr_col,
            y=cal_col,
            color=duration_col,
            title=f"{hr_col.title()} vs {cal_col.title()}",
            trendline="ols"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=hr_col.title(),
            yaxis_title=cal_col.title(),
            coloraxis_colorbar_title="Training Duration (minutes)"
        )
        
        return fig
    
    # Return None if columns are not found
    return None
