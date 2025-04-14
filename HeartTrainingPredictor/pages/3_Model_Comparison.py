import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.data_processor import process_data
from utils.model_trainer import train_all_models, get_best_model
from utils.visualization import plot_metrics_comparison

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Model Comparison")
st.markdown("""
This page compares the performance of different machine learning models to identify
the best algorithm for predicting exercise duration.
""")

# Check if models are trained
if 'model_results' not in st.session_state:
    st.warning("No models have been trained yet. Please go to the 'Model Training' page first.")
    st.stop()

# Get model results and data
model_results = st.session_state.model_results
data_dict = st.session_state.data_dict

# Display model comparison
with st.container():
    st.subheader("Model Performance Comparison")
    
    # Create comparison metrics table
    metrics_df = pd.DataFrame({
        'Model': [],
        'Train RMSE': [],
        'Test RMSE': [],
        'Train MAE': [],
        'Test MAE': [],
        'Train R²': [],
        'Test R²': []
    })
    
    # Populate metrics
    for model_name, results in model_results.items():
        metrics = results['metrics']
        new_row = pd.DataFrame({
            'Model': [model_name],
            'Train RMSE': [metrics['train_rmse']],
            'Test RMSE': [metrics['test_rmse']],
            'Train MAE': [metrics['train_mae']],
            'Test MAE': [metrics['test_mae']],
            'Train R²': [metrics['train_r2']],
            'Test R²': [metrics['test_r2']]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    # Sort by test R² (descending)
    metrics_df = metrics_df.sort_values('Test R²', ascending=False)
    
    # Display metrics table
    st.dataframe(metrics_df.style.highlight_max(subset=['Test R²']).highlight_min(subset=['Test RMSE']), use_container_width=True)
    
    # Plot metrics comparison
    comparison_fig = plot_metrics_comparison(model_results)
    st.plotly_chart(comparison_fig, use_container_width=True)

# Best model analysis
with st.container():
    st.subheader("Best Model Analysis")
    
    # Get best model
    best_model_name, best_model = get_best_model(model_results)
    
    st.write(f"**Best Model: {best_model_name}**")
    st.write(f"Test R² Score: {best_model['metrics']['test_r2']:.4f}")
    st.write(f"Test RMSE: {best_model['metrics']['test_rmse']:.4f}")
    
    # Model strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strengths")
        if best_model_name == "Linear Regression":
            st.write("- Simple and interpretable model")
            st.write("- Fast training and prediction time")
            st.write("- Good for understanding feature relationships")
        elif best_model_name == "Ridge Regression":
            st.write("- Regularized model that prevents overfitting")
            st.write("- Handles multicollinearity well")
            st.write("- Stable coefficient estimates")
        elif best_model_name == "Random Forest":
            st.write("- Captures non-linear relationships")
            st.write("- Robust to outliers")
            st.write("- Provides feature importance")
        elif best_model_name == "Gradient Boosting":
            st.write("- High predictive accuracy")
            st.write("- Captures complex patterns")
            st.write("- Provides feature importance")
    
    with col2:
        st.subheader("Limitations")
        if best_model_name == "Linear Regression":
            st.write("- Assumes linear relationship between features")
            st.write("- Sensitive to outliers")
            st.write("- May underfit complex data")
        elif best_model_name == "Ridge Regression":
            st.write("- Still assumes linear relationships")
            st.write("- Requires tuning of regularization parameter")
            st.write("- May underfit complex data")
        elif best_model_name == "Random Forest":
            st.write("- Less interpretable than linear models")
            st.write("- May overfit on noisy data")
            st.write("- Requires more memory and computation")
        elif best_model_name == "Gradient Boosting":
            st.write("- Less interpretable than linear models")
            st.write("- Sensitive to hyperparameter settings")
            st.write("- More prone to overfitting")
    
    # Save best model to session state
    st.session_state.best_model_name = best_model_name
    st.session_state.best_model = best_model

# Hyperparameter tuning suggestion
with st.expander("Hyperparameter Tuning Suggestion", expanded=False):
    st.write("""
    To further improve model performance, consider tuning the hyperparameters of the best model:
    """)
    
    if best_model_name == "Linear Regression":
        st.write("- Try different feature selection methods")
        st.write("- Consider polynomial features for non-linear relationships")
    elif best_model_name == "Ridge Regression":
        st.write("- Tune the alpha parameter (regularization strength)")
        st.write("- Try different solvers (e.g., 'sag', 'lsqr')")
    elif best_model_name == "Random Forest":
        st.write("- Tune n_estimators (number of trees)")
        st.write("- Adjust max_depth to control model complexity")
        st.write("- Modify min_samples_split and min_samples_leaf")
    elif best_model_name == "Gradient Boosting":
        st.write("- Tune learning_rate (smaller values often work better)")
        st.write("- Adjust n_estimators (more boosting stages)")
        st.write("- Modify max_depth to control model complexity")
