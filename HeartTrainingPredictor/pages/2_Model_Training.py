import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.data_processor import process_data
from utils.model_trainer import train_all_models
from utils.visualization import plot_feature_importance, plot_prediction_vs_actual, plot_residuals

st.set_page_config(
    page_title="Model Training",
    page_icon="🔬",
    layout="wide"
)

st.title("Model Training")
st.markdown("""
This page allows you to train various machine learning models to predict either:
1. Exercise **Duration** based on heart rate and other factors, or
2. **Heart Rate** based on duration and other factors

Select your target variable below and train the models to see which performs best.
""")

# Load and process data
try:
    # Load data
    df = load_data()
    
    # Process data
    with st.expander("Data Processing", expanded=True):
        st.subheader("Data Processing Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox(
                "Target Column:",
                options=[col for col in df.columns if col.lower() in ['duration', 'heart_rate']],
                index=0
            )
        
        with col2:
            test_size = st.slider("Test Set Size:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
        with col3:
            random_state = st.number_input("Random Seed:", min_value=1, max_value=100, value=42, step=1)
        
        # Process the data
        data_dict = process_data(df, target_col=target_col, test_size=test_size, random_state=random_state)
        
        # Display training and testing shapes
        st.write(f"Training set shape: {data_dict['X_train'].shape}")
        st.write(f"Testing set shape: {data_dict['X_test'].shape}")
    
    # Train models
    with st.expander("Model Training", expanded=True):
        st.subheader("Train Models")
        
        # Train all models
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                model_results = train_all_models(data_dict)
                st.session_state.model_results = model_results
                st.session_state.data_dict = data_dict
                st.success("Models trained successfully!")
        
        # If models are trained
        if 'model_results' in st.session_state:
            # Display the models
            model_names = list(st.session_state.model_results.keys())
            
            # Select model to view
            selected_model = st.selectbox("Select Model:", model_names)
            model_result = st.session_state.model_results[selected_model]
            
            # Display metrics
            metrics = model_result['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train RMSE", f"{metrics['train_rmse']:.4f}")
            with col2:
                st.metric("Test RMSE", f"{metrics['test_rmse']:.4f}")
            with col3:
                st.metric("Train R²", f"{metrics['train_r2']:.4f}")
            with col4:
                st.metric("Test R²", f"{metrics['test_r2']:.4f}")
            
            # Show feature importance if available
            if 'feature_importance' in model_result:
                st.subheader("Feature Importance")
                importance_fig = plot_feature_importance(model_result, data_dict['feature_names'])
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Show prediction vs actual
            st.subheader("Predictions vs Actual Values")
            pred_vs_actual = plot_prediction_vs_actual(
                data_dict['y_test'], 
                model_result['test_predictions'], 
                selected_model
            )
            st.plotly_chart(pred_vs_actual, use_container_width=True)
            
            # Show residuals
            st.subheader("Residuals Plot")
            residuals_fig = plot_residuals(
                data_dict['y_test'], 
                model_result['test_predictions'], 
                selected_model
            )
            st.plotly_chart(residuals_fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error in model training: {str(e)}")
    st.warning("Please check the dataset and model parameters.")
