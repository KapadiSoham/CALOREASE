import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.model_trainer import predict_with_model

st.set_page_config(
    page_title="Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("Prediction Tool")
st.markdown("""
Use this page to make predictions with your trained models:
- Predict **Exercise Duration** based on heart rate and other parameters, or
- Predict **Heart Rate** required to achieve certain duration and calorie targets

The available prediction type depends on which target variable you selected during model training.
""")

# Check if models are trained
if 'model_results' not in st.session_state or 'data_dict' not in st.session_state:
    st.warning("No models have been trained yet. Please go to the 'Model Training' page first.")
    st.stop()

# Get model results and data
model_results = st.session_state.model_results
data_dict = st.session_state.data_dict

# Load original dataset to get feature ranges
try:
    df = load_data()
    df_clean = data_dict['df_clean']
    feature_names = data_dict['feature_names']
    scaler = data_dict['scaler']
    
    # Get feature statistics for input validation
    feature_stats = {}
    for feature in feature_names:
        feature_stats[feature] = {
            'min': df_clean[feature].min(),
            'max': df_clean[feature].max(),
            'mean': df_clean[feature].mean(),
            'std': df_clean[feature].std()
        }
    
    # Prediction interface
    st.subheader("Enter Parameters")
    
    # Create multiple columns for inputs
    num_cols = 2
    cols = st.columns(num_cols)
    
    # Create input fields for each feature
    input_values = {}
    for i, feature in enumerate(feature_names):
        col_idx = i % num_cols
        with cols[col_idx]:
            # Use feature statistics to set min, max and default values
            min_val = feature_stats[feature]['min']
            max_val = feature_stats[feature]['max']
            mean_val = feature_stats[feature]['mean']
            
            # Create slider for each feature
            input_values[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}:",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100)
            )
    
    # Model selection
    st.subheader("Select Model")
    model_options = list(model_results.keys())
    
    # Add "Best Model" option if available
    if 'best_model_name' in st.session_state:
        model_options = ["Best Model"] + model_options
    
    selected_model_name = st.selectbox("Choose Model:", model_options)
    
    # Get the appropriate model
    if selected_model_name == "Best Model":
        model_name = st.session_state.best_model_name
        model_result = st.session_state.best_model
    else:
        model_name = selected_model_name
        model_result = model_results[model_name]
    
    # Determine which target variable is being predicted
    target_variable = data_dict['y'].name
    
    # Set prediction button label and metric label based on target variable
    if target_variable.lower() == 'duration':
        predict_button_label = "Predict Training Duration"
        result_metric_label = "Predicted Duration (minutes)"
        result_unit = "minutes"
    elif target_variable.lower() == 'heart_rate':
        predict_button_label = "Predict Heart Rate"
        result_metric_label = "Predicted Heart Rate (bpm)"
        result_unit = "bpm"
    else:
        predict_button_label = "Make Prediction"
        result_metric_label = f"Predicted {target_variable}"
        result_unit = "units"
    
    # Make prediction button
    if st.button(predict_button_label):
        with st.spinner("Making prediction..."):
            # Get the model
            model = model_result['model']
            
            # Make prediction
            prediction = predict_with_model(model, scaler, input_values, feature_names)
            
            # Display prediction with nice formatting
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(result_metric_label, f"{prediction:.2f}")
            
            with col2:
                # Calculate confidence range (simplified)
                rmse = model_result['metrics']['test_rmse']
                lower_bound = max(0, prediction - rmse)
                upper_bound = prediction + rmse
                
                st.write(f"**Model Used:** {model_name}")
                st.write(f"**Confidence Range:** {lower_bound:.2f} to {upper_bound:.2f} {result_unit}")
                st.write(f"**Model Accuracy (R²):** {model_result['metrics']['test_r2']:.4f}")
    
    # Additional information
    with st.expander("Interpretation Guide", expanded=False):
        # Different guides based on target variable
        if target_variable.lower() == 'duration':
            st.write("""
            ### How to Interpret the Results
            
            The prediction represents the estimated duration (in minutes) someone would need to exercise to achieve 
            the specified heart rate and calorie goals based on the historical data patterns.
            
            **Confidence Range:** This represents a simplified estimation of prediction uncertainty, 
            calculated using the model's RMSE (Root Mean Square Error). The actual value is likely to 
            fall within this range.
            
            **Model Accuracy (R²):** This value ranges from 0 to 1, with higher values indicating better model fit. 
            It represents the proportion of variance in the training duration that is predictable from the features.
            
            ### Tips for Accurate Predictions
            
            - Stay within realistic ranges for all input parameters
            - The model is most accurate within the ranges of the training data
            - Consider using the "Best Model" for the most reliable predictions
            """)
        elif target_variable.lower() == 'heart_rate':
            st.write("""
            ### How to Interpret the Results
            
            The prediction represents the estimated heart rate (in beats per minute) that would be achieved
            during exercise with the specified duration and calorie burn based on historical data patterns.
            
            **Confidence Range:** This represents a simplified estimation of prediction uncertainty, 
            calculated using the model's RMSE (Root Mean Square Error). The actual value is likely to 
            fall within this range.
            
            **Model Accuracy (R²):** This value ranges from 0 to 1, with higher values indicating better model fit. 
            It represents the proportion of variance in heart rate that is predictable from the features.
            
            ### Tips for Accurate Predictions
            
            - Stay within realistic ranges for all input parameters
            - The model is most accurate within the ranges of the training data
            - Consider using the "Best Model" for the most reliable predictions
            """)
        else:
            st.write("""
            ### How to Interpret the Results
            
            The prediction represents the estimated value based on the input parameters and historical data patterns.
            
            **Confidence Range:** This represents a simplified estimation of prediction uncertainty, 
            calculated using the model's RMSE (Root Mean Square Error). The actual value is likely to 
            fall within this range.
            
            **Model Accuracy (R²):** This value ranges from 0 to 1, with higher values indicating better model fit. 
            It represents the proportion of variance in the target variable that is predictable from the features.
            
            ### Tips for Accurate Predictions
            
            - Stay within realistic ranges for all input parameters
            - The model is most accurate within the ranges of the training data
            - Consider using the "Best Model" for the most reliable predictions
            """)

except Exception as e:
    st.error(f"Error in prediction page: {str(e)}")
    st.warning("Please check the dataset and model parameters.")
