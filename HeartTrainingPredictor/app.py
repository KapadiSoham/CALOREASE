import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data

st.set_page_config(
    page_title="Heart Rate & Calorie Duration Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Heart Rate & Calorie Duration Predictor")
st.markdown("""
This application analyzes the relationship between heart rate, calories burned, 
and exercise duration to predict how long someone needs to train to achieve specific goals.

**Features:**
- Data exploration with interactive visualizations
- Model training with multiple machine learning algorithms
- Model comparison to identify the best predictor
- Interactive prediction interface
""")

# Attempt to load data
try:
    df = load_data()
    
    # Display a sample of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display basic statistics
    st.subheader("Dataset Statistics")
    st.write(f"**Number of Records:** {df.shape[0]}")
    st.write(f"**Number of Features:** {df.shape[1]}")
    
    # Display navigation instructions
    st.markdown("""
    ## Navigation
    Use the sidebar to navigate to different sections of the application:
    - **Data Exploration**: Analyze the dataset with interactive visualizations
    - **Model Training**: Train various machine learning models
    - **Model Comparison**: Compare models and select the best one
    - **Prediction**: Make predictions with the trained models
    """)
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.markdown("""
    ### Troubleshooting:
    - Check that the Excel file is correctly placed in the 'attached_assets' folder
    - Ensure the Excel file format is compatible
    - Refresh the page and try again
    """)
