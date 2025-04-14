import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from utils.data_loader import load_data
from utils.data_processor import clean_data, process_data
from utils.visualization import (
    plot_correlation_heatmap, 
    plot_feature_distributions, 
    plot_pairplot,
    plot_heart_rate_vs_calories
)

st.set_page_config(
    page_title="Data Exploration",
    page_icon="📊",
    layout="wide"
)

st.title("Data Exploration")
st.markdown("""
This page provides exploratory data analysis of the heart rate and calorie dataset.
Explore the distribution of features, correlations, and relationships between variables.
""")

# Load data
try:
    df = load_data()
    
    # Data cleaning
    with st.expander("Data Cleaning", expanded=False):
        st.subheader("Original Data")
        st.dataframe(df.head())
        st.write(f"Original data shape: {df.shape}")
        
        # Display data types and missing values
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            st.write(df.dtypes)
        
        with col2:
            st.subheader("Missing Values")
            missing_values = df.isnull().sum()
            st.write(missing_values)
        
        # Clean data
        df_clean = clean_data(df)
        
        st.subheader("Cleaned Data")
        st.dataframe(df_clean.head())
        st.write(f"Cleaned data shape: {df_clean.shape}")
        st.write(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
    
    # Data statistics
    with st.expander("Data Statistics", expanded=True):
        st.subheader("Summary Statistics")
        st.write(df_clean.describe())
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr_fig = plot_correlation_heatmap(df_clean)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # Feature distributions
    with st.expander("Feature Distributions", expanded=True):
        st.subheader("Distribution of Features")
        dist_fig = plot_feature_distributions(df_clean)
        st.plotly_chart(dist_fig, use_container_width=True)
    
    # Feature relationships
    with st.expander("Feature Relationships", expanded=True):
        st.subheader("Pairwise Relationships")
        pair_fig = plot_pairplot(df_clean)
        st.plotly_chart(pair_fig, use_container_width=True)
        
        # Heart rate vs calories
        st.subheader("Heart Rate vs Calories")
        hr_cal_fig = plot_heart_rate_vs_calories(df_clean)
        if hr_cal_fig:
            st.plotly_chart(hr_cal_fig, use_container_width=True)
        else:
            st.warning("Could not identify heart rate and calorie columns for plotting")
    
    # Custom visualization
    with st.expander("Custom Visualization", expanded=True):
        st.subheader("Customize Your Visualization")
        
        # Select columns for customized plotting
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", df_clean.columns)
        with col2:
            y_col = st.selectbox("Y-axis:", df_clean.columns, index=1 if len(df_clean.columns) > 1 else 0)
        
        color_col = st.selectbox("Color by:", [None] + list(df_clean.columns))
        
        # Create customized scatter plot
        if color_col:
            fig = px.scatter(df_clean, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col} (colored by {color_col})")
        else:
            fig = px.scatter(df_clean, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        
        # Add trendline if requested
        add_trendline = st.checkbox("Add Trendline")
        if add_trendline:
            fig = px.scatter(df_clean, x=x_col, y=y_col, color=color_col if color_col else None, trendline="ols", title=f"{x_col} vs {y_col}")
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error in data exploration: {str(e)}")
    st.warning("Please check the dataset and try again.")
