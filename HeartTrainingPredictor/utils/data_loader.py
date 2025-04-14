import pandas as pd
import streamlit as st
import io
import requests
import os

@st.cache_data(ttl=3600)
def load_data():
    """
    Load data from the local Excel file or Google Sheets URL.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # First try to load from local Excel file
    excel_path = "attached_assets/ML_dataset_extended.xlsx"
    
    try:
        if os.path.exists(excel_path):
            # Load the Excel file
            data = pd.read_excel(excel_path, engine='openpyxl')
            
            # Clean column names (remove whitespace, lowercase)
            data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
            
            # Return data without gender column
            if 'gender' in data.columns:
                data = data.drop(columns=['gender'])
            
            return data
        else:
            # Fall back to Google Sheets URL if local file doesn't exist
            sheet_id = "1miEMhxToe7AR_hdtYhY8Brr1uJOqpUK4"
            sheet_name = "1796230041"
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_name}"
            
            # Download the file
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Read the CSV data
            data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            
            # Clean column names (remove whitespace, lowercase)
            data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
            
            # Return data without gender column
            if 'gender' in data.columns:
                data = data.drop(columns=['gender'])
            
            return data
    
    except pd.errors.ParserError as e:
        raise Exception(f"Failed to parse the data: {e}")
    
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")
