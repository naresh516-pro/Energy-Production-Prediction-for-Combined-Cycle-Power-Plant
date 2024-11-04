import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("C:/Users/Naresh/OneDrive/Desktop/energy prediction project/trained_model.sav", 'rb'))

# CSS for background image, font color, and layout enhancements
st.markdown(
    """
    <style>
    /* Full background with cover styling */
    .main {
        background-image: url('C:/Users/Naresh/OneDrive/Desktop/energy prediction project/background.jpg');
        background-size: cover;
        background-position: center;
        padding: 20px;
        font-family: Arial, sans-serif;
    }

    /* Title box with a semi-transparent background */
    .title-box {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: black;
        max-width: 700px;
        margin: auto;
    }

    /* Project description box */
    .description-box {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
        max-width: 700px;
        margin: 20px auto;
        color: black;
        text-align: justify;
    }

    /* Content box for input fields */
    .content-box {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
        max-width: 700px;
        margin: 20px auto;
        color: black;
    }

    /* Prediction result box */
    .result-box {
        background-color: rgba(0, 204, 102, 0.8);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 29px;
        color: white;
        max-width: 700px;
        margin: 20px auto;
    }

    /* Style for input headers */
    .input-header {
        font-weight: bold;
        font-size: 18px;
        margin-top: 10px;
        color: white;
    }

    /* Style for warning and info messages */
    .message-box {
        background-color: rgba(255, 255, 0, 0.3);
        padding: 15px;
        border-radius: 5px;
        color: black;
        font-size: 16px;
        max-width: 700px;
        margin: auto;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown(
    """
    <div class="title-box">
        <h1>🔍 Model Deployment for Energy Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Project description
st.markdown(
    """
    <div class="description-box">
        <h2>About the Project</h2>
        <p>This project is focused on predicting energy output based on various environmental factors. 
        Using a machine learning model trained on historical data, we analyze the impact of parameters 
        such as temperature, exhaust vacuum, ambient pressure, and relative humidity on energy production.</p>
        <p>The model deployed in this application leverages advanced regression techniques to provide 
        accurate predictions, assisting energy management and optimization efforts in industrial and commercial settings.</p>
        <p><strong>How it works:</strong> Simply enter the relevant environmental conditions below, 
        confirm your inputs, and the model will generate a prediction based on the values you provide. This information can 
        help in making informed decisions to optimize energy efficiency and reduce operational costs.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input parameters section
st.markdown(
    """
    <div class="content-box">
        <h2>Enter Input Parameters</h2>
        <p>Provide the following environmental conditions for an accurate prediction:</p>
    """,
    unsafe_allow_html=True
)

# Input fields with clear labels
st.markdown("<div class='input-header'>Temperature (°C)</div>", unsafe_allow_html=True)
temperature = st.number_input('', min_value=1.0, max_value=50.0, value=1.0, step=0.1, help="Enter the temperature in Celsius.")

st.markdown("<div class='input-header'>Exhaust Vacuum (in Hg)</div>", unsafe_allow_html=True)
exhaust_vacuum = st.number_input('', min_value=25.0, max_value=100.0, value=25.0, step=0.1, help="Enter the exhaust vacuum in inches of mercury.")

st.markdown("<div class='input-header'>Ambient Pressure (hPa)</div>", unsafe_allow_html=True)
amb_pressure = st.number_input('', min_value=900.0, max_value=1050.0, value=900.0, step=0.1, help="Enter the ambient pressure in hectopascals.")

st.markdown("<div class='input-header'>Relative Humidity (%)</div>", unsafe_allow_html=True)
r_humidity = st.slider('', 25.0, 100.0, 25.0, step=0.1, help="Adjust the relative humidity percentage.")

# Confirmation checkbox
if st.checkbox("Confirm Input Parameters"):
    # Prepare data for prediction
    input_data = pd.DataFrame([[temperature, exhaust_vacuum, amb_pressure, r_humidity]],
                              columns=['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity'])

    # Make the prediction
    prediction = loaded_model.predict(input_data)

    # Display the prediction result
    st.markdown(
        f"""
        <div class="result-box">
            <h3>Prediction Result:</h3>
            <p>The predicted energy output is: <strong>{prediction[0]}</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="message-box">
            Please confirm your input parameters to see the prediction.
        </div>
        """,
        unsafe_allow_html=True
    )
