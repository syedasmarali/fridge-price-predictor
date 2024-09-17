# Fridge Price Prediction

This project focuses on predicting the price of refrigerators using regression analysis in Python. The main steps involved include analyzing several independent variables, cleaning and formatting the dataset, and building a user-friendly web app for interactive analysis.

## Project Overview

- **Data Preparation**: 
  - The dataset was cleaned and formatted to ensure proper use in regression analysis.
  - Non-numeric variables were converted into numeric values to make them compatible with the model.

- **Regression Analysis**: 
  - A regression model was developed to predict fridge prices based on the provided variables.

- **Streamlit Web App**: 
  - A dashboard was created using the Streamlit library.
  - The app allows for visualization of the regression model and provides an interface for predicting refrigerator prices.

## Features

- Data cleaning and formatting for compatibility with regression analysis.
- Development of a regression model for accurate price predictions.
- Interactive web app for data analysis and price prediction using Streamlit.

## Installation

- Clone the repository:
   ```bash
   git clone https://github.com/syedasmarali/fridge-price-predictor.git
   ```bash
   
- Navigate into the project directory:
  ```bash
  cd fridge-price-predictor

3. Create a virtual environment (optional but recommended):
  ```bash
  python -m venv .venv

4. Activate the virtual environment:
  - On Windows:
    ```bash
    python -m venv .venv
  - On macOS/Linux:
    ```bash
    Source .venv/bin/activate

5. Install the required packages:
  ```bash
  Source .venv/bin/activate

6. Run the streamlit app:
  ```bash
  streamlit run src/app.py


## Techhnologies Used

- Python: For data analysis and regression model.
- Streamlit: For building the web app and creating the dashboard.
- Pandas, NumPy, SKLearn: For data manipulation and analysis.
