# Stock Price Prediction Dashboard

This repository contains a Streamlit web application for predicting stock prices using a pre-trained LSTM model. The application allows users to select a stock, specify a date range, visualize historical stock data, and compare predicted prices with actual prices.  It also includes a basic login functionality.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Login:** Basic username/password authentication (admin/admin).  **Note:** This is for demonstration purposes only and should be replaced with a more secure authentication method in a production environment.
- **Stock Selection:** Users can choose from a list of stock tickers (defined in `stock_tickers.xlsx`).
- **Date Range Selection:** Users can specify the start and end dates for the data.
- **Data Visualization:** The application displays charts of historical closing prices, including 100-day and 200-day moving averages.
- **Price Prediction:** The pre-trained LSTM model predicts future stock prices based on historical data.
- **Prediction Visualization:**  The application plots the predicted prices against the actual prices for comparison.

## Installation

1. **Clone the repository:** {}Optional{} **

   ```bash:
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME


2. **Create a virtual environment (recommended):**

Bash:
python3 -m venv venv  # For Linux/macOS
venv\Scripts\activate  # For Windows


3. **Install the required packages:**

Bash:
pip install -r requirements.txt

4. **Prepare the data:**

Create an Excel file named stock_tickers.xlsx with a column named "Symbol" containing the stock tickers you want to use.

Download or train your own stock price prediction model and save it as stock_prediction_model.h5 in the same directory as the Python script. (See Model Training for guidance.)

5. Usage
Run the Streamlit app:

Bash:
streamlit run appv2.3.1.py

Open the app in your web browser: Streamlit will provide a URL (usually http://localhost:8501) that you can use to access the application.

Log in: Use the default credentials (username: admin, password: admin).

Use the dashboard: Select a stock, specify the date range, and view the visualizations and price predictions.