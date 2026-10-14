import time
from fpdf import FPDF
from textblob import TextBlob  # For sentiment analysis
import ta  # Technical analysis library
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
import datetime
from polygon import RESTClient
import yfinance as yf
import pandas as pd
import numpy as np
import os
# Disable OneDNN optimization for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import necessary libraries

# ----------------------
# DATA FETCHING FUNCTIONS
# ----------------------


def safe_yf_download(ticker, start, end, retries=3):
    """
    Download stock data with retries to handle network issues.
    """
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=False)
            if not data.empty:
                # Fix column names - IMPORTANT: don't merge ticker name into column
                data.columns = [col[0] if isinstance(
                    col, tuple) else col for col in data.columns]
                # Standardize column names
                standard_columns = {'Adj Close': 'Adj_Close'}
                data = data.rename(columns=standard_columns)
                return data
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)  # Wait before retrying

    st.error(f"Failed to fetch data for {ticker} after {retries} attempts.")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stock_data_polygon(ticker, start, end, api_key):
    """Fetch OHLCV from Polygon REST API as a pandas DataFrame."""
    if not api_key:
        st.error("Polygon API key is required when using Polygon.io data.")
        return pd.DataFrame()

    try:
        # Polygon API expects ticker without ".NS" for US symbols (e.g. AAPL)
        client = RESTClient(api_key)
        from_date = start if isinstance(
            start, str) else start.strftime('%Y-%m-%d')
        to_date = end if isinstance(end, str) else end.strftime('%Y-%m-%d')

        bars = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date
        )

        data = pd.DataFrame([bar.__dict__ for bar in bars])
        if data.empty:
            return pd.DataFrame()

        # Polygon bars can have different key names, normalize
        if 'timestamp' in data.columns:
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
        elif 't' in data.columns:
            data['datetime'] = pd.to_datetime(data['t'], unit='ms')

        if 'o' in data.columns:
            data.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low',
                        'c': 'Close', 'v': 'Volume'}, inplace=True)
        if 'open' in data.columns:
            data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)

        if 'datetime' in data.columns:
            data = data.set_index('datetime').sort_index()

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Polygon data is missing expected column: {col}")
                return pd.DataFrame()

        return data[required_cols]

    except Exception as e:
        st.error(f"Error with Polygon.io: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_tickers(filepath='stock_tickers.xlsx'):
    """
    Load stock tickers from Excel file.
    """
    try:
        tickers_df = pd.read_excel(filepath)
        return tickers_df['Symbol']
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        # Return some default tickers if file loading fails
        return pd.Series(['RELIANCE', 'INFY', 'TCS', 'TATAMOTORS', 'HDFCBANK'])


@st.cache_data(show_spinner=False)
def load_stock_data(stock, start, end):
    """
    Load and preprocess stock data with technical indicators.
    """
    data = safe_yf_download(stock, start, end)
    if not data.empty:
        # Debug information
        st.write(f"Columns in raw data: {data.columns.tolist()}")

        # Add technical indicators
        data = add_indicators(data)

        # Add debug after indicators added
        st.write(f"Columns after adding indicators: {data.columns.tolist()}")
    return data

# ----------------------
# DATA PROCESSING FUNCTIONS
# ----------------------


def add_indicators(df):
    """
    Add technical indicators to the dataframe.
    """
    if df.empty:
        return df

    df = df.copy()

    try:
        # Verify required column exists
        if 'Close' not in df.columns:
            st.error("No Close price column found. Cannot add indicators.")
            return df

        # Add technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd_indicator = ta.trend.MACD(df['Close'])
        df['MACD'] = macd_indicator.macd_diff()
        bb = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()

        # Add moving averages
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()

        # Fill NaN values that might be created by indicators
        df = df.bfill().ffill()

    except Exception as e:
        st.error(f"Error adding indicators: {e}")
        import traceback
        st.error(traceback.format_exc())

    return df

# ----------------------
# MODEL FUNCTIONS
# ----------------------


def get_model(name):
    """
    Return a prediction model based on user selection.
    """
    try:
        if name == "LSTM":
            try:
                return load_model('stock_prediction_model.h5')
            except Exception as e:
                st.error(
                    f"Error loading LSTM model: {e}. Using Linear Regression instead.")
                return LinearRegression()
        elif name == "Linear Regression":
            return LinearRegression()
        elif name == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif name == "XGBoost":
            return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            st.error("Model not implemented")
            return LinearRegression()  # Default fallback
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return LinearRegression()  # Default fallback


def create_features(data, lookback=5):

    if 'Close' not in data.columns or data.empty:
        return None, None

    # Create features
    features = pd.DataFrame(index=data.index)
    features['Close'] = data['Close']

    # 🔥 Add sentiment
    features['Sentiment'] = get_news_sentiment("Reliance")

    # Add previous days' closing prices
    for i in range(1, lookback + 1):
        features[f'Close_lag_{i}'] = data['Close'].shift(i)

    # Add technical indicators if available
    for col in ['RSI', 'MACD']:
        if col in data.columns:
            features[col] = data[col]

    # Remove NaN values
    features = features.dropna()

    # Split into X and y
    X = features.drop('Close', axis=1)
    y = features['Close']

    return X, y


def predict_classical(model, data, test_size=0.2):
    """
    Train and predict using a classical ML model with improved features.
    """
    try:
        if 'Close' not in data.columns or data.empty:
            st.error("No Close price data available for prediction")
            return np.array([]), np.array([]), float('inf')

        # Create features with a lookback period
        X, y = create_features(data)
        if X is None or y is None:
            st.error("Failed to create features for prediction")
            return np.array([]), np.array([]), float('inf')

        # Split data
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        if len(X_train) < 10 or len(X_test) < 1:
            st.error(
                f"Insufficient data for prediction: train={len(X_train)}, test={len(X_test)}")
            return np.array([]), np.array([]), float('inf')

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate error
        mse = mean_squared_error(y_test, y_pred)

        return y_test.values, y_pred, mse

    except Exception as e:
        st.error(f"Error in classical model prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return np.array([]), np.array([]), float('inf')


def predict_lstm(model, data, scaler, sequence_length=60):
    """
    Prepare sequences and predict using an LSTM model.
    """
    try:
        # If we don't have enough data or an LSTM model, fall back to linear regression
        if len(data) < sequence_length * 2 or 'Close' not in data.columns:
            st.warning(
                "Insufficient data for LSTM prediction. Using linear regression instead.")
            lr_model = LinearRegression()
            return predict_classical(lr_model, data)

        # Prepare scaled data
        close_data = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(close_data)

        # Prepare sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train model if necessary
        if hasattr(model, 'predict') and hasattr(model, 'fit'):
            try:
                model.fit(X_train, y_train, epochs=10,
                          batch_size=32, verbose=0)
            except:
                # If model can't be trained, use prediction only
                pass

        # Make predictions
        y_pred = model.predict(X_test)

        # Transform back to original scale
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Calculate MSE
        mse = mean_squared_error(y_test_orig, y_pred_orig)

        return y_test_orig, y_pred_orig, mse

    except Exception as e:
        st.error(f"Error in LSTM prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        # Fall back to classical model
        lr_model = LinearRegression()
        return predict_classical(lr_model, data)

# ----------------------
# UTILITY FUNCTIONS
# ----------------------


def get_sentiment_score(text):
    """
    Perform sentiment analysis and return polarity score.
    """
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return 0.0


def generate_pdf(stock, summary):
    """
    Generate a PDF report for the stock summary.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Stock Report: {stock}", ln=True)

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            text = f"{key}: {value_str}"
            if len(text) > 80:
                text = text[:77] + "..."
            pdf.cell(200, 10, txt=text, ln=True)

        pdf_path = f"{stock}_report.pdf"
        pdf.output(pdf_path)
        return pdf_path

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# ----------------------
# STREAMLIT UI FUNCTIONS
# ----------------------


def login():
    """
    Display login screen and validate credentials.
    """
    st.title("📊 Stock Analysis - Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username == "admin" and password == "Datta@123":
                st.session_state.logged_in = True
                st.success("Login successful! Redirecting to dashboard...")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")


def display_portfolio_tracker(tickers):
    """
    Display portfolio/watchlist in sidebar.
    """
    st.sidebar.title("Portfolio Tracker")

    default_symbols = ["RELIANCE", "INFY"]
    default_symbols = [s for s in default_symbols if s in tickers.values]
    portfolio = st.sidebar.multiselect(
        "Watchlist", tickers, default=default_symbols)

    if portfolio:
        with st.sidebar.status("Fetching latest prices..."):
            for symbol in portfolio:
                ticker_str = symbol + ".NS"
                try:
                    current_data = safe_yf_download(
                        ticker_str,
                        start=datetime.datetime.today() - datetime.timedelta(days=7),
                        end=datetime.datetime.today()
                    )

                    if not current_data.empty and 'Close' in current_data.columns:
                        if len(current_data) >= 2:
                            price = current_data['Close'].iloc[-1]
                            change = current_data['Close'].iloc[-1] - \
                                current_data['Close'].iloc[-2]
                            pct_change = (
                                change / current_data['Close'].iloc[-2]) * 100
                            st.sidebar.metric(
                                label=symbol,
                                value=f"₹{price:.2f}",
                                delta=f"{pct_change:.2f}%"
                            )
                        else:
                            price = current_data['Close'].iloc[-1]
                            st.sidebar.metric(
                                label=symbol,
                                value=f"₹{price:.2f}"
                            )
                    else:
                        st.sidebar.warning(f"{symbol}: No data available")

                except Exception as e:
                    st.sidebar.warning(f"{symbol}: Error fetching data")


def dashboard():
    """
    Main dashboard UI after successful login.
    """
    st.title("📊 Stock Price Prediction Dashboard")

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Sign out", key="signout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    tickers = load_tickers()
    display_portfolio_tracker(tickers)

    st.subheader("Select Stock and Time Range")
    data_source = st.radio(
        "Data Source", ["Yahoo Finance", "Polygon.io"], index=0, horizontal=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        stock_name = st.selectbox(
            'Select Stock Symbol', tickers, index=0, placeholder="Select stock ticker...")
    with col2:
        start_date = st.date_input(
            "Start Date", value=datetime.date(2020, 1, 1))
    with col3:
        end_date = st.date_input("End Date", value=datetime.date.today())

    polygon_api_key = None
    if data_source == "Polygon.io":
        polygon_api_key = st.text_input("Polygon.io API Key", type="password")

    if stock_name:
        with st.status("Loading stock data...") as status:
            if data_source == "Polygon.io":
                symbol = stock_name if ".NS" not in stock_name else stock_name.replace(
                    ".NS", "")
                data = load_stock_data_polygon(
                    symbol, start_date, end_date, polygon_api_key)
            else:
                stock_ticker = stock_name + ".NS"
                data = load_stock_data(stock_ticker, start_date, end_date)

            if data.empty:
                st.error(
                    "No data found for the selected stock, data source, and date range.")
                status.update(label="Failed to load data", state="error")
                return

            try:
                ticker_info = yf.Ticker(stock_ticker).info
                company_name = ticker_info.get("longName", stock_name)
            except Exception:
                company_name = stock_name

            status.update(label="Data loaded successfully", state="complete")

        st.header(f"📈 {company_name} ({stock_name})")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Price Chart", "Technical Indicators", "Predictions", "Sentiment"])

        with tab1:
            st.subheader("Historical Price Data")

            if 'Close' in data.columns and not data['Close'].empty:
                st.write("Summary Statistics:")
                st.dataframe(
                    data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

                st.subheader("Closing Price with Moving Averages")

                # Ensure moving averages are calculated
                if 'MA50' not in data.columns:
                    data['MA50'] = data['Close'].rolling(50).mean()
                if 'MA200' not in data.columns:
                    data['MA200'] = data['Close'].rolling(200).mean()

                price_chart_data = pd.DataFrame({
                    'Close': data['Close'],
                    '50-Day MA': data['MA50'],
                    '200-Day MA': data['MA200']
                })
                st.line_chart(price_chart_data)

                if 'Volume' in data.columns:
                    st.subheader("Trading Volume")
                    st.bar_chart(data['Volume'])
            else:
                st.error("No Close price data available for charting")

        with tab2:
            st.subheader("Technical Indicators")

            # Calculate indicators if not present
            if 'RSI' not in data.columns or 'MACD' not in data.columns:
                data = add_indicators(data)

            if 'RSI' in data.columns and 'MACD' in data.columns:
                st.write("**Relative Strength Index (RSI)**")
                st.write("RSI measures the speed and change of price movements. Values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.")

                rsi_chart = pd.DataFrame({
                    'RSI': data['RSI'],
                    'Overbought (70)': 70,
                    'Oversold (30)': 30
                })
                st.line_chart(rsi_chart)

                st.write("**MACD (Moving Average Convergence Divergence)**")
                st.write(
                    "MACD is a trend-following momentum indicator that shows the relationship between two moving averages.")
                st.line_chart(data['MACD'])

                if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                    st.write("**Bollinger Bands**")
                    st.write(
                        "Bollinger Bands show price volatility with an upper and lower band that adjust with volatility.")
                    bb_chart = pd.DataFrame({
                        'Price': data['Close'],
                        'Upper Band': data['BB_Upper'],
                        'Lower Band': data['BB_Lower']
                    })
                    st.line_chart(bb_chart)
            else:
                st.warning(
                    "Technical indicators could not be calculated for this stock")

        with tab3:
            st.subheader("Price Predictions")
            model_choice = st.selectbox(
                "Select Prediction Model",
                ["Linear Regression", "Random Forest", "XGBoost", "LSTM"],
                index=0
            )

            # Ensure we have enough data for prediction
            if len(data) < 20:
                st.error("Not enough historical data for meaningful predictions.")
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))

                with st.spinner(f"Training {model_choice} model and generating predictions..."):
                    model = get_model(model_choice)

                    if model_choice == "LSTM":
                        y_actual, y_pred, mse = predict_lstm(
                            model, data, scaler)
                    else:  # Classical models
                        y_actual, y_pred, mse = predict_classical(model, data)

                    if len(y_actual) > 0 and len(y_pred) > 0:
                        # Create dates for the test period
                        test_size = len(y_actual)
                        test_dates = data.index[-test_size:]

                        pred_chart_data = pd.DataFrame({
                            'Actual': y_actual,
                            # Ensure same length
                            'Predicted': y_pred[:len(y_actual)]
                        }, index=test_dates[:len(y_actual)])

                        st.subheader(
                            f'Actual vs Predicted Price ({model_choice})')
                        st.line_chart(pred_chart_data)
                        st.metric("Mean Squared Error", f"{mse:.2f}")

                        # Future prediction
                        st.subheader("Future Price Prediction (7 days)")
                        try:
                            if model_choice != "LSTM":
                                # For classical models, create features for future prediction
                                last_data = data.tail(10).copy()
                                future_dates = pd.date_range(
                                    start=data.index[-1] + pd.Timedelta(days=1), periods=7)

                                # Simple method for future prediction
                                future_prediction = []
                                current_price = data['Close'].iloc[-1]

                                if model_choice == "Linear Regression":
                                    # Calculate average daily change
                                    avg_change = data['Close'].diff().mean()
                                    for i in range(7):
                                        current_price += avg_change
                                        future_prediction.append(current_price)
                                else:
                                    # Use last prediction to estimate trend
                                    if len(y_pred) >= 2:
                                        last_change = (
                                            y_pred[-1] - y_pred[-2]) / y_pred[-2]
                                        for i in range(7):
                                            current_price *= (1 + last_change)
                                            future_prediction.append(
                                                current_price)
                                    else:
                                        # Fallback
                                        future_prediction = [current_price] * 7

                                # Display future prediction
                                future_df = pd.DataFrame({
                                    'Date': future_dates,
                                    'Predicted Price': future_prediction
                                })
                                future_df = future_df.set_index('Date')
                                st.line_chart(future_df)

                                # Show in table format
                                st.table(future_df)

                        except Exception as e:
                            st.error(f"Error predicting future prices: {e}")
                    else:
                        st.error(
                            "Prediction failed. Please try another model or stock.")

        with tab4:
            st.subheader("Sentiment Analysis")
            st.write("**Note:** This is a simplified sentiment analysis demo.")
            headlines = [
                f"{company_name} announces quarterly results",
                f"Analysts provide positive outlook for {company_name}",
                f"{company_name} expands operations in new markets",
                f"Economic factors affecting {stock_name} performance"
            ]
            for headline in headlines:
                sentiment_score = get_sentiment_score(headline)
                sentiment_category = "Neutral"
                if sentiment_score > 0.2:
                    sentiment_category = "Positive"
                elif sentiment_score < -0.2:
                    sentiment_category = "Negative"
                st.write(f"**{headline}**")
                st.metric(
                    "Sentiment Score",
                    f"{sentiment_score:.2f}",
                    delta=sentiment_category,
                    delta_color="normal"
                )
                st.write("---")
            avg_sentiment = sum(get_sentiment_score(h)
                                for h in headlines) / len(headlines)
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")

        st.subheader("Download Analysis Report")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                summary = {
                    "Company": company_name,
                    "Symbol": stock_name,
                    "Date Range": f"{start_date} to {end_date}"
                }

                if 'Close' in data.columns:
                    summary["Current Price"] = data['Close'].iloc[-1]
                if 'High' in data.columns:
                    summary["52-Week High"] = data['High'].max()
                if 'Low' in data.columns:
                    summary["52-Week Low"] = data['Low'].min()
                if 'Volume' in data.columns:
                    summary["Average Volume"] = data['Volume'].mean()
                if 'RSI' in data.columns:
                    summary["RSI (Last)"] = data['RSI'].iloc[-1]
                if 'MACD' in data.columns:
                    summary["MACD (Last)"] = data['MACD'].iloc[-1]

                summary["Prediction Model"] = model_choice
                if 'mse' in locals():
                    summary["Prediction Error (MSE)"] = mse

                pdf_path = generate_pdf(stock_name, summary)

                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Download PDF Report",
                            f,
                            file_name=pdf_path,
                            mime="application/pdf"
                        )
                else:
                    st.error("Failed to generate PDF report.")

    else:
        st.info("Please select a stock ticker to view details.")

# ----------------------
# MAIN APPLICATION ENTRY
# ----------------------


st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    login()
