import time
from fpdf import FPDF
from textblob import TextBlob  # For sentiment analysis
import ta  # Technical analysis library
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
import tensorflow as tf


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")


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
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=start, end=end, auto_adjust=True)

            if not data.empty:
                # IMPORTANT FIX: Handle the column naming issue properly
                # yfinance can return columns as tuples like (Open, TICKER) or just 'Open'
                # This fixes the issue by taking only the first part of any tuple columns
                data.columns = [col[0] if isinstance(
                    col, tuple) else col for col in data.columns]

                # Standardize any column names with spaces or special format
                standard_columns = {'Adj Close': 'Adj_Close'}
                data = data.rename(columns=standard_columns)

                return data
        except Exception:
            time.sleep(2)  # Wait before retrying

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_tickers(filepath='stock_tickers.xlsx'):
    """
    Load stock tickers from Excel file.
    """
    try:
        # Try to load tickers from Excel file
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
    # Get the raw stock data
    data = safe_yf_download(stock, start, end)

    if not data.empty:
        # Debug information to show what columns we have in the raw data
        st.write(f"Columns in raw data: {data.columns.tolist()}")

        # Add technical indicators (RSI, MACD, Bollinger Bands)
        data = add_indicators(data)

        # Show what columns we have after adding indicators
        st.write(f"Columns after adding indicators: {data.columns.tolist()}")

    return data

# ----------------------
# DATA PROCESSING FUNCTIONS
# ----------------------


def add_indicators(df):
    """
    Add technical indicators to the dataframe.
    """
    # Check if dataframe is empty first
    if df.empty:
        return df

    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    try:
        # Verify required column exists before calculating indicators
        if 'Close' not in df.columns:
            st.error("No Close price column found. Cannot add indicators.")
            return df

        # Add Relative Strength Index (RSI)
        # RSI measures momentum - above 70 is overbought, below 30 is oversold
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Add Moving Average Convergence Divergence (MACD)
        # MACD shows relationship between two moving averages
        macd_indicator = ta.trend.MACD(df['Close'])
        df['MACD'] = macd_indicator.macd_diff()

        # Add Bollinger Bands - shows volatility with upper and lower bands
        bb = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()

        # Add moving averages for trend analysis
        df['MA50'] = df['Close'].rolling(50).mean()  # 50-day moving average
        df['MA200'] = df['Close'].rolling(200).mean()  # 200-day moving average

        # Fill NaN values that might be created by indicators that need historical data
        # bfill = backward fill, ffill = forward fill
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
                # Try to load a pre-trained LSTM model if it exists
                import os
                if os.path.exists('stock_prediction_model.h5'):
                    from tensorflow.keras.models import load_model
                    return load_model('stock_prediction_model.h5')
            except Exception:
                pass

            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, LSTM, Input
                model = Sequential()
                # Use an explicit Input layer to avoid UserWarning about passing
                # `input_shape` directly to recurrent layers when using Sequential.
                model.add(Input(shape=(None, 1)))
                model.add(LSTM(units=50, return_sequences=True))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                return model
            except Exception as e:
                # Fallback to Linear Regression if LSTM model can't be loaded
                st.error(
                    f"Error loading LSTM model: {e}. Using Linear Regression instead.")
                return LinearRegression()
        elif name == "Linear Regression":
            # Simple linear regression model
            return LinearRegression()
        elif name == "Random Forest":
            # Random Forest with 100 trees for regression
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif name == "XGBoost":
            # XGBoost regressor, good for time series
            return xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:
            st.error("Model not implemented")
            return LinearRegression()  # Default fallback
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return LinearRegression()  # Default fallback


def create_features(data, lookback=20):
    """
    Create features for prediction models using lookback periods.
    This improves prediction accuracy by using past values as features.
    """
    # Check if we have the required data
    if 'Close' not in data.columns or data.empty:
        return None, None

    # Create an empty dataframe with the same index as our data
    features = pd.DataFrame(index=data.index)

    # Target variable is the closing price
    features['Close'] = data['Close']

    # Add engineered features
    features['Return'] = data['Close'].pct_change()
    features['MA10'] = data['Close'].rolling(10).mean()
    features['MA20'] = data['Close'].rolling(20).mean()
    features['Volatility'] = features['Return'].rolling(10).std()

    # Add previous days' closing prices as features
    for i in range(1, lookback+1):
        features[f'Close_lag_{i}'] = data['Close'].shift(i)

    # Add technical indicators if available
    for col in ['RSI', 'MACD']:
        if col in data.columns:
            features[col] = data[col]

    # Drop rows with NaN values (created by shift operation)
    features = features.dropna()

    # Prepare X (features) and y (target)
    # Features are everything except current Close
    X = features.drop('Close', axis=1)
    y = features['Close']  # Target is current Close

    return X, y


def predict_classical(model, data, test_size=0.2, forecast_days=0):
    """
    Train and predict using a classical ML model with improved features.
    """
    try:
        # Check if we have the required data
        if 'Close' not in data.columns or data.empty:
            st.error("No Close price data available for prediction")
            return np.array([]), np.array([]), float('inf'), []

        # Create features with a lookback period
        X, y = create_features(data)
        if X is None or y is None:
            st.error("Failed to create features for prediction")
            return np.array([]), np.array([]), float('inf'), []

        # Split data into training and testing sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Check if we have enough data to train and test
        if len(X_train) < 10 or len(X_test) < 1:
            st.error(
                f"Insufficient data for prediction: train={len(X_train)}, test={len(X_test)}")
            return np.array([]), np.array([]), float('inf'), []

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        eval_metrics = {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE (%)': mape
        }

        future_prediction = []
        if forecast_days > 0:
            current_price = data['Close'].iloc[-1]
            if isinstance(model, LinearRegression):
                future_prediction = generate_historical_return_forecast(
                    data, forecast_days, start_price=current_price)
            else:
                if len(y_pred) >= 2:
                    last_change = (y_pred[-1] - y_pred[-2]) / y_pred[-2]
                    last_change = max(min(last_change, 0.05), -0.05)
                    for i in range(forecast_days):
                        current_price *= (1 + last_change)
                        future_prediction.append(current_price)
                else:
                    future_prediction = generate_historical_return_forecast(
                        data, forecast_days, start_price=current_price)

            if len(future_prediction) == forecast_days and np.ptp(future_prediction) < 1e-6:
                future_prediction = generate_historical_return_forecast(
                    data, forecast_days, start_price=current_price)

        return y_test.values, y_pred, eval_metrics, future_prediction

    except Exception as e:
        st.error(f"Error in classical model prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return np.array([]), np.array([]), {}, []


def predict_lstm(model, data, scaler, sequence_length=60, forecast_days=0):
    """
    Prepare sequences and predict using an LSTM model.
    LSTM is good for time series data as it can capture temporal patterns.
    """
    try:
        # If we don't have enough data for LSTM, fall back to linear regression
        if len(data) < 30 or 'Close' not in data.columns:
            st.warning(
                "Insufficient data (less than 30 days). Using linear regression instead.")
            lr_model = LinearRegression()
            return predict_classical(lr_model, data, forecast_days=forecast_days)

        if len(data) < sequence_length * 2:
            sequence_length = max(10, len(data) // 3)

        # Prepare scaled data - LSTM needs normalized inputs
        close_data = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(close_data)

        # Prepare sequences for LSTM
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train model if necessary
        if hasattr(model, 'predict') and hasattr(model, 'fit'):
            try:
                from tensorflow.keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(
                    monitor='loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=50, batch_size=16,
                          verbose=0, callbacks=[early_stop])
            except Exception as e:
                st.warning(f"Warning during model training: {e}")

        # Make predictions on test data
        y_pred = model.predict(X_test, verbose=0)

        # Transform predictions back to original scale
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100

        eval_metrics = {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE (%)': mape
        }

        future_prediction_orig = []
        if forecast_days > 0 and len(scaled_data) >= sequence_length:
            current_seq = scaled_data[-sequence_length:].copy()
            future_scaled = []

            for _ in range(forecast_days):
                pred_input = np.reshape(current_seq, (1, sequence_length, 1))
                next_pred = model.predict(pred_input, verbose=0)[0][0]
                future_scaled.append(next_pred)

                # Shift sequence forward and append prediction
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1, 0] = next_pred

            future_prediction_orig = scaler.inverse_transform(
                np.array(future_scaled).reshape(-1, 1)).flatten().tolist()

        return y_test_orig, y_pred_orig, eval_metrics, future_prediction_orig

    except Exception as e:
        st.error(f"Error in LSTM prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        lr_model = LinearRegression()
        return predict_classical(lr_model, data, forecast_days=forecast_days)

# ----------------------
# UTILITY FUNCTIONS
# ----------------------


def get_sentiment_score(text):
    """
    Perform sentiment analysis and return polarity score.
    Positive score = positive sentiment, negative score = negative sentiment.
    """
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return 0.0  # Neutral sentiment as fallback


def generate_pdf(stock, summary):
    """
    Generate a PDF report for the stock summary.
    """
    try:
        # Create a new PDF document
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Stock Report: {stock}", ln=True)

        # Add each summary item to the PDF
        for key, value in summary.items():
            # Format numbers to 2 decimal places
            if isinstance(value, (int, float)):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            # Create the text and truncate if too long
            text = f"{key}: {value_str}"
            if len(text) > 80:
                text = text[:77] + "..."

            # Add the text as a cell in the PDF
            pdf.cell(200, 10, txt=text, ln=True)

        # Save the PDF to a file
        pdf_path = f"{stock}_report.pdf"
        pdf.output(pdf_path)
        return pdf_path

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None


def analyze_market_data(data):
    """
    Analyze historical market behavior and recommend a model style.
    """
    if 'Close' not in data.columns or data['Close'].dropna().empty:
        return {
            'market_type': 'Unknown',
            'market_direction': 'Stable',
            'recommended_model': 'Linear Regression',
            'volatility': 0.0,
            'trend_strength': 0.0
        }

    close = data['Close'].dropna()
    returns = close.pct_change().dropna()
    volatility = returns.std()
    slope = np.polyfit(np.arange(len(close)), close.values, 1)[0]
    mean_close = close.mean()
    trend_strength = slope / mean_close if mean_close != 0 else 0.0
    market_direction = 'Upward' if slope > 0 else 'Downward' if slope < 0 else 'Stable'

    if len(close) < 60:
        market_type = 'Limited Data'
        recommended_model = 'Linear Regression'
    elif volatility > 0.03:
        market_type = f'Volatile {market_direction}'
        recommended_model = 'XGBoost' if len(close) > 120 else 'Random Forest'
    elif abs(trend_strength) > 0.002 and len(close) > 150:
        market_type = f'Trending {market_direction}'
        recommended_model = 'LSTM'
    else:
        market_type = 'Stable'
        recommended_model = 'Random Forest'

    return {
        'market_type': market_type,
        'market_direction': market_direction,
        'recommended_model': recommended_model,
        'volatility': volatility,
        'trend_strength': abs(trend_strength)
    }


def choose_best_model(results, market_profile=None):
    """
    Choose the best model using metrics and market profile.
    """
    if not results:
        return None

    best = None
    best_score = -np.inf

    for result in results:
        metrics = result['eval_metrics']
        r2 = metrics.get('R2', -np.inf)
        rmse = metrics.get('RMSE', np.inf)
        y_actual = result.get('y_actual', [])
        mean_actual = np.mean(np.abs(y_actual)) if len(y_actual) else 0.0
        rmse_norm = rmse / mean_actual if mean_actual else rmse

        bonus = 0.0
        if market_profile is not None:
            if result['model_name'] == market_profile.get('recommended_model'):
                bonus += 0.2
            if market_profile.get('market_type') != 'Stable' and result['model_name'] == 'Linear Regression':
                bonus -= 0.1

        score = r2 - rmse_norm + bonus

        if score > best_score:
            best_score = score
            best = result

    return best


def get_trading_signal(future_prediction, current_price):
    """
    Create a buy/hold/sell recommendation based on future prediction direction.
    """
    if not future_prediction or current_price is None:
        return "→ Hold", "→ Stable Market", 0.0

    forecast_start = future_prediction[0]
    forecast_end = future_prediction[-1]
    forecast_return = ((forecast_end - forecast_start) /
                       forecast_start) * 100 if forecast_start != 0 else 0.0
    pct_change = ((forecast_end - current_price) / current_price) * \
        100 if current_price != 0 else 0.0

    if pct_change > 0.5:
        action = "↑ Buy"
    elif pct_change < -0.5:
        action = "↓ Sell"
    else:
        action = "→ Hold"

    if abs(forecast_return) < 0.2:
        trend_label = "→ Stable Market"
    elif forecast_return > 0:
        trend_label = "↑ Upward Trend"
    else:
        trend_label = "↓ Downward Trend"

    return action, trend_label, pct_change


def generate_historical_return_forecast(data, forecast_days, start_price=None):
    """
    Generate a future price sequence using recent historical percent changes.
    This helps the future chart show actual up/down movement.
    """
    if forecast_days <= 0 or 'Close' not in data.columns or data['Close'].dropna().empty:
        return []

    price = float(
        start_price if start_price is not None else data['Close'].iloc[-1])
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return [price] * forecast_days

    recent_returns = returns.tail(
        max(forecast_days, min(len(returns), 20))).tolist()
    forecast = []
    for i in range(forecast_days):
        daily_ret = recent_returns[i] if i < len(
            recent_returns) else recent_returns[i % len(recent_returns)]
        price *= (1 + daily_ret)
        forecast.append(price)
    return forecast


def build_market_style_forecast_figure(dates, prices, trend_label, current_price=None):
    """
    Build a polished bar chart for future price predictions with a dark market theme.
    The bar colors reflect day-to-day direction, making up/down moves easier to read.
    """
    steps = []
    if current_price is not None and prices:
        steps.append(prices[0] - current_price)
    steps += [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    bar_colors = []
    for change in steps:
        if change > 0:
            bar_colors.append("rgba(0, 255, 127, 0.85)")
        elif change < 0:
            bar_colors.append("rgba(255, 77, 77, 0.85)")
        else:
            bar_colors.append("rgba(204, 204, 204, 0.85)")

    trend_arrows = []
    for i, change in enumerate(steps):
        if change > 0:
            trend_arrows.append('↑')
        elif change < 0:
            trend_arrows.append('↓')
        else:
            trend_arrows.append('→')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=prices,
        marker=dict(color=bar_colors, line=dict(color="#111111", width=1)),
        text=trend_arrows,
        textposition='outside',
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f} %{text}<extra></extra>",
        name="Predicted Price"
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines+markers",
        line=dict(color="#ffffff", width=2),
        marker=dict(color="#ffffff", size=6),
        hoverinfo='skip',
        name='Trend'
    ))

    y_min = min(prices) if prices else 0
    y_max = max(prices) if prices else 0
    y_pad = max((y_max - y_min) * 0.08, 1)

    fig.update_layout(
        title={
            'text': f"Future Price Forecast ({trend_label})",
            'x': 0.02,
            'xanchor': 'left'
        },
        xaxis_title="Date",
        yaxis_title="Predicted Price",
        template="plotly_dark",
        plot_bgcolor="#0b1320",
        paper_bgcolor="#0b1320",
        font=dict(color="#e5e5e5"),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#334155",
            tickangle=-45,
            tickfont=dict(color="#e5e5e5")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#192138",
            zeroline=False,
            showline=True,
            linecolor="#334155",
            range=[y_min - y_pad, y_max + y_pad]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=70, b=100)
    )
    return fig


def evaluate_models(data, scaler, forecast_days):
    """
    Run all prediction algorithms and collect their metrics.
    """
    model_names = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
    results = []

    for model_name in model_names:
        model = get_model(model_name)
        if model_name == "LSTM":
            y_actual, y_pred, eval_metrics, future_prediction = predict_lstm(
                model, data, scaler, forecast_days=forecast_days
            )
        else:
            y_actual, y_pred, eval_metrics, future_prediction = predict_classical(
                model, data, forecast_days=forecast_days
            )

        if len(y_actual) == 0 or len(y_pred) == 0:
            continue

        results.append({
            'model_name': model_name,
            'y_actual': y_actual,
            'y_pred': y_pred,
            'future_prediction': future_prediction,
            'eval_metrics': eval_metrics
        })

    return results

# ----------------------
# STREAMLIT UI FUNCTIONS
# ----------------------


def sync_selected_stock_symbol(selected_stock_symbol, watchlist_symbols, available_symbols):
    """
    Keep the main stock selector aligned with the watchlist selection.
    If a watchlist symbol is chosen, prefer it; otherwise fall back to the
    current selection or the first available stock.
    """
    available_symbols = [str(symbol) for symbol in available_symbols]
    watchlist_symbols = [str(symbol) for symbol in watchlist_symbols or []]

    if not available_symbols:
        return selected_stock_symbol

    for symbol in watchlist_symbols:
        if symbol in available_symbols:
            return symbol

    if selected_stock_symbol in available_symbols:
        return selected_stock_symbol

    return available_symbols[0]


def login():
    """
    Display login screen and validate credentials.
    """
    st.title("📊 Stock Analysis - Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        # Simple credential validation
        if submit:
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.success("Login successful! Redirecting to dashboard...")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")


def display_portfolio_tracker(tickers):
    """
    Show the sidebar watchlist selector and the latest price for the selected symbol.
    """
    st.sidebar.title("Portfolio Tracker")

    available_symbols = [str(symbol) for symbol in tickers.tolist()]
    if not available_symbols:
        return None

    default_symbol = "RELIANCE" if "RELIANCE" in available_symbols else available_symbols[0]
    if "watchlist_symbol_value" not in st.session_state:
        st.session_state.watchlist_symbol_value = default_symbol

    selected_watchlist = st.sidebar.selectbox(
        "Watchlist",
        available_symbols,
        index=available_symbols.index(st.session_state.watchlist_symbol_value)
        if st.session_state.watchlist_symbol_value in available_symbols
        else 0,
        key="watchlist_symbol_widget",
    )

    if selected_watchlist in available_symbols:
        st.session_state.watchlist_symbol_value = selected_watchlist
        st.session_state.stock_symbol = selected_watchlist

    with st.sidebar.status(f"Fetching latest prices for {selected_watchlist}..."):
        ticker_str = selected_watchlist + ".NS"
        try:
            current_data = safe_yf_download(
                ticker_str,
                start=datetime.datetime.today() - datetime.timedelta(days=7),
                end=datetime.datetime.today(),
            )

            if not current_data.empty and "Close" in current_data.columns:
                if len(current_data) >= 2:
                    price = current_data["Close"].iloc[-1]
                    change = current_data["Close"].iloc[-1] - \
                        current_data["Close"].iloc[-2]
                    pct_change = (
                        change / current_data["Close"].iloc[-2]) * 100
                    st.sidebar.metric(
                        label=selected_watchlist,
                        value=f"₹{price:.2f}",
                        delta=f"{pct_change:.2f}%",
                    )
                else:
                    price = current_data["Close"].iloc[-1]
                    st.sidebar.metric(label=selected_watchlist,
                                      value=f"₹{price:.2f}")
            else:
                pass

        except Exception:
            pass

    return selected_watchlist


def dashboard():
    """
    Main dashboard UI after successful login.
    """
    st.title("📊 Stock Price Prediction Dashboard")

    # Add a sign-out button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Sign out", key="signout"):
            st.session_state.logged_in = False
            st.rerun()

    # Load stock tickers and display portfolio tracker
    tickers = load_tickers()
    available_symbols = [str(symbol) for symbol in tickers.tolist()]

    if not available_symbols:
        st.error("No stock symbols available.")
        return

    if "stock_symbol" not in st.session_state:
        st.session_state.stock_symbol = available_symbols[0]
    if "watchlist_symbol_value" not in st.session_state:
        st.session_state.watchlist_symbol_value = available_symbols[0]

    selected_watchlist = display_portfolio_tracker(tickers)
    selected_stock_symbol = sync_selected_stock_symbol(
        st.session_state.get("stock_symbol"),
        [selected_watchlist] if selected_watchlist else [
            st.session_state.get("watchlist_symbol_value")],
        available_symbols,
    )
    st.session_state.stock_symbol = selected_stock_symbol

    # Stock and date selection UI
    st.subheader("Select Stock and Time Range")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        current_stock_symbol = st.session_state.get(
            "stock_symbol") or selected_stock_symbol
        stock_name = st.selectbox(
            "Select Stock Symbol",
            available_symbols,
            index=available_symbols.index(current_stock_symbol)
            if current_stock_symbol in available_symbols
            else 0,
            key="stock_symbol",
        )
    with col2:
        start_date = st.date_input(
            "Start Date", value=datetime.date(2024, 1, 1))
    with col3:
        end_date = st.date_input("End Date", value=datetime.date.today())

    # If a stock is selected, load and display data
    if stock_name:
        with st.status("Loading stock data...") as status:
            # Add .NS suffix for NSE (Indian stocks)
            stock_ticker = stock_name + ".NS"

            # Load stock data
            data = load_stock_data(stock_ticker, start_date, end_date)
            if data.empty:
                st.error("No data found for the selected stock and date range.")
                status.update(label="Failed to load data", state="error")
                return

            # Get company name if available
            try:
                ticker_info = yf.Ticker(stock_ticker).info
                company_name = ticker_info.get("longName", stock_name)
            except Exception:
                company_name = stock_name

            status.update(label="Data loaded successfully", state="complete")

        # Display stock header with company name
        st.header(f"📈 {company_name} ({stock_name})")

        # Create tabs for different types of analysis
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Price Chart", "Technical Indicators", "Predictions", "Sentiment"])

        # Tab 1: Price Chart
        with tab1:
            st.subheader("Historical Price Data")

            # Check if we have closing price data available
            if 'Close' in data.columns and not data['Close'].empty:
                # Show summary statistics
                st.write("Summary Statistics:")
                st.dataframe(
                    data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

                # Display closing price chart with moving averages
                st.subheader("Closing Price with Moving Averages")

                # Ensure moving averages are calculated
                if 'MA50' not in data.columns:
                    data['MA50'] = data['Close'].rolling(50).mean()
                if 'MA200' not in data.columns:
                    data['MA200'] = data['Close'].rolling(200).mean()

                # Create dataframe for chart
                price_chart_data = pd.DataFrame({
                    'Close': data['Close'],
                    '50-Day MA': data['MA50'],
                    '200-Day MA': data['MA200']
                })
                st.line_chart(price_chart_data)

                # Display volume chart if available
                if 'Volume' in data.columns:
                    st.subheader("Trading Volume")
                    st.bar_chart(data['Volume'])
            else:
                st.error("No Close price data available for charting")

        # Tab 2: Technical Indicators
        with tab2:
            st.subheader("Technical Indicators")

            # Calculate indicators if not already present
            if 'RSI' not in data.columns or 'MACD' not in data.columns:
                data = add_indicators(data)

            # Display technical indicators if available
            if 'RSI' in data.columns and 'MACD' in data.columns:
                # RSI Chart
                st.write("**Relative Strength Index (RSI)**")
                st.write("RSI measures the speed and change of price movements. Values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.")

                rsi_chart = pd.DataFrame({
                    'RSI': data['RSI'],
                    'Overbought (70)': 70,  # Horizontal line at 70
                    'Oversold (30)': 30     # Horizontal line at 30
                })
                st.line_chart(rsi_chart)

                # MACD Chart
                st.write("**MACD (Moving Average Convergence Divergence)**")
                st.write(
                    "MACD is a trend-following momentum indicator that shows the relationship between two moving averages.")
                st.line_chart(data['MACD'])

                # Bollinger Bands Chart
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

        # Tab 3: Predictions
        with tab3:
            st.subheader("Price Predictions")
            st.write(
                "All models are run automatically and the best-performing model is selected for you.")

            # Allow user to pick forecast horizon independently
            forecast_days = st.slider(
                "Select days to predict", min_value=1, max_value=30, value=30)

            # Check if we have enough data for prediction
            if len(data) < 20:
                st.error("Not enough historical data for meaningful predictions.")
            else:
                # Initialize scaler for normalizing data (especially for LSTM)
                scaler = MinMaxScaler(feature_range=(0, 1))

                # Run all models and choose the best one automatically
                with st.spinner("Running all prediction models and selecting the best one..."):
                    market_profile = analyze_market_data(data)
                    model_results = evaluate_models(
                        data, scaler, forecast_days)

                    st.markdown(
                        f"**Market type:** {market_profile['market_type']}  ")
                    st.markdown(
                        f"**Recommended model from market analysis:** {market_profile['recommended_model']}  ")
                    st.markdown(
                        f"**Volatility:** {market_profile['volatility']:.4f}  ")
                    st.markdown(
                        f"**Trend strength:** {market_profile['trend_strength']:.4f}  ")

                    if not model_results:
                        st.error(
                            "Unable to generate predictions with the available data.")
                    else:
                        comparison_rows = []
                        for result in model_results:
                            metrics = result['eval_metrics']
                            comparison_rows.append({
                                'Model': result['model_name'],
                                'R2': metrics.get('R2', np.nan),
                                'RMSE': metrics.get('RMSE', np.nan),
                                'MAE': metrics.get('MAE', np.nan),
                                'MAPE (%)': metrics.get('MAPE (%)', np.nan)
                            })

                        comparison_df = pd.DataFrame(
                            comparison_rows).set_index('Model')
                        st.subheader("Model Comparison")
                        st.dataframe(comparison_df.style.format(
                            "{:.4f}"), hide_index=False)

                        best_result = choose_best_model(
                            model_results, market_profile)
                        if best_result is None:
                            st.error("Could not determine the best model.")
                        else:
                            best_model_name = best_result['model_name']
                            y_actual = best_result['y_actual']
                            y_pred = best_result['y_pred']
                            future_prediction = best_result['future_prediction']
                            eval_metrics = best_result['eval_metrics']

                            st.success(
                                f"Best model selected: {best_model_name}")

                            # Create dates for the test period
                            test_size = len(y_actual)
                            test_dates = data.index[-test_size:]

                            pred_chart_data = pd.DataFrame({
                                'Actual': y_actual,
                                'Predicted': y_pred[:len(y_actual)]
                            }, index=test_dates[:len(y_actual)])

                            st.subheader(
                                f'Actual vs Predicted Price ({best_model_name})')
                            st.line_chart(pred_chart_data)

                            st.subheader("Evaluation Metrics")
                            st.dataframe(pd.DataFrame([eval_metrics]).style.format(
                                "{:.4f}"), hide_index=True)

                            st.subheader("Test Data Predictions List")
                            st.dataframe(pred_chart_data)

                            # Future prediction and recommendation for best model
                            st.subheader(
                                f"Future Price Prediction ({forecast_days} days)")
                            try:
                                future_dates = pd.date_range(
                                    start=data.index[-1] +
                                    pd.Timedelta(days=1),
                                    periods=forecast_days
                                )

                                if len(future_prediction) == len(future_dates):
                                    current_price = data['Close'].iloc[-1] if 'Close' in data.columns else None
                                    action, trend_label, pct_change = get_trading_signal(
                                        future_prediction, current_price)

                                    future_df = pd.DataFrame({
                                        "Date": future_dates,
                                        "Predicted Price": future_prediction
                                    })
                                    future_df = future_df.set_index('Date')

                                    fig = build_market_style_forecast_figure(
                                        future_df.index,
                                        future_df["Predicted Price"].tolist(),
                                        trend_label
                                    )

                                    st.markdown(
                                        f"**Recommendation:** {action}")
                                    st.markdown(
                                        f"**Market direction:** {trend_label} ({pct_change:.2f}% change from current price)")
                                    st.plotly_chart(
                                        fig, use_container_width=True)
                                    st.table(future_df)

                                else:
                                    st.error(
                                        "Future prediction length does not match requested days.")
                            except Exception as e:
                                st.error(
                                    f"Error displaying future prices: {e}")
        # Tab 4: Sentiment Analysis
        with tab4:
            st.subheader("Sentiment Analysis")
            st.write("**Note:** This is a simplified sentiment analysis demo.")

            # Sample headlines for sentiment analysis
            headlines = [
                f"{company_name} announces quarterly results",
                f"Analysts provide positive outlook for {company_name}",
                f"{company_name} expands operations in new markets",
                f"Economic factors affecting {stock_name} performance"
            ]

            # Analyze sentiment for each headline
            for headline in headlines:
                sentiment_score = get_sentiment_score(headline)

                # Categorize sentiment based on score
                sentiment_category = "Neutral"
                if sentiment_score > 0.2:
                    sentiment_category = "Positive"
                elif sentiment_score < -0.2:
                    sentiment_category = "Negative"

                # Display headline and sentiment
                st.write(f"**{headline}**")
                st.metric(
                    "Sentiment Score",
                    f"{sentiment_score:.2f}",
                    delta=sentiment_category,
                    delta_color="normal"
                )
                st.write("---")

            # Display average sentiment
            avg_sentiment = sum(get_sentiment_score(h)
                                for h in headlines) / len(headlines)
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")

        # PDF Report Generation
        st.subheader("Download Analysis Report")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                # Create summary dictionary for report
                summary = {
                    "Company": company_name,
                    "Symbol": stock_name,
                    "Date Range": f"{start_date} to {end_date}"
                }

                # Add available metrics to summary
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

                # Add prediction info
                # Safely determine which model was used (if any)
                model_choice = best_model_name if 'best_model_name' in locals() else 'N/A'
                summary["Prediction Model"] = model_choice
                if 'eval_metrics' in locals() and eval_metrics:
                    summary["Prediction R2 Score"] = eval_metrics.get("R2", 0)
                    summary["Prediction RMSE"] = eval_metrics.get("RMSE", 0)

                # Generate PDF and provide download button
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


# Configure the Streamlit page
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for login status if not present
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show dashboard or login screen based on login status
if __name__ == "__main__":
    if st.session_state.logged_in:
        dashboard()
    else:

        login()
