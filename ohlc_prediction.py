import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import streamlit as st

st.title("Stock Price Prediction using LSTM")
st.write("This app predicts stock OHLC (Open, High, Low, Close) values using an LSTM model.")
st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):")
start_date = st.sidebar.text_input("Start Date (provide Start Date in the following format (YYYY-MM-DD)):")
end_date = st.sidebar.text_input("End Date (provide End Date in the following format (YYYY-MM-DD)):")

@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.sidebar.error(f"Please make sure you provide a valid Stock Ticker, Start Date and End Date.")
        return None
    return data

if not ticker or not start_date or not end_date:
    st.write("Please enter a valid Stock Ticker, Start Date and End Date")

else:
    try:
        data = load_data(ticker, start_date, end_date)
        if data is not None:
            st.write(f"### {ticker} Stock Data")
            st.write(data.tail())
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']].values)
            @st.cache_data
            def create_dataset(data, time_step=60):
                X, y = [], []
                for i in range(len(data) - time_step - 1):
                    X.append(data[i:(i + time_step)])
                    y.append(data[i + time_step])
                return np.array(X), np.array(y)
            time_step = 60
            X, y = create_dataset(scaled_data, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 4)
            @st.cache_resource
            def build_and_train_model(X, y):
                model = Sequential()
                model.add(LSTM(200, return_sequences=True, input_shape=(time_step, 4)))
                model.add(LSTM(200, return_sequences=False))
                model.add(Dense(50))
                model.add(Dense(4))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, batch_size=1, epochs=1, verbose=0)
                return model

            model = build_and_train_model(X, y)
            train_predict = model.predict(X)
            train_predict = scaler.inverse_transform(train_predict)
            st.write("### Stock Price Predictions (Open, High, Low, Close)")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Open'].index[time_step + 1:], data['Open'].values[time_step + 1:], label='Actual Open Price')
            plt.plot(data['Open'].index[time_step + 1:], train_predict[:, 0], label='Predicted Open Price')
            plt.title(f"{ticker} Open Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            plt.figure(figsize=(10, 6))
            plt.plot(data['High'].index[time_step + 1:], data['High'].values[time_step + 1:], label='Actual High Price')
            plt.plot(data['High'].index[time_step + 1:], train_predict[:, 1], label='Predicted High Price')
            plt.title(f"{ticker} High Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            plt.figure(figsize=(10, 6))
            plt.plot(data['Low'].index[time_step + 1:], data['Low'].values[time_step + 1:], label='Actual Low Price')
            plt.plot(data['Low'].index[time_step + 1:], train_predict[:, 2], label='Predicted Low Price')
            plt.title(f"{ticker} Low Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'].index[time_step + 1:], data['Close'].values[time_step + 1:], label='Actual Close Price')
            plt.plot(data['Close'].index[time_step + 1:], train_predict[:, 3], label='Predicted Close Price')
            plt.title(f"{ticker} Close Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            future_days = 30
            last_60_days = scaled_data[-time_step:]
            future_predictions = []
            for _ in range(future_days):
                pred = model.predict(last_60_days.reshape(1, time_step, 4))
                future_predictions.append(pred[0])
                last_60_days = np.append(last_60_days[1:], pred, axis=0)
            future_predictions = scaler.inverse_transform(np.array(future_predictions))
            st.write("### Future Stock Price Predictions (Open, High, Low, Close)")
            future_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='B')[1:]

            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_predictions[:, 0], label='Future Predicted Open Price', color='green')
            plt.title(f"{ticker} Future Open Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_predictions[:, 1], label='Future Predicted High Price', color='blue')
            plt.title(f"{ticker} Future High Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_predictions[:, 2], label='Future Predicted Low Price', color='orange')
            plt.title(f"{ticker} Future Low Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)
            
            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_predictions[:, 3], label='Future Predicted Close Price', color='red')
            plt.title(f"{ticker} Future Close Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)
    except ValueError:
        st.sidebar.error("Please provide valid date format (YYYY-MM-DD) for Start and End Dates.")