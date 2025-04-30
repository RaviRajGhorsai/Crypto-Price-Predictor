import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,Input
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.sidebar.image('IMG-20250117-WA0006.jpg', use_container_width=True)
st.sidebar.header("Model Parameters")

crypto_symbol = st.sidebar.text_input("Crypto Symbol", "BTC-USD")
prediction_ahead = st.sidebar.number_input("Prediction Days Ahead", min_value=1)

if st.sidebar.button("Predict"):
    # download btc data
    btc_data = yf.download(crypto_symbol, period='1y', interval='1d')
    ser = btc_data
    btc_data = btc_data['Close']
    
    # scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(btc_data)
    
    #split data into train and test data
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    def create_dataset(data, time_step = 1):
        X, y = [], []

        for i in range(len(data) - time_step):
            X.append(data[i: (i+time_step) ,0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    
    # further split data into X_train, X_test, y_train and y_test
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step=time_step)
    X_test, y_test = create_dataset(test_data, time_step=time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # create an model and train it
    
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1)  # training data

    train_preds = model.predict(X_train)  # Predictions
    test_preds = model.predict(X_test)

    train_preds = scaler.inverse_transform(train_preds)
    y_train = scaler.inverse_transform(y_train.reshape(-1,1))

    test_preds = scaler.inverse_transform(test_preds)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    
    # forecast future

    last_60_days = scaled_data[-time_step:]
    future_input = last_60_days.reshape(1, time_step, 1)
    future_forecast = []

    for _ in range(prediction_ahead):
        next_pred = model.predict(future_input)[0,0]
        future_forecast.append(next_pred)
        next_input = np.append(future_input[0,1:], [[next_pred]], axis=0)
        future_input = next_input.reshape(1, time_step, 1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1))

    latest_closed_price = float(ser['Close'].iloc[-1])
    latest_predicted_price = float(future_forecast[-1])
    
    
    # centered layout for metrics
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"""
                    <div style="display: flex; justify-content: space-around;">
    <div style="background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center">
        <h3>Latest Close Price</h3>
        <p style="font-size: 20px;">{latest_closed_price:.3f}</p>
    </div>
    <div style="background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center">
        <h3>Price after {prediction_ahead} days</h3>
        <p style="font-size: 20px;">{latest_predicted_price:.3f}</p>
    </div>
    </div> """, unsafe_allow_html=True)
    
    # plot the trained, test, predicted data
    
    plt.figure(figsize=(14,5))
    plt.plot(btc_data.index, btc_data['BTC-USD'], label='Actual', color='blue')
    plt.axvline(x=btc_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')

    train_range = btc_data.index[time_step:train_size]
    test_range = btc_data.index[train_size:train_size + len(test_preds)]

    plt.plot(train_range, train_preds[:len(train_range)], label='Train Prediction', color='green')
    plt.plot(test_range, test_preds[:len(test_range)], label="Test Predictions", color='orange')

    future_index = pd.date_range(start=btc_data.index[-1], periods=prediction_ahead+1, freq='D')[1:]
    plt.plot(future_index, future_forecast, label=f'{prediction_ahead}-Day Forecast', color='red')
    plt.legend()
    st.pyplot(plt)
