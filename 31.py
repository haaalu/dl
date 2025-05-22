# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# %%
df=pd.read_csv("GOOGL.csv")

# %%
df.head()

# %%
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close']].values

# %%
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# %%
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# %%
SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# %%
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %%
def train_lstm(learning_rate, epochs):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, history.history['loss'], mse, preds

# %%
configs = [(0.001, 20), (0.0001, 50)]
results = {}

# %%
for lr, ep in configs:
    print(f"Training with learning_rate={lr}, epochs={ep}")
    model, loss, mse, preds = train_lstm(lr, ep)
    results[(lr, ep)] = {'loss': loss, 'mse': mse, 'preds': preds}

# %%
plt.figure(figsize=(12, 6))
for key, value in results.items():
    plt.plot(value['loss'], label=f'lr={key[0]}, epochs={key[1]}')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# %%
print("\nFinal Test MSEs:")
for key, value in results.items():
    print(f"LR={key[0]}, Epochs={key[1]} → Test MSE: {value['mse']:.6f}")

# %%
for key, value in results.items():
    preds = scaler.inverse_transform(value['preds'])
    actual = scaler.inverse_transform(y_test)
    
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual Price', color='blue')
    plt.plot(preds, label='Predicted Price', color='red')
    plt.title(f"Prediction vs Actual (LR={key[0]}, Epochs={key[1]})")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()   

# %%



