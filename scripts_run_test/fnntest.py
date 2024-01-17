import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import Callback

class PrintEpochProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} - {logs}")

# Load data
df = pd.read_csv("test.ascii", delim_whitespace=True, header=None)
df.columns = ['PM25', 'RAD2', 'RAD5', 'RAD8', 'RAD12', 'RAD14', 'RAD20', 'RAD22', 'RAD23', 'ANG1', 'ANG3', 'RF1', 'RF2', 'RF3', 'RF4', 'RF5', 'RF6', 'RF7', 'MON', 'ALAT', 'ALON', 'light', 'ndvi', 'TPW', 'SKT']

# Feature selection
X = df.drop(['PM25', 'MON', 'ALAT', 'ALON', 'light', 'ndvi', 'TPW', 'SKT', 'ANG1'], axis=1)
y = df[['PM25']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.values
y_test = y_test.values

# Simple Feed-Forward Neural Network Model
feedforward_model = Sequential([
    Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='linear')
])

# Compile and train Feed-Forward Neural Network Model
feedforward_model.compile(optimizer='adam', loss='mean_squared_error')
feedforward_model.fit(X_train_scaled, y_train, epochs=1000, batch_size=25, verbose=10, callbacks=[PrintEpochProgress()])

# Evaluate on both train and test sets
predictions_train_feedforward = feedforward_model.predict(X_train_scaled)
predictions_test_feedforward = feedforward_model.predict(X_test_scaled)

mse_train_feedforward = mean_squared_error(y_train, predictions_train_feedforward)
mse_test_feedforward = mean_squared_error(y_test, predictions_test_feedforward)

print(f"Feed-Forward Model Train RMSE: {np.sqrt(mse_train_feedforward):.4f}")
print(f"Feed-Forward Model Test RMSE: {np.sqrt(mse_test_feedforward):.4f}")

# Scatter plots for train and test sets for Feed-Forward Model
plt.scatter(y_train, predictions_train_feedforward)
plt.title("Feed-Forward Model - Train Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

plt.scatter(y_test, predictions_test_feedforward)
plt.title("Feed-Forward Model - Test Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
