import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras import layers  # Add this import
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')


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

# DilatedConvNetwork Model
dilated_conv_network_model = Sequential([
    Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Conv1D(filters=64, kernel_size=3, dilation_rate=3, activation='relu'),
    Flatten(),
    Dense(units=1)
])

# Compile and train DilatedConvNetwork Model
dilated_conv_network_model.compile(optimizer='adam', loss='mean_squared_error')
dilated_conv_network_model.fit(X_train_scaled[:, :, np.newaxis], y_train, epochs=3000, batch_size=25, verbose=10, callbacks=[PrintEpochProgress()])

# Evaluate on both train and test sets
predictions_train_dilated_conv_network = dilated_conv_network_model.predict(X_train_scaled[:, :, np.newaxis])
predictions_test_dilated_conv_network = dilated_conv_network_model.predict(X_test_scaled[:, :, np.newaxis])

mse_train_dilated_conv_network = mean_squared_error(y_train, predictions_train_dilated_conv_network)
mse_test_dilated_conv_network = mean_squared_error(y_test, predictions_test_dilated_conv_network)

print(f"DilatedConvNetwork Model Train RMSE: {np.sqrt(mse_train_dilated_conv_network):.4f}")
print(f"DilatedConvNetwork Model Test RMSE: {np.sqrt(mse_test_dilated_conv_network):.4f}")

# # Scatter plots for train and test sets
# plt.scatter(y_train, predictions_train_dilated_conv_network)
# plt.title("DilatedConvNetwork Model - Train Set")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()

# plt.scatter(y_test, predictions_test_dilated_conv_network)
# plt.title("DilatedConvNetwork Model - Test Set")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()
