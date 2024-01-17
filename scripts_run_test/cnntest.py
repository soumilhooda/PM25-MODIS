import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import Callback

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

# CNN Model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    Flatten(),
    
    Dense(units=256, activation='relu'),
    Dense(units=128, activation='relu'),
    
    Dense(units=1)
])

# Custom callback to print every 10th epoch
class PrintEpochProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} - {logs}")

# Compile and train CNN Model
cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.fit(X_train_scaled[:, :, np.newaxis], y_train, epochs=3000, batch_size=25, verbose=10, callbacks=[PrintEpochProgress()])

# Evaluate on both train and test sets
predictions_train_cnn = cnn_model.predict(X_train_scaled[:, :, np.newaxis])
predictions_test_cnn = cnn_model.predict(X_test_scaled[:, :, np.newaxis])

mse_train_cnn = mean_squared_error(y_train, predictions_train_cnn)
mse_test_cnn = mean_squared_error(y_test, predictions_test_cnn)

print(f"CNN Model Train RMSE: {np.sqrt(mse_train_cnn):.4f}")
print(f"CNN Model Test RMSE: {np.sqrt(mse_test_cnn):.4f}")

# # Scatter plots for train and test sets
# plt.scatter(y_train, predictions_train_cnn)
# plt.title("CNN Model - Train Set")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()

# plt.scatter(y_test, predictions_test_cnn)
# plt.title("CNN Model - Test Set")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()
