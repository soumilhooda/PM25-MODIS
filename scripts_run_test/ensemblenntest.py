import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

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

# Define Ensemble Model
ensemble_input = Input(shape=(X_train_scaled.shape[1],))
ensemble_models = []

for _ in range(5):  # Number of ensemble models
    model = Sequential([
        Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dense(units=1)
    ])
    ensemble_models.append(model)

ensemble_output = layers.average([model(ensemble_input) for model in ensemble_models])
ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output)

# Train and evaluate Ensemble Model
ensemble_model.compile(optimizer='adam', loss='mean_squared_error')
ensemble_model.fit(X_train_scaled, y_train, epochs=3000, batch_size=25, verbose=10, callbacks=[PrintEpochProgress()])

predictions_train_ensemble = ensemble_model.predict(X_train_scaled)
predictions_test_ensemble = ensemble_model.predict(X_test_scaled)

mse_train_ensemble = mean_squared_error(y_train, predictions_train_ensemble)
mse_test_ensemble = mean_squared_error(y_test, predictions_test_ensemble)

print(f"Ensemble NN Train RMSE: {np.sqrt(mse_train_ensemble):.4f}")
print(f"Ensemble NN Test RMSE: {np.sqrt(mse_test_ensemble):.4f}")

# Scatter plots for train and test sets for Ensemble NN
plt.scatter(y_train, predictions_train_ensemble)
plt.title("Ensemble NN - Train Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

plt.scatter(y_test, predictions_test_ensemble)
plt.title("Ensemble NN - Test Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
