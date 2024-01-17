import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
# Custom callback to print every 10th epoch
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

# Residual Block for ResNet
def residual_block(x, units):
    y = Dense(units, activation='relu')(x)
    y = Dense(units, activation=None)(y)  # Linear activation for identity mapping

    # Add a Reshape layer to match the shapes if necessary
    if x.shape[-1] != units:
        x = Dense(units, activation=None)(x)
    
    return Add()([x, y])

# ResNet Model
resnet_input = Input(shape=(X_train_scaled.shape[1],))
resnet_units = 64

resnet_output = residual_block(resnet_input, resnet_units)
resnet_output = Dense(units=1)(resnet_output)

resnet_model = Model(inputs=resnet_input, outputs=resnet_output)

# Compile and train ResNet Model
resnet_model.compile(optimizer='adam', loss='mean_squared_error')
resnet_model.fit(X_train_scaled, y_train, epochs=3000, batch_size=25, verbose=10, callbacks=[PrintEpochProgress()])

# Evaluate on both train and test sets
predictions_train_resnet = resnet_model.predict(X_train_scaled)
predictions_test_resnet = resnet_model.predict(X_test_scaled)

mse_train_resnet = mean_squared_error(y_train, predictions_train_resnet)
mse_test_resnet = mean_squared_error(y_test, predictions_test_resnet)

print(f"ResNet Model Train RMSE: {np.sqrt(mse_train_resnet):.4f}")
print(f"ResNet Model Test RMSE: {np.sqrt(mse_test_resnet):.4f}")

# # Scatter plots for train and test sets for ResNet
# plt.scatter(y_train, predictions_train_resnet)
# plt.title("ResNet Model - Train Set")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()

# plt.scatter(y_test, predictions_test_resnet)
# plt.title("ResNet Model - Test Set")
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.show()
