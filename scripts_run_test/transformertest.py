from tensorflow.keras.layers import LayerNormalization, Input, Dropout, Conv1D, GlobalAveragePooling1D, Dense
from keras.layers import MultiHeadAttention, Reshape
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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

def transformer_model(units, num_heads, ff_dim, input_shape, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(2):  # Two transformer blocks
        # Reshape for Multi-Head Attention
        x_reshaped = Reshape((input_shape[0], 1))(x)
        x_reshaped = Reshape((1, input_shape[0]))(x_reshaped)  # Correct reshaping
        x_reshaped = Reshape((1, input_shape[0], 1))(x_reshaped)  # Additional reshaping
        x = MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)([x_reshaped, x_reshaped])
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + x_reshaped)

        y = Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
        y = Dropout(dropout)(y)
        y = Conv1D(filters=units, kernel_size=1)(y)
        x = LayerNormalization(epsilon=1e-6)(x + y)

    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="linear")(x)

    return Model(inputs=inputs, outputs=x)

# Create and compile Transformer Model
transformer_units = 64
transformer_num_heads = 2
transformer_ff_dim = 32
transformer_input_shape = (X_train_scaled.shape[1],)  # Input shape should be 2D
transformer_dropout = 0.1

transformer_model = transformer_model(
    units=transformer_units,
    num_heads=transformer_num_heads,
    ff_dim=transformer_ff_dim,
    input_shape=transformer_input_shape,
    dropout=transformer_dropout
)

transformer_model.compile(optimizer='adam', loss='mean_squared_error')
transformer_model.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=10, callbacks=[PrintEpochProgress()])

# Evaluate on both train and test sets
predictions_train_transformer = transformer_model.predict(X_train_scaled)
predictions_test_transformer = transformer_model.predict(X_test_scaled)

mse_train_transformer = mean_squared_error(y_train, predictions_train_transformer)
mse_test_transformer = mean_squared_error(y_test, predictions_test_transformer)

print(f"Transformer Model Train RMSE: {np.sqrt(mse_train_transformer):.4f}")
print(f"Transformer Model Test RMSE: {np.sqrt(mse_test_transformer):.4f}")

# Scatter plots for train and test sets for Transformer Model
plt.scatter(y_train, predictions_train_transformer)
plt.title("Transformer Model - Train Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

plt.scatter(y_test, predictions_test_transformer)
plt.title("Transformer Model - Test Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
