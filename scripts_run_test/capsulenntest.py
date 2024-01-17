import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, Add
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import keras

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
    return Add()([x, y])

# Primary Capsule Layer
def primary_capsules(x, num_capsules, dim_capsule, routing_iters):
    # Expand dimensions for sequence_length
    x_expanded =  keras.layers.Reshape(target_shape=(int(x.shape[1]), 1))(x)
    
    # Apply Conv1D to the expanded tensor
    conv =  keras.layers.Conv1D(num_capsules * dim_capsule, kernel_size=3, activation='relu', padding='valid')(x_expanded)
    
    # Reshape back to 3D tensor
    reshaped =  keras.layers.Reshape(target_shape=(int(x.shape[1]) // num_capsules, num_capsules, dim_capsule))(conv)
    
    # Sum along the last dimension
    return keras.layers.Lambda(lambda z: tensorflow.keras.backend.sum(z, axis=-2), output_shape=(num_capsules, dim_capsule))(reshaped)

# Capsule Layer
def capsule_layer(x, num_capsules, dim_capsule, routing_iters):
    caps =  keras.layers.Lambda(lambda z: z / tensorflow.keras.backend.sqrt(tensorflow.keras.backend.sum(tensorflow.keras.backend.square(z), axis=-1, keepdims=True) + 1e-9))(x)
    routing_weights =  keras.layers.Dense(num_capsules * dim_capsule, activation='softmax')(caps)
    reshaped =  keras.layers.Reshape(target_shape=(int(routing_weights.shape[1]) // num_capsules, num_capsules, dim_capsule))(routing_weights)
    outputs =  keras.layers.Lambda(lambda z: layers.keras.backend.sum(z[0] * z[1], axis=-2), output_shape=(num_capsules, dim_capsule))([x, reshaped])

    for _ in range(routing_iters - 1):
        caps =  keras.layers.Lambda(lambda z: z / tensorflow.keras.backend.sqrt(tensorflow.keras.backend.sum(tensorflow.keras.backend.square(z), axis=-1, keepdims=True) + 1e-9))(outputs)
        routing_weights = layers.Dense(num_capsules * dim_capsule, activation='softmax')(caps)
        reshaped =  keras.layers.Reshape(target_shape=(int(routing_weights.shape[1]) // num_capsules, num_capsules, dim_capsule))(routing_weights)
        outputs =  keras.layers.Lambda(lambda z: layers.keras.backend.sum(z[0] * z[1], axis=-2), output_shape=(num_capsules, dim_capsule))([x, reshaped])

    return outputs

# Capsule Network Model
capsule_input = Input(shape=(X_train_scaled.shape[1],))
capsule_units = 64
capsule_num_capsules = 8
capsule_dim_capsule = 8
capsule_routing_iters = 3

capsule_primary = primary_capsules(capsule_input, capsule_num_capsules, capsule_dim_capsule, capsule_routing_iters)
capsule_output = capsule_layer(capsule_primary, capsule_num_capsules, capsule_dim_capsule, capsule_routing_iters)
capsule_output =  keras.layers.Flatten()(capsule_output)
capsule_output =  keras.layers.Dense(units=1, activation='linear')(capsule_output)

capsule_model = Model(inputs=capsule_input, outputs=capsule_output)

# Compile and train Capsule Network Model
capsule_model.compile(optimizer='adam', loss='mean_squared_error')
capsule_model.fit(X_train_scaled, y_train, epochs=5, batch_size=25, verbose=10, callbacks=[PrintEpochProgress()])

# Evaluate on both train and test sets
predictions_train_capsule = capsule_model.predict(X_train_scaled)
predictions_test_capsule = capsule_model.predict(X_test_scaled)

mse_train_capsule = mean_squared_error(y_train, predictions_train_capsule)
mse_test_capsule = mean_squared_error(y_test, predictions_test_capsule)

print(f"Capsule Model Train RMSE: {np.sqrt(mse_train_capsule):.4f}")
print(f"Capsule Model Test RMSE: {np.sqrt(mse_test_capsule):.4f}")

# Scatter plots for train and test sets
plt.scatter(y_train, predictions_train_capsule)
plt.title("Capsule Model - Train Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

plt.scatter(y_test, predictions_test_capsule)
plt.title("Capsule Model - Test Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
