import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras import layers
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

# Define regression models
regression_models = {
    # 'FNN': Sequential([
    #     Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]),
    #     Dense(units=1)
    # ]),
    'CNN': Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units=1)
    ]),
    'RNN': Sequential([
        SimpleRNN(units=64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        Dense(units=1)
    ]),
    'LSTM': Sequential([
        LSTM(units=64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        Dense(units=1)
    ]),
    'GRU': Sequential([
        GRU(units=64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        Dense(units=1)
    ]),
        'ProbabilisticNN': Sequential([
        Dense(units=64, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dense(units=1),
        layers.Lambda(lambda x: x * 5.0)  
    ]),
    'DilatedConvNetwork': Sequential([
        layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu'),
        Flatten(),
        Dense(units=1)
    ])
}
class PrintEpochProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} - {logs}")

# Train and evaluate regression models
for model_name, model in regression_models.items():
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Modify the following line to print every 10th epoch
    model.fit(X_train_scaled[:, :, np.newaxis], y_train, epochs=50, batch_size=10, verbose=10, callbacks=[PrintEpochProgress()])

    predictions_train = model.predict(X_train_scaled[:, :, np.newaxis])
    predictions_test = model.predict(X_train_scaled[:, :, np.newaxis])

    mse_train = mean_squared_error(y_train, predictions_train)
    mse_test = mean_squared_error(y_test, predictions_test)

    print(f"{model_name} Train RMSE: {np.sqrt(mse_train):.4f}")
    print(f"{model_name} Test RMSE: {np.sqrt(mse_test):.4f}")

    # Scatter plots for train and test sets
    plt.scatter(y_train, predictions_train)
    plt.title(f"{model_name} - Train Set")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

    plt.scatter(y_test, predictions_test)
    plt.title(f"{model_name} - Test Set")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

    print("\n")


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
ensemble_model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)

predictions_train_ensemble = ensemble_model.predict(X_train_scaled)
predictions_test_ensemble = ensemble_model.predict(X_test_scaled)

mse_train_ensemble = mean_squared_error(y_train, predictions_train_ensemble)
mse_test_ensemble = mean_squared_error(y_test, predictions_test_ensemble)

print(f"EnsembleFNN Train RMSE: {np.sqrt(mse_train_ensemble):.4f}")
print(f"EnsembleFNN Test RMSE: {np.sqrt(mse_test_ensemble):.4f}")

# Scatter plots for train and test sets for EnsembleFNN
plt.scatter(y_train, predictions_train_ensemble)
plt.title("EnsembleFNN - Train Set")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

plt.scatter(y_test, predictions_test_ensemble)
plt.title("EnsembleFNN - Test Set")  
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
