import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
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

# Define a function to create the Keras model
def create_keras_model():
    model = Sequential([
        Dense(units=256, activation='softplus', input_dim=X_train_scaled.shape[1]),
        Dense(units=128, activation='softplus'),
        Dense(units=64, activation='softplus'),
        Dense(units=32, activation='softplus'),
        Dense(units=1, activation='linear'),
        Lambda(lambda x: x * 5.0)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create a KerasRegressor using the wrapper
keras_estimator = KerasRegressor(build_fn=create_keras_model, epochs=50, batch_size=10, verbose=0)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Linear Regression
lr_model = LinearRegression()

# Support Vector Regressor
svr_model = SVR(kernel='rbf')

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Stacking Model
stacked_model = StackingRegressor(
    estimators=[
        ('pnn', keras_estimator),
        ('rf', rf_model),
        # ('lr', lr_model),
        # ('svr', svr_model),
        # ('gb', gb_model)
    ],
    final_estimator=lr_model,
    cv=3
)

# Fit Stacking Model
stacked_model.fit(X_train_scaled, y_train)

# Predict using Stacking Model
predictions_stacked = stacked_model.predict(X_test_scaled)
mse_test_stacked = mean_squared_error(y_test, predictions_stacked)

print(f"Stacked Model Test RMSE: {np.sqrt(mse_test_stacked):.4f}")

