import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
import tensorflow as tfp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

# Custom callback to print every 10th epoch
class PrintEpochProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} - {logs}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Assuming X_train_scaled.shape[1] is the input dimension
input_dim = X_train_scaled.shape[1]

# Define the model
model = Sequential([
    Input(shape=(input_dim,)),
    layers.Flatten(),
    tfpl.DenseVariational(units=256,
                          make_posterior_fn=lambda _: tfd.Independent(tfd.Normal(loc=tf.Variable(tf.zeros([input_dim, 256])),
                                                                             scale=tfp.util.TransformedVariable(tf.ones([input_dim, 256]), tfp.bijectors.Softplus())),
                                                                     reinterpreted_batch_ndims=1),
                          make_prior_fn=tfpl.default_multivariate_normal_fn,
                          kl_weight=1/X_train.shape[0],
                          activation='softplus'),
    tfpl.DenseVariational(units=128,
                          make_posterior_fn=lambda _: tfd.Independent(tfd.Normal(loc=tf.Variable(tf.zeros([256, 128])),
                                                                             scale=tfp.util.TransformedVariable(tf.ones([256, 128]), tfp.bijectors.Softplus())),
                                                                     reinterpreted_batch_ndims=1),
                          make_prior_fn=tfpl.default_multivariate_normal_fn,
                          kl_weight=1/X_train.shape[0],
                          activation='softplus'),
    tfpl.DenseVariational(units=64,
                          make_posterior_fn=lambda _: tfd.Independent(tfd.Normal(loc=tf.Variable(tf.zeros([128, 64])),
                                                                             scale=tfp.util.TransformedVariable(tf.ones([128, 64]), tfp.bijectors.Softplus())),
                                                                     reinterpreted_batch_ndims=1),
                          make_prior_fn=tfpl.default_multivariate_normal_fn,
                          kl_weight=1/X_train.shape[0],
                          activation='softplus'),
    tfpl.DenseVariational(units=32,
                          make_posterior_fn=lambda _: tfd.Independent(tfd.Normal(loc=tf.Variable(tf.zeros([64, 32])),
                                                                             scale=tfp.util.TransformedVariable(tf.ones([64, 32]), tfp.bijectors.Softplus())),
                                                                     reinterpreted_batch_ndims=1),
                          make_prior_fn=tfpl.default_multivariate_normal_fn,
                          kl_weight=1/X_train.shape[0],
                          activation='softplus'),
    tfpl.DenseVariational(units=1,
                          make_posterior_fn=lambda _: tfd.Independent(tfd.Normal(loc=tf.Variable(tf.zeros([32, 1])),
                                                                             scale=tfp.util.TransformedVariable(tf.ones([32, 1]), tfp.bijectors.Softplus())),
                                                                     reinterpreted_batch_ndims=1),
                          make_prior_fn=tfpl.default_multivariate_normal_fn,
                          kl_weight=1/X_train.shape[0])
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=lambda y, model: -model.log_prob(y))  # Negative log-likelihood as the loss function

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[PrintEpochProgress()])

# Evaluate the model
neg_log_likelihood = -model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Negative Log-Likelihood: {neg_log_likelihood:.4f}")
